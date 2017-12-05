#include "image.h"
#include "kmeans.h"

#include "lodepng.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/color_space.hpp>
#include <glm/gtx/norm.hpp>

void DecodePng(const char* filename, image::Image& img) {
  std::vector<unsigned char> bytes;
  uint32_t width;
  uint32_t height;
  unsigned error = lodepng::decode(bytes, width, height, filename);

  img.SetDimensions(width, height);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint32_t i = 4 * y + x;
      img(x, y) = image::Pixel(bytes[i], bytes[i + 1], bytes[i + 2]);
    }
  }
  if (error)
    std::cout << "decoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;
}

bool HasSuffix(const std::string& s, const std::string& suffix) {
  return (s.size() >= suffix.size()) &&
         equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

void LoadImages(const std::string& dirname, std::vector<image::Image>& images) {
  DIR* dir = opendir(dirname.c_str());
  if (!dir) return;
  dirent* entry = nullptr;
  images.clear();
  std::vector<std::string> file_names;
  // Load up all the filenames.
  while ((entry = readdir(dir)) != nullptr) {
    if (HasSuffix(entry->d_name, ".png"))
      file_names.push_back(dirname + "/" + std::string(entry->d_name));
  }
  closedir(dir);
  // Load all images threaded.
  images.resize(file_names.size(), image::Image());
  uint32_t num_threads = 8;
  std::vector<std::thread> threads(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads[i] =
        std::thread([&num_threads, &file_names, &images ](int tid)->void {
                      uint32_t block_size = file_names.size() / num_threads;
                      uint32_t start = tid * block_size;
                      uint32_t end =
                          std::min(start + block_size,
                                   static_cast<uint32_t>(file_names.size()));
                      for (int j = start; j < end; ++j)
                        DecodePng(file_names[j].c_str(), images[j]);
                    },
                    i);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
}

void ComputeAverageImage(const std::vector<image::Image>& images,
                         image::Image& average) {
  uint32_t width = images[0].width(), height = images[0].height(),
           num_threads = 8;
  std::vector<float> sum(width * height * 3);
  for (int i = 0; i < images.size(); ++i) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        image::Pixel p = images[i](x, y);
        sum[y * width + x] += static_cast<float>(p.r);
        sum[y * width + x + 1] += static_cast<float>(p.g);
        sum[y * width + x + 2] += static_cast<float>(p.b);
      }
    }
  }
  for (int i = 0; i < width * height * 3; ++i)
    sum[i] /= static_cast<float>(images.size());
  average.SetDimensions(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image::Pixel p;
      p.r = static_cast<uint8_t>(std::round(sum[y * width + x]));
      p.g = static_cast<uint8_t>(std::round(sum[y * width + x + 1]));
      p.b = static_cast<uint8_t>(std::round(sum[y * width + x + 2]));
    }
  }
}

void GetTrainingData(const std::vector<image::Image>& images,
                     std::vector<int>& indices, int num_centers,
                     const std::vector<int>& labels, image::Image& average,
                     float** train_data, int* train_data_dim1,
                     int* train_data_dim2, float** train_labels,
                     int* train_labels_dim1, int* train_labels_dim2,
                     std::vector<int>& batch_sizes) {
  const uint32_t coord_dim = 2;
  const uint32_t light_dim = 1;
  const uint32_t pixel_dim = 3;
  const uint32_t data_size = coord_dim + light_dim + pixel_dim;
  const uint32_t label_size = pixel_dim;
  uint32_t width = images[0].width(), height = images[0].height(),
           num_threads = 8;
  uint32_t sample_size = indices.size();
  uint32_t num_pixels = sample_size * width * height;
  *train_data_dim1 = num_pixels;
  *train_data_dim2 = data_size;
  *train_data = new float[(*train_data_dim1) * (*train_data_dim2)];
  *train_labels_dim1 = num_pixels;
  *train_labels_dim2 = label_size;
  *train_labels = new float[(*train_labels_dim1) * (*train_labels_dim2)];
  std::vector<uint32_t> cluster_sizes(num_centers, 0);
  std::vector<uint32_t> cluster_counts(num_centers, 0);
  // Count pixels per cluster.
  for (int i = 0; i < labels.size(); ++i) ++cluster_sizes[labels[i]];
  batch_sizes.clear();
  std::copy(cluster_sizes.begin(), cluster_sizes.end(),
            std::back_inserter(cluster_counts));
  // Copy out the batch sizes.
  std::copy(cluster_sizes.begin(), cluster_sizes.end(),
            std::back_inserter(batch_sizes));
  // Convert to total counts.
  for (int i = 1; i < cluster_counts.size(); ++i)
    cluster_counts[i] = cluster_counts[i - 1] + cluster_sizes[i - 1];
  std::vector<std::thread> threads(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads[i] = std::thread(
        [
          &num_threads,
          &num_centers,
          &labels,
          &cluster_counts,
          &data_size,
          &label_size,
          &width,
          &height,
          &indices,
          &images,
          &average,
          &train_data,
          &train_labels
        ](int tid)
             ->void {
          uint32_t block_size = indices.size() / num_threads,
                   start = tid * block_size,
                   end = std::min(start + block_size,
                                  static_cast<uint32_t>(images.size()));
          for (int i = start; i < end; ++i) {
            const image::Image& img = images[indices[i]];
            std::vector<uint32_t> pixel_counts(num_centers, 0);
            for (int j = 0; j < labels.size(); ++j) {
              uint32_t center = labels[j], x = j % width, y = j / width,
                       count = cluster_counts[center] + pixel_counts[center],
                       k = count * data_size, l = count * label_size;
              image::Pixel p = img(x, y), a = average(x, y);
              (*train_data)[k] = x / static_cast<float>(width);
              (*train_data)[k + 1] = y / static_cast<float>(height);
              (*train_data)[k + 2] =
                  indices[i] / static_cast<float>(images.size());
              (*train_data)[k + 3] = a.r / static_cast<float>(255.0);
              (*train_data)[k + 4] = a.g / static_cast<float>(255.0);
              (*train_data)[k + 5] = a.b / static_cast<float>(255.0);
              (*train_labels)[l] = p.r / static_cast<float>(255.0);
              (*train_labels)[l + 1] = p.g / static_cast<float>(255.0);
              (*train_labels)[l + 2] = p.b / static_cast<float>(255.0);
              ++pixel_counts[center];
            }
          }
        },
        i);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
}

void PickRandomIndices(uint32_t total, uint32_t amount,
                       std::vector<int>& indices) {
  auto cmp = [](std::pair<int, float> left, std::pair<int, float> right) {
    return left.second < right.second;
  };
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::priority_queue<std::pair<int, float>, std::deque<std::pair<int, float>>,
                      decltype(cmp)> q(cmp);
  for (int i = 0; i < total; ++i) q.push(std::make_pair(i, dis(gen)));
  indices.clear();
  for (int i = 0; i < amount; ++i) {
    indices.push_back(q.top().first);
    q.pop();
  }
  auto cmp2 = [](int a, int b) { return a < b; };
  std::sort(indices.begin(), indices.end(), cmp2);
}

// void KmeansDataAndLabels(const std::string& directory, int num_centers,
// int& width, int& height, float** training_data,
// int* training_data_dim1, int* training_data_dim2,
// float** training_labels, int* training_labels_dim1,
// int* training_labels_dim2,
// std::vector<int>& indices,
// std::vector<glm::vec2>& centers,
// std::vector<int>& labels,
// std::vector<int>& batch_sizes);
void KmeansDataAndLabels(const std::string& directory, int num_centers,
                         int& width, int& height, float** training_data,
                         int* training_data_dim1, int* training_data_dim2,
                         float** training_labels, int* training_labels_dim1,
                         int* training_labels_dim2, std::vector<int>& indices,
                         std::vector<glm::vec2>& centers,
                         std::vector<int>& labels,
                         std::vector<int>& batch_sizes) {
  std::vector<image::Image> images;
  // Load images.
  LoadImages(directory, images);
  width = (images.size() > 0 ? images[0].width() : -1);
  height = (images.size() > 0 ? images[0].height() : -1);
  if (images.size() < 1) return;
  image::Image average;
  // Compute average.
  ComputeAverageImage(images, average);
  // Pick random sample to use for training.
  uint32_t sample_size = static_cast<uint32_t>(0.70 * images.size());
  if (indices.empty()) PickRandomIndices(images.size(), sample_size, indices);
  // Run kmeans.
  centers.resize(num_centers);
  labels.resize(width * height);
  kmeans(width, height, centers, labels);
  // Get labels and data.
  GetTrainingData(images, indices, num_centers, labels, average, training_data,
                  training_data_dim1, training_data_dim2, training_labels,
                  training_labels_dim1, training_labels_dim2, batch_sizes);
}
