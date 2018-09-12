#include "image.h"
#include "kmeans.h"
#include "types.h"

#define LODEPNG_COMPILE_DISK

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
  std::vector<uint8_t> bytes;
  uint32_t width;
  uint32_t height;
  unsigned error = lodepng::decode(bytes, width, height, filename);

  img.SetDimensions(width, height);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint32_t i = 4 * (width * y + x);
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
                      uint32_t num_files = file_names.size();
                      uint32_t block_size = num_files / num_threads + 1;
                      uint32_t start = tid * block_size;
                      uint32_t end = std::min(start + block_size, num_files);
                      for (int j = start; j < end; ++j)
                        DecodePng(file_names[j].c_str(), images[j]);
                    },
                    i);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
}

void ComputeAverageImage(const std::vector<image::Image>& images,
                         const std::vector<int> indices,
                         image::Image& average) {
  uint32_t width = images[0].width(), height = images[0].height(),
           num_threads = 8;
  std::vector<float> sum(width * height * 3, 0.0f);
  uint32_t count = 0;
  for (int i = 0; i < indices.size(); ++i) {
    const image::Image& img = images[indices[i]];
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        image::Pixel p = img(x, y);
        int i = 3 * (y * width + x);
        sum[i] += static_cast<float>(p.r);
        sum[i + 1] += static_cast<float>(p.g);
        sum[i + 2] += static_cast<float>(p.b);
      }
    }
    ++count;
  }
  for (int i = 0; i < width * height * 3; ++i)
    sum[i] /= static_cast<float>(count);
  average.SetDimensions(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int i = 3 * (y * width + x);
      average(x, y) =
          image::Pixel(static_cast<uint8_t>(std::round(sum[i])),
                       static_cast<uint8_t>(std::round(sum[i + 1])),
                       static_cast<uint8_t>(std::round(sum[i + 2])));
    }
  }
  lodepng::encode("average.png", average.GetBytes(), width, height,
                  LodePNGColorType::LCT_RGB, 8);
}

void GetTrainingData(const std::vector<image::Image>& images,
                     std::vector<int>& indices, int num_centers,
                     const std::vector<int>& labels, image::Image& average,
                     float** train_data, int* train_data_dim1,
                     int* train_data_dim2, float** train_labels,
                     int* train_labels_dim1, int* train_labels_dim2,
                     std::vector<int>& batch_sizes) {
  // std::cout << "GetTrainingData:images.size()  = " << images.size() << "\n";
  // std::cout << "GetTrainingData:indices.size() = " << indices.size() << "\n";
  // std::cout << "GetTrainingData:min_index = "
  //<< *std::min_element(indices.begin(),
  // indices.begin() + indices.size()) << "\n";
  // std::cout << "GetTrainingData:max_index = "
  //<< *std::max_element(indices.begin(),
  // indices.begin() + indices.size()) << "\n";
  const uint32_t data_size = sizeof(TestData) / sizeof(float);
  const uint32_t label_size = sizeof(PixelData) / sizeof(float);
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
  // std::cout << "GetTrainingData:width  = " << width << "\n";
  // std::cout << "GetTrainingData:height  = " << height << "\n";
  // std::cout << "GetTrainingData:labels.size= " << labels.size() << "\n";
  // Count pixels per cluster.
  for (int i = 0; i < labels.size(); ++i) ++cluster_sizes[labels[i]];
  batch_sizes.clear();
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
          &num_pixels,
          &labels,
          &cluster_counts,
          &cluster_sizes,
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

          uint32_t sample_size = indices.size(),
                   block_size = sample_size / num_threads + 1,
                   start = tid * block_size,
                   end = std::min(start + block_size, sample_size),
                   num_pixels = width * height;
          float* train_data_pos = *train_data + data_size * start * num_pixels;
          float* train_labels_pos =
              *train_labels + label_size * start * num_pixels;
          for (int i = start; i < end; ++i) {
            const image::Image& img = images[indices[i]];
            for (int j = 0; j < labels.size(); ++j) {
              uint32_t center = labels[j], x = j % width, y = j / width;
              image::Pixel p = img(x, y), a = average(x, y);
              *reinterpret_cast<TestData*>(train_data_pos) =
                  TestData(x, y, indices[i], a, width, height, images.size(),
                           ::PixelConversion::DefaultConversion());
              *reinterpret_cast<PixelData*>(train_labels_pos) =
                  PixelData(p, ::PixelConversion::DefaultConversion());
              train_data_pos += data_size;
              train_labels_pos += label_size;
            }
          }
        },
        i);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
}

void PickRandomIndices(uint32_t total, uint32_t amount,
                       std::vector<int>& indices, std::vector<int>& order) {
  std::cout << "PickRandomIndices\n";
  std::cout << "total = " << total << "\n";
  std::cout << "amount = " << amount << "\n";
  auto cmp = [](std::pair<int, float> left, std::pair<int, float> right) {
    return left.second < right.second;
  };
  std::priority_queue<std::pair<int, float>, std::deque<std::pair<int, float> >,
                      decltype(cmp)> q(cmp);
#define USE_STD_UNIFORM_RANDOM_DEVICE 0
#if USE_STD_UNIFORM_RANDOM_DEVICE
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (int i = 0; i < total; ++i) q.push(std::make_pair(i, dis(gen)));
#else
  int seed = 12345;
  srand(seed);
  for (int i = 0; i < total; ++i)
    q.push(std::make_pair(i, static_cast<float>(rand()) / RAND_MAX));
#endif
  indices.clear();
  for (int i = 0; i < amount; ++i) {
    indices.push_back(q.top().first);
    q.pop();
  }
  auto cmp2 = [](int a, int b) { return a < b; };
  std::cout << "min_index = " << *std::min_element(
                                      indices.begin(),
                                      indices.begin() + indices.size()) << "\n";
  std::cout << "max_index = " << *std::max_element(
                                      indices.begin(),
                                      indices.begin() + indices.size()) << "\n";
  std::sort(indices.begin(), indices.end(), cmp2);
  order.resize(total, -1);
  std::cout << "indices = ";
  for (int i = 0; i < indices.size(); ++i) std::cout << indices[i] << " ";
  std::cout << "\n";
  for (int i = 0; i < indices.size(); ++i) order[indices[i]] = i;
  std::cout << "total = " << total << "\n";
  std::cout << "order= ";
  for (int i = 0; i < order.size(); ++i) std::cout << order[i] << " ";
  std::cout << "\n";
}

void KmeansDataAndLabels(
    const std::string& directory, int num_centers, int& width, int& height,
    float** training_data, int* training_data_dim1, int* training_data_dim2,
    float** training_labels, int* training_labels_dim1,
    int* training_labels_dim2, float** average_img, int* average_dim1,
    int* average_dim2, int* average_dim3, std::vector<int>& indices,
    std::vector<int>& order, std::vector<glm::vec2>& centers,
    std::vector<int>& labels, std::vector<int>& batch_sizes) {
  std::cout << "KmeansDataAndLabels\n";
  std::vector<image::Image> images;
  // Load images.
  std::cout << "directory = " << directory << "\n";
  LoadImages(directory, images);
  width = (images.size() > 0 ? images[0].width() : -1);
  height = (images.size() > 0 ? images[0].height() : -1);
  std::cout << "width = " << width << " height = " << height << "\n";
  std::cout << "#images = " << images.size() << "\n";
  if (images.size() < 1) return;
  image::Image average;
  // Pick random sample to use for training.
  uint32_t sample_size = static_cast<uint32_t>(0.70 * images.size());
  std::cout << "pick indices\n";
  std::cout << "sample_size = " << sample_size << "\n";
  if (indices.empty())
    PickRandomIndices(images.size(), sample_size, indices, order);
  std::cout << "average\n";
  // Compute average.
  ComputeAverageImage(images, indices, average);
  // Return average img to user in normalized form.
  // std::cout << "write out average image\n";
  *average_img = new float[width * height * 3];
  *average_dim1 = height;
  *average_dim2 = width;
  *average_dim3 = 3;
  const image::Pixel* pixels =
      reinterpret_cast<const image::Pixel*>(average.GetBytes());
  int num_pixels = width * height;
  std::transform(pixels, pixels + num_pixels,
                 reinterpret_cast<PixelData*>(*average_img),
                 [](const image::Pixel& p) {
    return PixelData(p.r / 255.0f, p.g / 255.0f, p.b / 255.0f);
  });
  // Run kmeans.
  centers.resize(num_centers);
  labels.resize(width * height);
  std::cout << "kmeans\n";
  kmeans(width, height, centers, labels);
  // Get labels and data.
  GetTrainingData(images, indices, num_centers, labels, average, training_data,
                  training_data_dim1, training_data_dim2, training_labels,
                  training_labels_dim1, training_labels_dim2, batch_sizes);
}
