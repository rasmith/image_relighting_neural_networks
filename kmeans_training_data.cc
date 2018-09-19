#include "image.h"
#include "kmeans.h"
#include "logger.h"
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
#include <unordered_map>
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
  std::vector<double> sum(width * height * 3, 0.0);
  uint32_t count = 0;
  for (int i = 0; i < indices.size(); ++i) {
    const image::Image& img = images[indices[i]];
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        image::Pixel p = img(x, y);
        int i = 3 * (y * width + x);
        sum[i] += static_cast<double>(p.r);
        sum[i + 1] += static_cast<double>(p.g);
        sum[i + 2] += static_cast<double>(p.b);
      }
    }
    ++count;
  }
  for (int i = 0; i < width * height * 3; ++i)
    sum[i] /= static_cast<double>(count);
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
                     double** train_data, int* train_data_dim1,
                     int* train_data_dim2, double** train_labels,
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
  const uint32_t data_size = sizeof(TestData) / sizeof(double);
  const uint32_t label_size = sizeof(PixelData) / sizeof(double);
  uint32_t width = images[0].width(), height = images[0].height(),
           num_threads = 8;
  uint32_t sample_size = indices.size();
  uint32_t num_pixels = width * height;
  uint32_t total_pixels = sample_size * num_pixels;
  *train_data_dim1 = total_pixels;
  *train_data_dim2 = data_size;
  *train_data = new double[(*train_data_dim1) * (*train_data_dim2)];
  *train_labels_dim1 = total_pixels;
  *train_labels_dim2 = label_size;
  *train_labels = new double[(*train_labels_dim1) * (*train_labels_dim2)];
  std::vector<uint32_t> cluster_sizes(num_centers, 0);
  std::vector<uint32_t> cluster_offsets(num_centers, 0);
  // std::cout << "GetTrainingData:width  = " << width << "\n";
  // std::cout << "GetTrainingData:height  = " << height << "\n";
  // std::cout << "GetTrainingData:labels.size= " << labels.size() << "\n";
  // Count pixels per cluster.
  for (int i = 0; i < labels.size(); ++i) ++cluster_sizes[labels[i]];
  batch_sizes.clear();
  // Copy out the batch sizes.
  std::copy(cluster_sizes.begin(), cluster_sizes.end(),
            std::back_inserter(batch_sizes));
  for (int i = 0; i < cluster_sizes.size(); ++i)
    std::cout << cluster_sizes[i] << " ";
  std::cout << "\n";
  LOG(STATUS) << "Compute cluster offsets.\n";
  // Convert to total counts.
  for (int i = 1; i < cluster_offsets.size(); ++i)
    cluster_offsets[i] = cluster_offsets[i - 1] + cluster_sizes[i - 1];

  // Compute starts, ends, and pixel offsets for each thread.
  std::vector<int> starts(num_threads, 0);
  std::vector<int> ends(num_threads, 0);
  std::vector<int> pixel_offsets(num_threads, 0);
  std::vector<int> pixel_counts(num_threads, 0);
  for (int i = 0; i < num_threads; ++i) {
    int block_size = num_centers / num_threads + 1;
    starts[i] = i * block_size;
    ends[i] = std::min(starts[i] + block_size, num_centers);
    assert(ends[i] >= 0 && ends[i] <= num_centers);
    assert(starts[i] >= 0 && starts[i] <= num_centers);
    for (int j = starts[i]; j < ends[i]; ++j)
      pixel_counts[i] += cluster_sizes[j];
  }
  for (int i = 1; i < num_threads; ++i)
    pixel_offsets[i] += pixel_offsets[i - 1] + pixel_counts[i - 1];
  for (int i = 1; i < num_threads; ++i)
    assert(pixel_offsets[i] >= 0 &&
           pixel_offsets[i] < num_pixels * indices.size());
  // Compute a list of pixels for each cluster.
  std::unordered_map<int, std::vector<int>> cluster_to_pixels_map;
  for (int i = 0; i < num_pixels; ++i) {
    int cluster_id = labels[i];
    if (cluster_to_pixels_map.find(cluster_id) == cluster_to_pixels_map.end())
      cluster_to_pixels_map.insert(
          std::make_pair(cluster_id, std::vector<int>()));
    cluster_to_pixels_map.find(cluster_id)->second.push_back(i);
    assert(cluster_to_pixels_map.find(cluster_id)->second.size() <=
           cluster_sizes[cluster_id]);
  }
  std::vector<std::thread> threads(num_threads);
  LOG(STATUS) << "Output the training data and labels.\n";
  for (int i = 0; i < num_threads; ++i) {
    threads[i] =
        std::thread([
                      &width,
                      &height,
                      &num_pixels,
                      &starts,
                      &ends,
                      &pixel_offsets,
                      &cluster_to_pixels_map,
                      &indices,
                      &images,
                      &average,
                      &train_data,
                      &cluster_sizes,
                      &train_labels
                    ](int tid)
                         ->void {
                      TestData* train_data_begin =
                          reinterpret_cast<TestData*>(*train_data);
                      TestData* train_data_pos =
                          train_data_begin + pixel_offsets[tid];
                      PixelData* train_labels_begin =
                          reinterpret_cast<PixelData*>(*train_labels);
                      PixelData* train_labels_pos =
                          train_labels_begin + pixel_offsets[tid];
                      for (int i = starts[tid]; i < ends[tid]; ++i) {
                        const std::vector<int>& pixels =
                            cluster_to_pixels_map.find(i)->second;
                        assert(pixels.size() == cluster_sizes[i]);
                        for (int j = 0; j < indices.size(); ++j) {
                          for (int k = 0; k < pixels.size(); ++k) {
                            int x = pixels[k] % width, y = pixels[k] / width;
                            assert(train_data_pos - train_data_begin <
                                   num_pixels * indices.size());
                            assert(train_labels_pos - train_labels_begin <
                                   num_pixels * indices.size());
                            *train_data_pos =
                                TestData(x, y, indices[j], average(x, y), width,
                                         height, images.size(),
                                         PixelConversion::DefaultConversion());
                            *train_labels_pos =
                                PixelData(images[indices[j]](x, y),
                                          PixelConversion::DefaultConversion());
                            ++train_data_pos;
                            ++train_labels_pos;
                          }
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
  auto cmp = [](std::pair<int, double> left, std::pair<int, double> right) {
    return left.second < right.second;
  };
  std::priority_queue<std::pair<int, double>,
                      std::deque<std::pair<int, double>>, decltype(cmp)> q(cmp);
#define USE_STD_UNIFORM_RANDOM_DEVICE 0
#if USE_STD_UNIFORM_RANDOM_DEVICE
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  for (int i = 0; i < total; ++i) q.push(std::make_pair(i, dis(gen)));
#else
  int seed = 12345;
  srand(seed);
  for (int i = 0; i < total; ++i)
    q.push(std::make_pair(i, static_cast<double>(rand()) / RAND_MAX));
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
    double** training_data, int* training_data_dim1, int* training_data_dim2,
    double** training_labels, int* training_labels_dim1,
    int* training_labels_dim2, double** average_img, int* average_dim1,
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
  uint32_t sample_size =
      std::max(static_cast<uint32_t>(0.70 * images.size()), 1U);
  std::cout << "pick indices\n";
  std::cout << "sample_size = " << sample_size << "\n";
  if (indices.empty())
    PickRandomIndices(images.size(), sample_size, indices, order);
  std::cout << "average\n";
  // Compute average.
  ComputeAverageImage(images, indices, average);
  // Return average img to user in normalized form.
  // std::cout << "write out average image\n";
  *average_img = new double[width * height * 3];
  *average_dim1 = height;
  *average_dim2 = width;
  *average_dim3 = 3;
  const image::Pixel* pixels =
      reinterpret_cast<const image::Pixel*>(average.GetBytes());
  int num_pixels = width * height;
  std::transform(pixels, pixels + num_pixels,
                 reinterpret_cast<PixelData*>(*average_img),
                 [](const image::Pixel& p) {
    return PixelData(p, PixelConversion::DefaultConversion());
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
