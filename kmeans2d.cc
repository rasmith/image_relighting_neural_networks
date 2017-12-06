#include "image.h"
#include "kdtree.h"
#include "kmeans_training_data.h"

#include <dirent.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <random>
#include <thread>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/color_space.hpp>
#include <glm/gtx/norm.hpp>

void init_centers(int num_threads, int num_centers, int width, int height,
                  std::vector<glm::vec2>& centers) {
  centers.clear();
  int num_pixels = width * height;
  int gap = num_pixels / num_centers;
  for (int i = 0; i < num_pixels && centers.size() < num_centers; i += gap) {
    int x = i % width;
    int y = i / width;
    glm::vec2 pixel(x, y);
    centers.push_back(pixel);
  }
}

void assign_labels_thread(int tid, int num_threads, int width, int height,
                          const kdtree::KdTree& tree,
                          std::vector<int>& labels) {
  int num_pixels = width * height;
  int block_size = num_pixels / num_threads;
  int t_start = tid * block_size;
  int t_end = std::min(t_start + block_size, num_pixels);
  std::vector<float> distances(block_size, std::numeric_limits<float>::max());
  for (int j = t_start; j < t_end; ++j) {
    int x = j % width;
    int y = j / width;
    glm::vec2 pixel(x, y);
    int best = -1;
    float best_distance = std::numeric_limits<float>::max();
    tree.NearestNeighbor(pixel, &best, &best_distance);
    distances[j - t_start] = best_distance;
    labels[j] = best;
  }
}

void assign_labels_threaded(int num_threads, int width, int height,
                            const kdtree::KdTree& tree,
                            std::vector<int>& labels) {
  // std::thread threads[num_threads];
  std::vector<std::thread> threads(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads[i] =
        std::thread([&num_threads, &width, &height, &tree, &labels ](int tid)
                                                                        ->void {
                      assign_labels_thread(tid, num_threads, width, height,
                                           tree, labels);
                    },
                    i);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
}

void update_centers_thread(int tid, int num_threads, int width, int height,
                           const std::vector<int>& labels,
                           std::vector<glm::vec2>& centers,
                           std::vector<int>& counts) {
  int num_pixels = width * height;
  int block_size = num_pixels / num_threads;
  int t_start = tid * block_size;
  int t_end = std::min(t_start + block_size, num_pixels);
  std::vector<float> distances(block_size, std::numeric_limits<float>::max());
  std::fill(centers.begin(), centers.end(), glm::vec2(0.0f));
  std::fill(counts.begin(), counts.end(), 0);
  for (int j = t_start; j < t_end; ++j) {
    int x = j % width;
    int y = j / width;
    glm::vec2 pixel(x, y);
    ++counts[labels[j]];
    centers[labels[j]] += pixel;
  }
}

void update_centers_threaded(int num_threads, int width, int height,
                             const std::vector<int>& labels,
                             std::vector<glm::vec2>& centers) {
  // std::thread threads[num_threads];
  std::vector<std::thread> threads(num_threads);
  std::vector<std::vector<glm::vec2>> center_arrays(num_threads);
  std::vector<std::vector<int>> counts_arrays(num_threads);
  for (int i = 0; i < num_threads; ++i) center_arrays[i].resize(centers.size());
  for (int i = 0; i < num_threads; ++i) counts_arrays[i].resize(centers.size());
  for (int i = 0; i < num_threads; ++i) {
    std::vector<glm::vec2>& center_array = center_arrays[i];
    std::vector<int>& counts_array = counts_arrays[i];
    threads[i] =
        std::thread([
                      &num_threads,
                      &width,
                      &height,
                      &labels,
                      &center_array,
                      &counts_array
                    ](int tid)
                         ->void {
                      update_centers_thread(tid, num_threads, width, height,
                                            labels, center_array, counts_array);
                    },
                    i);
  }
  std::fill(centers.begin(), centers.end(), glm::vec2(0.0f));
  for (int i = 0; i < num_threads; ++i) threads[i].join();
  std::vector<int> counts(centers.size(), 0);
  for (int i = 0; i < counts.size(); ++i) {
    for (int j = 0; j < num_threads; ++j) {
      centers[i] += center_arrays[j][i];
      counts[i] += counts_arrays[j][i];
    }
  }
  for (int i = 0; i < centers.size(); ++i) centers[i] /= counts[i];
}

float compute_difference(const std::vector<glm::vec2>& centers1,
                         const std::vector<glm::vec2>& centers2) {
  float diff = 0;
  for (int i = 0; i < centers1.size(); ++i)
    diff += glm::distance2(centers1[i], centers2[i]);
  return diff;
}

void kmeans(int width, int height, std::vector<glm::vec2>& centers,
            std::vector<int>& labels) {
  int num_threads = 8;
  labels.resize(width * height);
  auto start = std::chrono::high_resolution_clock::now();
  init_centers(num_threads, centers.size(), width, height, centers);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::vector<glm::vec2> old_centers(centers.size());
  float diff = std::numeric_limits<float>::max();
  while (diff > 1e-5) {
    std::copy(centers.begin(), centers.end(), old_centers.begin());
    kdtree::KdTree tree;
    tree.AssignPoints(old_centers);
    tree.Build();
    assign_labels_threaded(num_threads, width, height, tree, labels);
    update_centers_threaded(num_threads, width, height, labels, centers);
    diff = compute_difference(old_centers, centers);
  }
}

void closest_n(int width, int height, int n, std::vector<float>& centers,
               int** closest, int* dim1, int* dim2, int* dim3) {
  *dim1 = width;
  *dim2 = height;
  *dim3 = n;
  *closest = new int[(*dim1) * (*dim2) * (*dim3)];
  std::vector<glm::vec2> glm_centers(centers.size() / 2);
  for (int i = 0; i < centers.size() / 2; ++i)
    glm_centers[i] = glm::vec2(centers[2 * i], centers[2 * i + 1]);
  kdtree::KdTree tree;
  tree.AssignPoints(glm_centers);
  tree.Build();
  uint32_t num_threads = 8;
  uint32_t num_pixels = width * height;
  std::vector<std::thread> threads(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads[i] = std::thread(
        [&num_threads,
         &num_pixels,
         &closest,
         &tree,
         &n,
         &width,
         &height ](int tid)
                      ->void {
          uint32_t block_size = num_pixels / num_threads;
          uint32_t start = block_size * tid;
          uint32_t end = std::min(num_pixels, start + block_size);
          for (uint32_t i = start; i < end; ++i) {
            uint32_t x = i % width;
            uint32_t y = i / width;
            glm::vec2 pixel(x, y);
            float min_distance = -std::numeric_limits<float>::max();
            for (int k = 0; k < n; ++k) {
              float best_distance = std::numeric_limits<float>::max();
              int best = -1;
              tree.NearestNeighbor(pixel, min_distance, &best, &best_distance);
              min_distance = best_distance;
              (*closest)[k + n * (x * height + y)] = best;
            }
          }
        },
        i);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
}

void kmeans2d(int width, int height, std::vector<float>& centers,
              std::vector<int>& labels) {
  std::vector<glm::vec2> glm_centers(centers.size() / 2);
  kmeans(width, height, glm_centers, labels);
  for (int i = 0; i < glm_centers.size(); ++i) {
    centers[2 * i] = glm_centers[i][0];
    centers[2 * i + 1] = glm_centers[i][1];
  }
}

void kmeans_training_data(const std::string& directory, int num_centers,
                          int* width, int* height, std::vector<int>& indices,
                          std::vector<float>& centers, std::vector<int>& labels,
                          std::vector<int>& batch_sizes, float** train_data,
                          int* train_data_dim1, int* train_data_dim2,
                          float** train_labels, int* train_labels_dim1,
                          int* train_labels_dim2) {
  int width_out = -1, height_out = -1;
  std::vector<glm::vec2> glm_centers(num_centers);
  KmeansDataAndLabels(directory, num_centers, width_out, height_out, train_data,
                      train_data_dim1, train_data_dim2, train_labels,
                      train_labels_dim1, train_labels_dim2, indices,
                      glm_centers, labels, batch_sizes);
  *width = width_out;
  *height = height_out;
  centers.resize(2 * num_centers);
  for (int i = 0; i < glm_centers.size(); ++i) {
    centers[2 * i] = glm_centers[i][0];
    centers[2 * i + 1] = glm_centers[i][1];
  }
}
