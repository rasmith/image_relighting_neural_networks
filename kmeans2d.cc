#include "image.h"
#include "kdtree.h"
#include "kmeans_training_data.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
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
  int block_size = num_pixels / num_threads + 1;
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
  int block_size = num_pixels / num_threads + 1;
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

//test, target = kmeans2d.closest_test_target(k, closest, cluster_id,\
                                                  //train_data, target_data)

//test, target = kmeans2d.closest_k_test_target(k, cluster_id, closest,\
                                                  //train_data, train_labels)
// Array indexing:
// i = z  + depth * (y + height * x)
//
// closest_k_target_target
//
// for a given centroid, returns the training and target data for the pixels
// that have this centroid as k-th closest.
//
// k - the number k that where all pixels have cluster_id as k closest.
// cluster_id - centroid id
// closest - for each pixel, the 5 closest centroids in order.
// closest_dim1 - height
// closest_dim2 - width
// closest_dim3 - maximum value for k
// train_data - input training data (x, y, i, r, g, b)
// train_data_dim1 - number of samples
// train_labels_dim2 - data size per sample (3)
// target_data - actual values (r,g,b)
// target_data_dim1 - number of samples
// target_data_dim2 - data size per sample (6)
// test - output data from training input (x, y, i, r, g, b)
// test_dim1 - number of values
// test_dim2 - number of samples
// target - output data from training input (r, g, b)
// target_dim1 - number of values
// target_dim2 - number of samples
void closest_k_test_target(int k, int cluster_id, int* closest,
                           int closest_dim1, int closest_dim2, int closest_dim3,
                           float* train_data, int train_data_dim1,
                           int train_data_dim2, float* target_data,
                           int target_data_dim1, int target_data_dim2,
                           float** test, int* test_dim1, int* test_dim2,
                           float** target, int* target_dim1, int* target_dim2) {
  // Train data configuratian.
  int pixel_dim = 3;
  int light_dim = 1;
  int coord_dim = 2;
  int train_data_size = pixel_dim + light_dim + coord_dim;
  int target_data_size = pixel_dim;
  int num_images = train_data_dim1 / (closest_dim1 * closest_dim2);
  std::cout << "closest_k_test_target: num_images = " << num_images << "\n";
  std::cout << "closest_k_test_target: closest_dim1 = " << closest_dim1 << "\n";
  std::cout << "closest_k_test_target: closest_dim2 = " << closest_dim2 << "\n";
  std::cout << "closest_k_test_target: closest_dim3 = " << closest_dim3 << "\n";
  std::cout << "closest_k_test_target: k = " << k << "\n";
  // closest = np.zeros((height, width, channels))
  // Count how many pixels are k-th closest to this cluster.
  std::cout << "closest_k_test_target: count pixels\n";
  std::cout << "closest_k_test_target: train_data_dim1 = " << train_data_dim1
            << "\n";
  std::cout << "closest_k_test_target: train_data_dim2 = " << train_data_dim2
            << "\n";
  std::cout << "closest_k_test_target: train_total = "
            << (train_data_dim1 * train_data_dim2) << "\n";
  std::cout << "closest_k_test_target: target_data_dim1 = " << target_data_dim1
            << "\n";
  std::cout << "closest_k_test_target: target_data_dim2 = " << target_data_dim2
            << "\n";
  std::cout << "closest_k_test_target: target_data_total = "
            << (target_data_dim1 * target_data_dim2) << "\n";
  int cluster_size = 0;
  for (int y = 0; y < closest_dim1; ++y) {
    for (int x = 0; x < closest_dim2; ++x) {
      int i = k + closest_dim3 * (y * closest_dim2 + x);
      if (cluster_id == closest[i]) ++cluster_size;
    }
  }

  std::cout << "closest_k_test_target: cluster_size = " << cluster_size << "\n";
  // Set test and target dimensions.
  std::cout << "closest_k_test_target:allocate test\n";
  *test_dim1 = cluster_size * num_images;
  *test_dim2 = train_data_size;
  *test = new float[(*test_dim1) * (*test_dim2)];
  std::cout << "closest_k_test_target:allocate target\n";
  *target_dim1 = cluster_size * num_images;
  *target_dim2 = target_data_size;
  *target = new float[(*target_dim1) * (*target_dim2)];
  std::cout << "closest_k_test_target: test_dim1 = " << *test_dim1
            << " test_dim2 = " << *test_dim2
            << " total = " << (*test_dim1) * (*test_dim2) << "\n";
  std::cout << "closest_k_target_target: target_dim1 = " << *target_dim1
            << " target_dim2 = " << *target_dim2
            << " total = " << (*target_dim1) * (*target_dim2) << "\n";

  const int num_threads = 8;
  std::vector<std::thread> threads(num_threads);
  std::vector<int> visited(num_threads, 0);
  std::vector<int> starts(num_threads, 0);
  std::vector<int> ends(num_threads, 0);
  std::vector<int> test_out_starts(num_threads, 0);
  std::vector<int> test_out_ends(num_threads, 0);
  std::vector<int> target_out_starts(num_threads, 0);
  std::vector<int> target_out_ends(num_threads, 0);
  std::vector<int> train_in_starts(num_threads, 0);
  std::vector<int> train_in_ends(num_threads, 0);
  std::vector<int> pixels_copied(num_threads, 0);
  for (int t = 0; t < num_threads; ++t) {
    threads[t] =
        std::thread([
                      &k,
                      &num_images,
                      &pixel_dim,
                      &light_dim,
                      &coord_dim,
                      &cluster_size,
                      &cluster_id,
                      &closest,
                      &closest_dim1,
                      &closest_dim2,
                      &closest_dim3,
                      &train_data,
                      &train_data_dim1,
                      &train_data_dim2,
                      &target_data,
                      &target_data_dim1,
                      &target_data_dim2,
                      &test,
                      &test_dim1,
                      &test_dim2,
                      &target,
                      &target_dim1,
                      &target_dim2,
                      &target_data_size,
                      &train_data_size,
                      &visited,
                      &starts,
                      &ends,
                      &test_out_starts,
                      &test_out_ends,
                      &target_out_starts,
                      &target_out_ends,
                      &train_in_starts,
                      &train_in_ends,
                      &pixels_copied,
                      &num_threads
                    ](int tid)
                         ->void {
                      // Write out the test and target data.
                      int num_pixels = closest_dim1 * closest_dim2;
                      int pos = 0;
                      int block_size = num_images / num_threads + 1;
                      int start = tid * block_size;
                      int end = std::min(num_images, start + block_size);
                      // Input.
                      int train_in_start = num_pixels * start * train_data_size;
                      int train_in_end = num_pixels * end * train_data_size;
                      train_in_starts[tid] = train_in_start;
                      train_in_ends[tid] = train_in_end;
                      int target_in_start =
                          num_pixels * start * target_data_size;
                      int target_in_end = num_pixels * end * target_data_size;
                      // Output.
                      int test_out_start =
                          cluster_size * start * train_data_size;
                      int test_out_end = cluster_size * end * train_data_size;
                      test_out_starts[tid] = test_out_start;
                      test_out_ends[tid] = test_out_end;
                      int target_out_start =
                          cluster_size * start * target_data_size;
                      int target_out_end =
                          cluster_size * end * target_data_size;
                      target_out_starts[tid] = target_out_start;
                      target_out_ends[tid] = target_out_end;
                      starts[tid] = start;
                      ends[tid] = end;
                      float* train_in_pos = train_data + train_in_start;
                      float* test_out_pos = *test + test_out_start;
                      float* target_out_pos = *target + target_out_start;
                      float* target_in_pos = target_data + target_in_start;
                      for (int i = start; i < end; ++i) {
                        for (int j = 0; j < num_pixels; ++j) {
                          ++visited[tid];
                          int x = round(closest_dim2 * *(train_in_pos));
                          int y = round(closest_dim1 * *(train_in_pos + 1));
                          int l = k + closest_dim3 * (y * closest_dim2 + x);
                          if (cluster_id == closest[l]) {
                            ++pixels_copied[tid];
                            for (int k = 0; k < train_data_size; ++k)
                              test_out_pos[k] = train_in_pos[k];
                            for (int k = 0; k < target_data_size; ++k)
                              target_out_pos[k] = target_in_pos[k];
                          }
                          train_in_pos += train_data_size;
                          target_in_pos += target_data_size;
                        }
                      }
                    },
                    t);
  }
  for (int t = 0; t < num_threads; ++t) threads[t].join();
  int total_visited = 0;
  for (int t = 0; t < num_threads; ++t) {
    std::cout << "Thread " << t << " visited " << visited[t] << " values.\n";
    total_visited += visited[t];
    std::cout << "t=" << t << " start =" << starts[t] << " end = " << ends[t]
              << "\n";
    std::cout << "test_out_start = " << test_out_starts[t]
              << " test_out_ends = " << test_out_ends[t] << "\n";
    std::cout << "target_out_start = " << target_out_starts[t]
              << " target_out_ends = " << target_out_ends[t] << "\n";
    std::cout << "train_in_start = " << train_in_starts[t]
              << " train_in_ends = " << train_in_ends[t] << "\n";
    std::cout << "pixels_copied = " << pixels_copied[t] << "\n";
  }
  std::cout << "Visited " << total_visited << " values.\n";
}

// kmeans2d.update_errors(test_data, target_data, predictions, errors)
void compute_total_values(float* train, int train_dim1, int train_dim2,
                          float* target, int target_dim1, int target_dim2,
                          float* totals, int totals_dim1, int totals_dim2) {

  const int num_threads = 8;
  std::vector<std::thread> threads(num_threads);
  std::vector<float*> total_values_threads;
  for (int t = 0; t < num_threads; ++t)
    total_values_threads[t] = new float[totals_dim1 * totals_dim2];
  for (int t = 0; t < num_threads; ++t) {
    threads[t] =
        std::thread([
                      &train,
                      &train_dim1,
                      &train_dim2,
                      &target,
                      &target_dim1,
                      &target_dim2,
                      &totals_dim1,
                      &totals_dim2,
                      &num_threads,
                      &total_values_threads
                    ](int tid)
                         ->void {
                      float* totals = total_values_threads[tid];
                      int block_size = target_dim1 / num_threads + 1;
                      int start = tid * block_size;
                      int end = std::min(start + block_size, target_dim1);
                      for (int i = start; i < end; ++i) {
                        float* train_values = &train[i * train_dim2];
                        float* target_values = &target[i * target_dim2];
                        float x_value = train_values[0];
                        float y_value = train_values[1];
                        int x = round(x_value * totals_dim2);
                        int y = round(y_value * totals_dim1);
                        for (int c = 0; c < target_dim2; ++c) {
                          float value = target_values[c];
                          totals[y * totals_dim2 + x] += value * value;
                        }
                      }
                    },
                    t);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
  for (int i = 0; i < num_threads; ++i) {
    float* values = total_values_threads[i];
    for (int j = 0; j < totals_dim2 * totals_dim1; ++j) totals[j] += values[j];
    delete[] values;
  }
}

// kmeans2d.update_errors(test_data, target_data, predictions, errors)
void predictions_to_images(std::vector<int>& order, float* test, int test_dim1,
                           int test_dim2, float* predictions,
                           int predictions_dim1, int predictions_dim2,
                           float* predicted_images, int predicted_images_dim1,
                           int predicted_images_dim2, int predicted_images_dim3,
                           int predicted_images_dim4) {
  std::cout << "test_dim1 = " << test_dim1 << "\n";
  std::cout << "test_dim2 = " << test_dim2 << "\n";
  std::cout << "predictions_dim1 = " << predictions_dim1 << "\n";
  std::cout << "predictions_dim2 = " << predictions_dim2 << "\n";
  std::cout << "predicted_images_dim1 = " << predicted_images_dim1 << "\n";
  std::cout << "predicted_images_dim2 = " << predicted_images_dim2 << "\n";
  std::cout << "predicted_images_dim3 = " << predicted_images_dim3 << "\n";
  std::cout << "predicted_images_dim4 = " << predicted_images_dim4 << "\n";
  const int num_threads = 8;
  std::vector<std::thread> threads(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads[t] =
        std::thread([
                      &order,
                      &test,
                      &test_dim1,
                      &test_dim2,
                      &predictions,
                      &predictions_dim1,
                      &predictions_dim2,
                      &predicted_images,
                      &predicted_images_dim1,
                      &predicted_images_dim2,
                      &predicted_images_dim3,
                      &predicted_images_dim4,
                      &num_threads
                    ](int tid)
                         ->void {
                      int block_size = predictions_dim1 / num_threads + 1;
                      int start = tid * block_size;
                      int end = std::min(start + block_size, predictions_dim1);
                      for (int i = start; i < end; ++i) {
                        float* predicted_values =
                            &predictions[i * predictions_dim2];
                        float* test_values = &test[i * test_dim2];
                        float x_value = test_values[0];
                        float y_value = test_values[1];
                        float i_value = test_values[2];
                        int num_images = predicted_images_dim4;
                        int height = predicted_images_dim2;
                        int width = predicted_images_dim3;
                        int channels = predicted_images_dim1;
                        // predicted_images[i, y, x, c]
                        // I = channels * (width * (height * i + y) + x) + c
                        // I = channels * (width * height * i + width * y + x) +
                        // c
                        // I = width * height * channels * i + (width * y + x) *
                        // channels
                        //     + c
                        int x = round(x_value * predicted_images_dim3);
                        int y = round(y_value * predicted_images_dim2);
                        int n = round(i_value * order.size());
                        for (int c = 0; c < channels; ++c) {
                          int idx =
                              channels * (width * (height * order[n] + y) + x) +
                              c;
                          if (idx < 0) {
                            std::cout << "idx = " << idx << " x = " << x
                                      << " y =  " << y << " n = " << n
                                      << " c = " << c
                                      << " i_value = " << i_value
                                      << " order.size() = " << order.size()
                                      << " order[n] = " << order[n] << "\n";
                          }
                          assert(idx >= 0);
                          assert(idx <= height * width * channels * num_images);
                          predicted_images[idx] += predicted_values[c];
                        }
                      }
                    },
                    t);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
}

// kmeans2d.update_errors(test_data, target_data, predictions, errors)
void compute_errors(int ensemble_size, std::vector<int>& order, float* train,
                    int train_dim1, int train_dim2, float* target,
                    int target_dim1, int target_dim2, float* predictions,
                    int predictions_dim1, int predictions_dim2,
                    float* predicted_images, int predicted_images_dim1,
                    int predicted_images_dim2, int predicted_images_dim3,
                    int predicted_images_dim4, float* errors, int errors_dim1,
                    int errors_dim2) {
  const int num_threads = 8;
  std::vector<std::thread> threads(num_threads);
  std::vector<float*> error_values_threads(num_threads, nullptr);
  for (int t = 0; t < num_threads; ++t)
    error_values_threads[t] = new float[errors_dim1 * errors_dim2];
  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread(
        [
          &ensemble_size,
          &order,
          &train,
          &train_dim1,
          &train_dim2,
          &target,
          &target_dim1,
          &target_dim2,
          &predictions,
          &predictions_dim1,
          &predictions_dim2,
          &predicted_images,
          &predicted_images_dim1,
          &predicted_images_dim2,
          &predicted_images_dim3,
          &predicted_images_dim4,
          &error_values_threads,
          &errors_dim1,
          &errors_dim2,
          &num_threads
        ](int tid)
             ->void {
          float* errors = error_values_threads[tid];
          int block_size = predictions_dim1 / num_threads + 1;
          int start = tid * block_size;
          int end = std::min(start + block_size, predictions_dim1);
          for (int i = start; i < end; ++i) {
            float* predicted_values = &predictions[i * predictions_dim2];
            float* train_values = &train[i * train_dim2];
            float* target_values = &target[i * target_dim2];
            float x_value = train_values[0];
            float y_value = train_values[1];
            float i_value = train_values[2];
            int x = round(x_value * predicted_images_dim3);
            int y = round(y_value * predicted_images_dim2);
            int num_images = predicted_images_dim1;
            int height = predicted_images_dim2;
            int width = predicted_images_dim3;
            int channels = predicted_images_dim4;
            int n = round(i_value * predicted_images_dim1);
            float sum = 0;
            for (int c = 0; c < channels; ++c) {
              float value = predicted_images
                  [channels * (width * (height * order[n] + y) + x) + c];
              float diff = value / ensemble_size - target_values[c];
              sum += diff * diff;
            }
            errors[y * width + x] += sum;
          }
        },
        t);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
  for (int t = 0; t < num_threads; ++t) {
    float* values = error_values_threads[t];
    for (int j = 0; j < errors_dim1 * errors_dim2; ++j) errors[j] += values[j];
    delete[] values;
  }
}

void closest_n(int width, int height, int n, std::vector<float>& centers,
               int** closest, int* dim1, int* dim2, int* dim3) {
  *dim1 = height;
  *dim2 = width;
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
          uint32_t block_size = num_pixels / num_threads + 1;
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
              (*closest)[k + n * (x + y * width)] = best;
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

//test, batch_sizes, levels, cluster_ids = \
      //kmeans2d.assignments_to_predict_data(assignments, average)
typedef std::pair<int, int> Tuple;
typedef std::vector<Tuple> Tuples;
struct TupleHash {
  std::size_t operator()(const Tuple& t) const {
    std::hash<std::string> h;
    return h(std::to_string(t.first) + "-" + std::to_string(t.second));
  }
};

void assign_to_predict_data(int num_images, int* assign, int assign_dim1,
                            int assign_dim2, int assign_dim3, uint8_t* average,
                            int average_dim1, int average_dim2,
                            int average_dim3, float** test, int* test_dim1,
                            int* test_dim2, int** batch_sizes,
                            int* batch_sizes_dim1, int** levels,
                            int* levels_dim1, int** cluster_ids,
                            int* cluster_ids_dim1) {
  int width = assign_dim2;
  int height = assign_dim1;
  std::unordered_map<Tuple, Tuples, TupleHash> map;
  for (int y = 0; y < width; ++y) {
    for (int x = 0; x < width; ++x) {
      int* assignment = &assign[assign_dim3 * (y * width + x)];
      int level = assignment[0];
      for (int i = 1; i < assign_dim3; ++i) {
        Tuple t = std::make_pair(x, y);
        int center = assignment[i];
        if (map.find(std::make_pair(level, center)) == map.end())
          map.emplace(t, Tuples());
        auto item = map.find(std::make_pair(level, center));
        item->second.push_back(t);
      }
    }
  }
  int total_size = 0;
  for (auto iter = map.begin(); iter != map.end(); ++iter)
    total_size += iter->second.size();
  *test_dim1 = total_size;
  *test_dim2 = 6;
  *test = new float[(*test_dim1) * (*test_dim2)];
  *cluster_ids_dim1 = *batch_sizes_dim1 = *levels_dim1 = map.size();
  *cluster_ids = new int[*cluster_ids_dim1];
  *levels = new int[*levels_dim1];
  *batch_sizes = new int[*batch_sizes_dim1];
  int current = 0;
  for (auto iter = map.begin(); iter != map.end(); ++iter) {
    auto& v = iter->second;
    auto& cl = iter->first;
    int level = cl.first;
    int center = cl.second;
    (*cluster_ids)[current] = center;
    (*levels)[current] = level;
    (*batch_sizes)[current] = v.size();
    float* values = &(*test)[(*test_dim2) * current];
    for (int i = 0; i < v.size(); ++i) {
      auto& xy = v[i];
      int x = xy.first;
      int y = xy.second;
      values[0] = x / static_cast<float>(width);
      values[1] = y / static_cast<float>(height);
      values[2] = i / static_cast<float>(num_images);
      uint8_t* rgb = &average[3 * (width * y + x)];
      values[3] = rgb[0] / 255.0f;
      values[4] = rgb[1] / 255.0f;
      values[5] = rgb[2] / 255.0f;
    }
    ++current;
  }
}

// kmeans2d.predictions_to_img(test[start:end], predictions, predicted_img)
void predictions_to_img(float* test, int test_dim1, int test_dim2,
                        float* predictions, int predictions_dim1,
                        int predictions_dim2, float* img, int img_dim1,
                        int img_dim2, int img_dim3) {
  int height = img_dim1;
  int width = img_dim2;
  for (int i = 0; i < test_dim1; ++i) {
    float* test_values = &test[i * test_dim2];
    int x = round(width * test_values[0]);
    int y = round(height * test_values[1]);
    float* predicted_values = &predictions[i * predictions_dim2];
    for (int c = 0; c < img_dim3; ++c)
      img[c + img_dim3 * (y * img_dim2 + x)] += predicted_values[c];
  }
}

void kmeans_training_data(const std::string& directory, int num_centers,
                          int* width, int* height, std::vector<int>& indices,
                          std::vector<int>& order, std::vector<float>& centers,
                          std::vector<int>& labels,
                          std::vector<int>& batch_sizes, float** train_data,
                          int* train_data_dim1, int* train_data_dim2,
                          float** train_labels, int* train_labels_dim1,
                          int* train_labels_dim2, float** average,
                          int* average_dim1, int* average_dim2,
                          int* average_dim3) {
  int width_out = -1, height_out = -1;
  std::vector<glm::vec2> glm_centers(num_centers);
  KmeansDataAndLabels(directory, num_centers, width_out, height_out, train_data,
                      train_data_dim1, train_data_dim2, train_labels,
                      train_labels_dim1, train_labels_dim2, average,
                      average_dim1, average_dim2, average_dim3, indices, order,
                      glm_centers, labels, batch_sizes);
  *width = width_out;
  *height = height_out;
  centers.resize(2 * num_centers);
  for (int i = 0; i < glm_centers.size(); ++i) {
    centers[2 * i] = glm_centers[i][0];
    centers[2 * i + 1] = glm_centers[i][1];
  }
}
