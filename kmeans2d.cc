#include "image.h"
#include "kdtree.h"
#include "kmeans_training_data.h"
#include "logger.h"
#include "types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <map>
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
  // closest_k_test_target(
  // k, cluster_id, reinterpret_cast<int*>(&closest[0]), height, width,
  // ensemble_size, reinterpret_cast<float*>(&train_data[0]),
  // num_pixels * num_images, sizeof(TestData) / sizeof(float),
  // reinterpret_cast<float*>(&target_data[0]), num_pixels * num_images,
  // sizeof(PixelData) / sizeof(float), &test_in, &test_dim1, &test_dim2,
  //&target_in, &target_dim1, &target_dim2);
  // Train data configuratian.
  int num_images = train_data_dim1 / (closest_dim1 * closest_dim2);
  int width = closest_dim2;
  int height = closest_dim1;
  int ensemble_size = closest_dim3;
  int cluster_size = 0;
  std::vector<int> cluster;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int p = y * width + x;
      if (cluster_id == closest[k + ensemble_size * p]) cluster.push_back(p);
    }
  }
  cluster_size = cluster.size();

  // Set test and target dimensions.
  *test_dim1 = cluster_size * num_images;
  *test_dim2 = sizeof(TestData) / sizeof(float);
  *test = new float[(*test_dim1) * (*test_dim2)];
  *target_dim1 = cluster_size * num_images;
  *target_dim2 = sizeof(PixelData) / sizeof(float);
  *target = new float[(*target_dim1) * (*target_dim2)];
  LOG(DEBUG) << "closest_k_test_target: test_dim1 = " << *test_dim1
             << " test_dim2 = " << *test_dim2
             << " total = " << (*test_dim1) * (*test_dim2) << "\n";
  LOG(DEBUG) << "closest_k_target_target: target_dim1 = " << *target_dim1
             << " target_dim2 = " << *target_dim2
             << " total = " << (*target_dim1) * (*target_dim2) << "\n";
  LOG(DEBUG) << " width = " << width << " height = " << height
             << " cluster_size =" << cluster_size << "\n ";
  LOG(DEBUG) << " num_images = " << num_images << "\n";
  const int num_threads = 8;
  std::vector<std::thread> threads(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread(
        [
          &num_images,
          &cluster_size,
          &cluster,
          &train_data,
          &target_data,
          &test,
          &target,
          &num_threads,
          &width,
          &height
        ](int tid)
             ->void {
          // Write out the test and target data.
          int num_pixels = width * height;
          int block_size = num_images / num_threads + 1;
          int start = tid * block_size;
          int end = std::min(num_images, start + block_size);
          block_size = end - start;
          // Output.
          TestData* test_out =
              reinterpret_cast<TestData*>(*test) + cluster_size * start;
          TestData* train_in =
              reinterpret_cast<TestData*>(train_data) + num_pixels * start;
          PixelData* target_out =
              reinterpret_cast<PixelData*>(*target) + cluster_size * start;
          PixelData* target_in =
              reinterpret_cast<PixelData*>(target_data) + num_pixels * start;
          for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < cluster_size; ++j) {
              int c = cluster[j];
              test_out[i * cluster_size + j] = train_in[i * num_pixels + c];
              target_out[i * cluster_size + j] = target_in[i * num_pixels + c];
            }
          }
        },
        t);
  }
  for (int t = 0; t < num_threads; ++t) threads[t].join();
}

void predictions_to_errors(std::vector<int>& order, int ensemble_size,
                           float* test, int test_dim1, int test_dim2,
                           float* target, int target_dim1, int target_dim2,
                           float* predictions, int predictions_dim1,
                           int predictions_dim2, float* errors, int errors_dim1,
                           int errors_dim2) {
  LOG(DEBUG) << "predictions_to_errors:ensemble_size = " << ensemble_size
             << "\n";
  LOG(DEBUG) << "predictions_to_errors:test_dim1 = " << test_dim1 << "\n";
  LOG(DEBUG) << "predictions_to_errors:test_dim2 = " << test_dim2 << "\n";
  LOG(DEBUG) << "predictions_to_errors:predictions_dim1 = " << predictions_dim1
             << "\n";
  LOG(DEBUG) << "predictions_to_errors:predictions_dim2 = " << predictions_dim2
             << "\n";
  LOG(DEBUG) << "predictions_to_errors:errors_dim1 = " << errors_dim1 << "\n";
  LOG(DEBUG) << "predictions_to_errors:errors_dim2 = " << errors_dim2 << "\n";
  const int num_threads = 8;
  int height = errors_dim1;
  int width = errors_dim2;
  int num_pixels = height * width;
  std::vector<float> thread_errors(num_threads * num_pixels, 0.0f);
  std::vector<std::thread> threads(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread([
                               &order,
                               &height,
                               &width,
                               &num_pixels,
                               &ensemble_size,
                               &test,
                               &test_dim1,
                               &test_dim2,
                               &target,
                               &target_dim1,
                               &target_dim2,
                               &predictions,
                               &predictions_dim1,
                               &predictions_dim2,
                               &thread_errors,
                               &num_threads
                             ](int tid)
                                  ->void {
                               float* errors = &thread_errors[tid * num_pixels];
                               int block_size = test_dim1 / num_threads + 1;
                               int start = tid * block_size;
                               int end =
                                   std::min(start + block_size, test_dim1);
                               float* test_pos = test + test_dim2 * start;
                               float* target_pos = target + target_dim2 * start;
                               float* predictions_pos =
                                   predictions + predictions_dim2 * start;
                               for (int i = start; i < end; ++i) {
                                 int x = round(*(test_pos) * (width - 1));
                                 int y = round(*(test_pos + 1) * (height - 1));
                                 int index = y * width + x;
                                 float total = 0;
                                 for (int c = 0; c < 3; ++c)
                                   total += target_pos[c];
                                 if (total == 0.0f) total = 1.0f;
                                 for (int c = 0; c < 3; ++c) {
                                   int k = y * width + x;
                                   float e = predictions_pos[c] - target_pos[c];
                                   errors[k] += (e * e) / (total * total);
                                 }
                                 target_pos += target_dim2;
                                 predictions_pos += predictions_dim2;
                                 test_pos += test_dim2;
                               }
                             },
                             t);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
  for (int i = 0; i < num_threads; ++i) {
    for (int j = 0; j < num_pixels; ++j)
      errors[j] += thread_errors[i * num_pixels + j];
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

// Assignment data : [[L, i0, i1, i2, i3, i4], ...]
//  shape = W x H x (E + 1)
//  L = level assigned to
//  iN = model # at this level
//  W = width, H = heght, E = ensemble size
//
// Test data: [[x, y, i, r, g, b], ....]
//  shape = W * H x 6
//  x = x position
//  y = y position
//  i = image number
//  r, g, b = average rgb value at (x, y) across all images
//
// Network data: [[L, i, s, n], ... ]
//  shape = W * H x 4
//  L = level assigned to
//  i = model # at this level
//  s = start index
//  n =  count
//
//  Given image_number, generate test data to feed to models in assignment data
//  output as test data
void assignment_data_to_test_data(
    int* assignment_data, int assignment_data_dim1, int assignment_data_dim2,
    int assignment_data_dim3, int image_number, int num_images,
    float* average_image, int average_image_dim1, int average_image_dim2,
    int average_image_dim3, float** test_data, int* test_data_dim1,
    int* test_data_dim2, int** network_data, int* network_data_dim1,
    int* network_data_dim2) {
  int width = assignment_data_dim2;
  int height = assignment_data_dim1;
  int num_pixels = width * height;
  int ensemble_size = assignment_data_dim3 - 1;
  int num_networks = 0;
  LOG(STATUS) << "ensemble_size = " << ensemble_size << "\n";
  // 1. Put all assignment data into vectors for each network.
  std::unordered_map<NetworkData, std::vector<int>, HashNetworkData>
      network_map;
  int* pos = &assignment_data[0];
  for (int i = 0; i < num_pixels; ++i) {
    int x = i % width, y = i / width;
    for (int j = 0; j < ensemble_size; ++j) {
      NetworkData query(pos[0], pos[j + 1]);
      if (network_map.find(query) == network_map.end())
        network_map.insert(std::make_pair(query, std::vector<int>()));
      if (!(network_map.find(query)->first.level == pos[0])) {
        LOG(STATUS) << "level mismatch: network_map.find(query)->first.level = "
                    << network_map.find(query)->first.level
                    << " pos->level = " << pos[0] << "\n";
      }
      assert(network_map.find(query)->first.level == pos[0]);
      network_map.find(query)->second.push_back(i);
    }
    pos += ensemble_size + 1;
  }
  int count = 0;
  for (const auto& entry : network_map) {
    assert(entry.second.size() > 0);
    count += entry.second.size();
  }
  if (count != num_pixels * ensemble_size) {
    LOG(ERROR) << "Found." << count << " test entries but expected."
               << num_pixels * ensemble_size << "\n";
  }
  assert(count == num_pixels * ensemble_size);
  // 2. Allocate test data.
  *test_data_dim1 = num_pixels * ensemble_size;
  *test_data_dim2 = sizeof(TestData) / sizeof(float);
  *test_data = new float[(*test_data_dim1) * (*test_data_dim2)];
  TestData* test_pos = reinterpret_cast<TestData*>(*test_data);
  // 3. Transform assignment data into test data.
  std::vector<int> assignment_counts(num_pixels, 0);
  std::vector<NetworkData> networks;
  int index = 0;
  // pos = reinterpret_cast<const AssignmentData*>(&assignment_data[0]);
  pos = &assignment_data[0];
  for (const auto& entry : network_map) {
    networks.push_back(entry.first);
    networks.back().count = entry.second.size();
    networks.back().start = (index == 0 ? 0 : networks[index - 1].count +
                                                  networks[index - 1].start);
    for (int i = 0; i < entry.second.size(); ++i) {
      int pixel_index = entry.second[i], x = pixel_index % width,
          y = pixel_index / width;
      if (!(entry.first.level == pos[pixel_index * (ensemble_size + 1)])) {
        LOG(STATUS) << "level mismatch "
                    << "entry.first.level = " << entry.first.level
                    << " pos[pixel_index].level = "
                    << pos[pixel_index * (ensemble_size + 1)] << "\n";
      }
      assert(entry.first.level == pos[pixel_index * (ensemble_size + 1)]);
      assert(pixel_index >= 0 && pixel_index < num_pixels);
      test_pos[networks.back().start + i] = TestData(
          x, y, image_number, &average_image[pixel_index * average_image_dim3],
          width, height, num_images, PixelConversion::DefaultConversion());
      ++assignment_counts[pixel_index];
      assert(assignment_counts[pixel_index] >= 1 &&
             assignment_counts[pixel_index] <= ensemble_size);
    }
    ++index;
  }
  // 4. Write back network data.
  *network_data_dim1 = networks.size();
  *network_data_dim2 = sizeof(NetworkData) / sizeof(int);
  *network_data = new int[(*network_data_dim1) * (*network_data_dim2)];
  NetworkData* network_pos = reinterpret_cast<NetworkData*>(*network_data);
  --network_pos;
  for (int i = 0; i < networks.size(); ++i) {
    *++network_pos = networks[i];
  }
  for (int i = 0; i < assignment_counts.size(); ++i) {
    if (assignment_counts[i] != ensemble_size) {
      LOG(ERROR) << "assignment_counts[" << i << "]=" << assignment_counts[i]
                 << "\n";
    }
    assert(assignment_counts[i] == ensemble_size);
  }
}

void predictions_to_image(float* image_out, int image_out_dim1,
                          int image_out_dim2, int image_out_dim3, float* test,
                          int test_dim1, int test_dim2, float* predictions,
                          int predictions_dim1, int predictions_dim2) {
  int height = image_out_dim1;
  int width = image_out_dim2;
  int channels = image_out_dim3;
  float* test_pos = test;
  assert(predictions_dim2 == image_out_dim3);
  for (float* predictions_pos = predictions;
       predictions_pos < predictions + predictions_dim1 * predictions_dim2;
       predictions_pos += predictions_dim2) {
    int x = (width - 1) * (*test_pos);
    int y = (height - 1) * (*(test_pos + 1));
    for (int j = 0; j < image_out_dim3; ++j)
      image_out[image_out_dim3 * (y * width + x) + j] += *(predictions_pos + j);
    test_pos += test_dim2;
  }
}
