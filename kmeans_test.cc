#include "image.h"
#include "kdtree.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/color_space.hpp>
#include <glm/gtx/norm.hpp>

std::ostream& operator<<(std::ostream& out, const glm::vec2& c) {
  out << "(" << c.x << "," << c.y << ")";
  return out;
}

std::ostream& operator<<(std::ostream& out, const image::Pixel& pixel) {
  out << "(" << static_cast<int>(pixel.r) << "," << static_cast<int>(pixel.g)
      << "," << static_cast<int>(pixel.b) << ")";
  return out;
}

template <typename T>
void print_vector(const std::vector<T>& centers) {
  for (int i = 0; i < centers.size(); ++i) {
    std::cout << centers[i] << " ";
  }
  std::cout << "\n";
}

inline void nearest_neighbor(const std::vector<glm::vec2>& values,
                             const glm::vec2& query, float* best_distance,
                             int* best) {
  *best = -1;
  *best_distance = std::numeric_limits<float>::max();
  for (int i = 0; i < values.size(); ++i) {
    float distance = glm::distance2(values[i], query);
    if (distance < *best_distance) {
      *best = i;
      *best_distance = distance;
    }
  }
}

void init_centers(int num_threads, int num_centers, int width, int height,
                  std::vector<glm::vec2>& centers) {
  centers.clear();
  int num_pixels = width * height;
  int gap = num_pixels / num_centers;
  for (int i = 0; i < num_pixels; i += gap) {
    int x = i % width;
    int y = i / width;
    glm::vec2 pixel(x, y);
    centers.push_back(pixel);
  }
}

void assign_labels_thread(int tid, int num_threads, int width, int height,
                          const kdtree::KdTree& tree,
                          const std::vector<glm::vec2>& centers,
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
#if 0 
    nearest_neighbor(centers, pixel, &best_distance, &best);
#else
    tree.NearestNeighbor(pixel, &best, &best_distance);
#endif
    distances[j - t_start] = best_distance;
    labels[j] = best;
    if(!(best >= 0  && best < centers.size()))  {
      std::cout << "best =  " << best << " best_distance = " << best_distance
        << " num_centers = " << centers.size() << "\n";

      exit(0);
    }
  }
}

void assign_labels_threaded(int num_threads, int width, int height,
                            const kdtree::KdTree& tree,
                            const std::vector<glm::vec2>& centers,
                            std::vector<int>& labels) {
  std::thread threads[num_threads];
  for (int i = 0; i < num_threads; ++i) {
    threads[i] = std::thread(
        [&num_threads, &width, &height, &tree, &centers,
         &labels](int tid) -> void {
          assign_labels_thread(tid, num_threads, width, height, tree, centers,
                               labels);
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
    assert(labels[j] >= 0 && labels[j] < centers.size());
    ++counts[labels[j]];
    centers[labels[j]] += pixel;
  }
}

void update_centers_threaded(int num_threads, int width, int height,
                             const std::vector<int>& labels,
                             std::vector<glm::vec2>& centers) {
  std::thread threads[num_threads];
  std::vector<std::vector<glm::vec2>> center_arrays(num_threads);
  std::vector<std::vector<int>> counts_arrays(num_threads);
  for (int i = 0; i < num_threads; ++i) center_arrays[i].resize(centers.size());
  for (int i = 0; i < num_threads; ++i) counts_arrays[i].resize(centers.size());
  for (int i = 0; i < num_threads; ++i) {
    std::vector<glm::vec2>& center_array = center_arrays[i];
    std::vector<int>& counts_array = counts_arrays[i];
    threads[i] = std::thread(
        [&num_threads, &width, &height, &labels, &center_array,
         &counts_array](int tid) -> void {
          update_centers_thread(tid, num_threads, width, height, labels,
                                center_array, counts_array);
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

void assign_labels(int width, int height, const std::vector<glm::vec2>& centers,
                   std::vector<int>& labels, std::vector<float>& distances) {
  std::fill(distances.begin(), distances.end(),
            std::numeric_limits<float>::max());
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      glm::vec2 pixel(x, y);
      int pixel_index = x + y * width;
      int best = -1;
      float best_distance = std::numeric_limits<float>::max();
      nearest_neighbor(centers, pixel, &best_distance, &best);
      distances[pixel_index] = best_distance;
      labels[pixel_index] = best;
    }
  }
}

void update_centers(int width, int height, const std::vector<int>& labels,
                    std::vector<glm::vec2>& centers) {
  std::fill(centers.begin(), centers.end(), glm::vec2(0.0f, 0.0f));
  std::vector<int> counts(centers.size(), 0);
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      glm::vec2 pixel(x, y);
      int pixel_index = x + y * width;
      centers[labels[pixel_index]] += pixel;
      ++counts[labels[pixel_index]];
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
  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_real_distribution<> x_distribution(0, width);
  // std::uniform_real_distribution<> y_distribution(0, height);

  int num_threads = 8;
  labels.resize(width * height);
  auto start = std::chrono::high_resolution_clock::now();
  init_centers(num_threads, centers.size(), width, height, centers);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  // std::cout << "init:" << elapsed.count() << "\n";

  // std::generate(centers.begin(), centers.end(),
  //[&x_distribution, &y_distribution, &gen](void) -> glm::vec2 {
  // return glm::vec2(x_distribution(gen), y_distribution(gen));
  //});

  std::vector<glm::vec2> old_centers(centers.size());
  float diff = std::numeric_limits<float>::max();
  int iteration = 0;
  while (diff > 1e-5) {
    // print_vector(centers);
    std::copy(centers.begin(), centers.end(), old_centers.begin());
    kdtree::KdTree tree;
    tree.AssignPoints(old_centers);
    tree.Build();
    // std::cout << "Assign labels.\n";
    // assign_labels(width, height, centers, labels, distances);
    assign_labels_threaded(num_threads, width, height, tree, centers, labels);
    // std::cout << "Update centers.\n";
    // update_centers(width, height, labels, centers);
    update_centers_threaded(num_threads, width, height, labels, centers);
    diff = compute_difference(old_centers, centers);
    std::cout << "[" << iteration << "] diff = " << diff << "\n";
    ++iteration;
  }
}

void GenerateColorRamp(int num_centers, std::vector<image::Pixel>& colors) {
  float v = 0.5f;
  float s = 0.5f;
  float min_h = 0.0f;
  float max_h = 360.0f;
  float step = (max_h - min_h) / num_centers;
  float h = 0.0f;
  colors.resize(num_centers);
  for (int i = 0; i < num_centers; ++i) {
    glm::vec3 rgb = glm::rgbColor(glm::vec3(h, s, v));
    colors[i] = image::Pixel(static_cast<uint8_t>(255.0f * rgb[0]),
                             static_cast<uint8_t>(255.0f * rgb[1]),
                             static_cast<uint8_t>(255.0f * rgb[2]));
    h += step;
    std::cout << "color[" << i << "]=" << colors[i] << "\n";
  }
}

void WriteKmeansImage(const std::string& filename, int width, int height,
                      int num_centers, const std::vector<int>& labels) {
  std::vector<image::Pixel> ramp_look_up_table;
  GenerateColorRamp(num_centers, ramp_look_up_table);
  image::Image im(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int l = labels[y * width + x];
      if (!(l >= 0 && l < num_centers)) {
        std::cout << "l = " << l << " num_centers = " << num_centers << "\n";
        exit(0);
      }
      assert(l >= 0 && l < num_centers);

      im(x, y) = ramp_look_up_table[l];
    }
  }
  std::ofstream ofs(filename);
  std::string ppm_data;
  im.ToPpm(ppm_data);
  ofs << ppm_data;
  ofs.close();
}

int main(int argc, char** argv) {
  int num_centers = 2;
  int iteration = 0;
  std::vector<double> timings;
  int width = 640;
  int height = 480;
  std::vector<glm::vec2> centers(num_centers);
  std::vector<int> labels;
  // for (int i = 2; i <= 12000; i*=2) {
  num_centers = 8192;
  centers.resize(num_centers);
  auto start = std::chrono::high_resolution_clock::now();
  kmeans(width, height, centers, labels);
  std::chrono::duration<double> elapsed =
      std::chrono::high_resolution_clock::now() - start;
  ++iteration;
  assert(labels.size() == width * height);
  for (int i = 0; i < labels.size(); ++i)  {
      int l = labels[i];
      assert(l >= 0 && l < num_centers);
  }
  timings.push_back(elapsed.count());
  std::cout << iteration << " " << num_centers << " " << elapsed.count()
            << "\n";
  WriteKmeansImage("ppm/kmeans" + std::to_string(iteration) + ".ppm", width,
                   height, num_centers, labels);
  //}

  return 0;
}
