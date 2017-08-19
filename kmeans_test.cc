#include "kdtree.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

std::ostream& operator<<(std::ostream& out, const glm::vec2& c) {
  out << "(" << c.x << "," << c.y << ")";
  return out;
}

template <typename T>
void print_vector(const std::vector<T>& centers) {
  for (int i = 0; i < centers.size(); ++i) {
    std::cout << centers[i] << " ";
  }
  std::cout << "\n";
}

void assign_labels_thread(int tid, int num_threads, int width, int height,
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
    for (int i = 0; i < centers.size(); ++i) {
      float distance = glm::distance2(centers[i], pixel);
      if (distance < distances[j - t_start]) {
        distances[j - t_start] = distance;
        labels[j] = i;
      }
    }
  }
}

void assign_labels_threaded(int num_threads, int width, int height,
                            const std::vector<glm::vec2>& centers,
                            std::vector<int>& labels) {
  std::thread threads[num_threads];
  for (int i = 0; i < num_threads; ++i) {
    threads[i] = std::thread(
        [&num_threads, &width, &height, &centers, &labels](int tid) -> void {
          assign_labels_thread(tid, num_threads, width, height, centers,
                               labels);
        },
        i);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
}

void update_centers_thread(int tid, int num_threads, int width, int height,
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
    for (int i = 0; i < centers.size(); ++i) {
      float distance = glm::distance2(centers[i], pixel);
      if (distance < distances[j - t_start]) {
        distances[j - t_start] = distance;
        labels[j] = i;
      }
    }
  }
}

void update_centers_threaded(int num_threads, int width, int height,
                             const std::vector<glm::vec2>& centers,
                             std::vector<int>& labels) {
  std::thread threads[num_threads];
  std::vector<std::vector<glm::vec2>> center_arrays(num_threads);
  for (int i = 0; i < num_threads; ++i) center_arrays.resize(centers.size());

  for (int i = 0; i < num_threads; ++i) {
    std::vector<glm::vec2>& center_array = center_arrays[i];
    threads[i] = std::thread(
        [&num_threads, &width, &height, &center_array,
         &labels](int tid) -> void {
          update_centers_thread(tid, num_threads, width, height, center_array,
                                labels);
        },
        i);
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
}

void assign_labels(int width, int height, const std::vector<glm::vec2>& centers,
                   std::vector<int>& labels, std::vector<float>& distances) {
  std::fill(distances.begin(), distances.end(),
            std::numeric_limits<float>::max());
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      glm::vec2 pixel(x, y);
      int pixel_index = x + y * width;
      for (int i = 0; i < centers.size(); ++i) {
        float distance = glm::distance2(centers[i], pixel);
        if (distance < distances[pixel_index]) {
          distances[pixel_index] = distance;
          labels[pixel_index] = i;
        }
      }
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
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> x_distribution(0, width);
  std::uniform_real_distribution<> y_distribution(0, height);

  labels.resize(width * height);

  std::generate(centers.begin(), centers.end(),
                [&x_distribution, &y_distribution, &gen](void) -> glm::vec2 {
                  return glm::vec2(x_distribution(gen), y_distribution(gen));
                });

  std::vector<glm::vec2> old_centers(centers.size());
  bool changed = true;
  float diff = std::numeric_limits<float>::max();
  int iteration = 0;
  while (diff > 1e-5) {
    // print_vector(centers);
    std::copy(centers.begin(), centers.end(), old_centers.begin());
    std::cout << "Assign labels.\n";
    // assign_labels(width, height, centers, labels, distances);
    assign_labels_threaded(16, width, height, centers, labels);
    std::cout << "Update centers.\n";
    update_centers(width, height, labels, centers);
    diff = compute_difference(old_centers, centers);
    std::cout << "[" << iteration << "] diff = " << diff << "\n";
    ++iteration;
  }
}

int main(int argc, char** argv) {
  std::vector<glm::vec2> centers(1000);
  std::vector<int> labels;
  kmeans(640, 480, centers, labels);
  return 0;
}
