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

#include "quickselect.h"

template <typename GlmType, int Dimension>
struct GlmComparator {
  GlmComparator() {}
  bool operator()(const GlmType& a, const GlmType& b) const {
    return a[Dimension] < b[Dimension];
  }
};

struct KdNode {
  KdNode()
      : type(kInternal),
        left(-1),
        right(-1),
        split_dimension(0),
        split_value(0.0f),
        index(-1) {}
  KdNode(const KdNode& node)
      : type(node.type),
        left(node.left),
        right(node.right),
        split_dimension(node.split_dimension),
        split_value(node.split_value),
        index(node.index) {}
  enum Type {
    kInternal,
    kLeaf
  };
  Type type;
  int split_dimension;
  float split_value;
  int index;
};

class KdTree {
 public:
  KdTree() {}
  void AddPoints(const std::vector<glm::vec2>& input_points) {
    points.resize(input_points.size());
    std::copy(input_points.begin(), input_points.end(), points_.begin());
  }

  void ComputeBounds() {
    min_ = glm::vec2(std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max());
    max_ = glm::vec2(-std::numeric_limits<float>::max(),
                     -std::numeric_limits<float>::max());
    for (auto& point : points_) {
      min_.x = std::min(point.x, min_.x);
      min_.y = std::min(point.y, min_.y);
      max_.x = std::max(point.x, max_.x);
      max_.y = std::max(point.y, may_.y);
    }
  }

  void RecursiveBuild(int first, int last, int dim, glm::vec2& min_values,
      glm::vec2& max_values) {
    KdNode node;
    if (last - first  == 1) {
      node.type = KdNode::kLeaf;
      node.point = first;
    } else {

      if (split_index > first)
        RecursiveBuild(first, split_index, (dim + 1) % 2);
      if (split_index < last) RecursiveBuild(split_index, last, (dim + 1) % 2);
    }
  }

  void Build() { RecursiveBuild(0, points_.size(), 0); }
  int RecursiveNearestNeighbor(const glm::vec2 query, const KdNode& node) {
  }

 protected:
  glm::vec2 min_;
  glm::vec2 max_;
  std::vector<glm::vec2> points_;
  std::vector<KdNode> nodes_;
};

std::ostream& operator<<(std::ostream& out, const glm::vec2& c) {
  out << "(" << c.x << "," << c.y << ")";
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
        [&num_threads, &width, &height, &centers, &labels ](int tid)->void {
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
    threads[i] = std::thread(
        [&num_threads,
         &width,
         &height,
         &centers_arrays[i],
         &labels ](int tid)
                      ->void {
          update_centers_thread(tid, num_threads, width, height,
                                center_arrays[i], labels);
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
                [&x_distribution, &y_distribution, &gen ](void)->glm::vec2 {
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
