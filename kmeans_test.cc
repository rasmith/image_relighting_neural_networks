#include "image.h"
#include "kmeans_training_data.h"

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
  std::string dirname =
      "/Users/randallsmith/image_relighting_neural_networks/data/bull/rgb";
  int width = 640;
  int height = 480;
  float* training_data = nullptr;
  float* training_labels = nullptr;
  int training_data_dim1 = 0;
  int training_data_dim2 = 0;
  int training_labels_dim1 = 0;
  int training_labels_dim2 = 0;
  std::vector<glm::vec2> centers;
  std::vector<int> indices;
  std::vector<int> labels;
  std::vector<int> batch_sizes;
  int iteration = 0;
  std::vector<double> timings;
  for (int i = 1; i < 16; ++i) {
    num_centers = 0x1 << i;
    auto start = std::chrono::high_resolution_clock::now();
    // void KmeansDataAndLabels(const std::string& directory, int num_centers,
    // int& width, int& height, float** training_data,
    // int* training_data_dim1, int* training_data_dim2,
    // float** training_labels, int* training_labels_dim1,
    // int* training_labels_dim2,
    // std::vector<glm::vec2>& centers,
    // std::vector<int>& labels,
    // std::vector<int>& batch_sizes);
    KmeansDataAndLabels(dirname, num_centers, width, height, &training_data,
                        &training_data_dim1, &training_data_dim2,
                        &training_labels, &training_labels_dim1,
                        &training_labels_dim2, indices, centers, labels,
                        batch_sizes);
    std::chrono::duration<double> elapsed =
        std::chrono::high_resolution_clock::now() - start;
    ++iteration;
    timings.push_back(elapsed.count());
    std::cout << iteration << " " << num_centers << " " << elapsed.count()
              << "\n";
    WriteKmeansImage("ppm/kmeans" + std::to_string(iteration) + ".ppm", width,
                     height, num_centers, labels);
  }
  return 0;
}
