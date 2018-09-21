#include "image.h"
#include "kmeans_training_data.h"
#include "kmeans2d.h"

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
  double v = 0.5;
  double s = 0.5;
  double min_h = 0.0;
  double max_h = 360.0;
  double step = (max_h - min_h) / num_centers;
  double h = 0.0;
  colors.resize(num_centers);
  for (int i = 0; i < num_centers; ++i) {
    glm::vec3 rgb = glm::rgbColor(glm::vec3(h, s, v));
    colors[i] = image::Pixel(static_cast<uint8_t>(255.0 * rgb[0]),
                             static_cast<uint8_t>(255.0 * rgb[1]),
                             static_cast<uint8_t>(255.0 * rgb[2]));
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
  double* training_data = nullptr;
  double* training_labels = nullptr;
  int training_data_dim1 = 0;
  int training_data_dim2 = 0;
  int training_labels_dim1 = 0;
  int training_labels_dim2 = 0;
  double* average = nullptr;
  int average_dim1 = 0;
  int average_dim2 = 0;
  int average_dim3 = 0;
  std::vector<glm::vec2> centers;
  std::vector<double> centers_2;
  std::vector<int> indices;
  std::vector<int> order;
  std::vector<int> labels;
  std::vector<int> batch_sizes;
  int iteration = 0;
  std::vector<double> timings;
  for (int i = 1; i < 16; ++i) {
    num_centers = 0x1 << i;
    auto start = std::chrono::high_resolution_clock::now();
    // void KmeansDataAndLabels(const std::string& directory, int num_centers,
    // int& width, int& height, double** training_data,
    // int* training_data_dim1, int* training_data_dim2,
    // double** training_labels, int* training_labels_dim1,
    // int* training_labels_dim2,
    // std::vector<glm::vec2>& centers,
    // std::vector<int>& labels,
    // std::vector<int>& batch_sizes);
    int* closest;
    int closest_dim1, closest_dim2, closest_dim3;
//void KmeansDataAndLabels(
    //const std::string& directory, int num_centers, int& width, int& height,
    //double** training_data, int* training_data_dim1, int* training_data_dim2,
    //double** training_labels, int* training_labels_dim1,
    //int* training_labels_dim2, double** average_img, int* average_dim1,
    //int* average_dim2, int* average_dim3, int** closest, int* closest_dim1,
    //int* closest_dim2, int* closest_dim3, std::vector<int>& indices,
    //std::vector<int>& order, std::vector<glm::vec2>& centers,
    //std::vector<int>& labels, std::vector<int>& batch_sizes);
    KmeansDataAndLabels(
        dirname, num_centers, width, height, &training_data,
        &training_data_dim1, &training_data_dim2, &training_labels,
        &training_labels_dim1, &training_labels_dim2, &average, &average_dim1,
        &average_dim2, &average_dim3, &closest, &closest_dim1, &closest_dim2,
        &closest_dim3, indices, order, centers, labels, batch_sizes);
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
