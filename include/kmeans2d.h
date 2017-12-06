#pragma once

#include <vector>

void kmeans2d(int width, int height, std::vector<float>& centers,
              std::vector<int>& labels);

void kmeans_training_data(const std::string& directory, int num_centers,
                          int* width, int* height, std::vector<int>& indices,
                          std::vector<float>& centers, std::vector<int>& labels,
                          std::vector<int>& batch_sizes, float** train_data,
                          int* train_data_dim1, int* train_data_dim2,
                          float** train_labels, int* train_labels_dim1,
                          int* train_labels_dim2);

void closest_n(int width, int height, int n,
               std::vector<float>& centers, int** closest, int* dim1, int* dim2,
               int* dim3);
