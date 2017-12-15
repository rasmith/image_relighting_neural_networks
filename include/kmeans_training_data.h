#pragma once

#include <string>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>

void KmeansDataAndLabels(
    const std::string& directory, int num_centers, int& width, int& height,
    float** training_data, int* training_data_dim1, int* training_data_dim2,
    float** training_labels, int* training_labels_dim1,
    int* training_labels_dim2, float** average_img, int* average_dim1,
    int* average_dim2, int* average_dim3, std::vector<int>& indices,
    std::vector<int>& order, std::vector<glm::vec2>& centers,
    std::vector<int>& labels, std::vector<int>& batch_sizes);
