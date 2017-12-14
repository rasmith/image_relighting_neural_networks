#pragma once

#include <vector>

void kmeans2d(int width, int height, std::vector<float>& centers,
              std::vector<int>& labels);

void kmeans_training_data(const std::string& directory, int num_centers,
                          int* width, int* height, std::vector<int>& indices,
                          std::vector<int>& order, std::vector<float>& centers,
                          std::vector<int>& labels,
                          std::vector<int>& batch_sizes, float** train_data,
                          int* train_data_dim1, int* train_data_dim2,
                          float** train_labels, int* train_labels_dim1,
                          int* train_labels_dim2);

void closest_n(int width, int height, int n, std::vector<float>& centers,
               int** closest, int* dim1, int* dim2, int* dim3);

/*
* k - get the k closest pixels
* cluster_id - centroid identifier
* closest - contains closest_dim3 centroids for each (x, y)
*         - in test_dim1 x test_dim2 pixels.
* closest_dim1 - width
* closest_dim2 - height
* train_data - train data
* train_data_dim1 - number of input train points
* train_data_dim2 - dimension of train points
* target_data - true values corresponding to the train points
* target_dim1 - numer of input target points, should be same as train_data_dim1
* target_dim2 - dimension of target points
* test - output test data
* test_data_dim1 - output test points
* test_data_dim2 - dimension of output test points, should be same as
*                - test_data_dim2
* target - output target data
* target_data_dim1 - output target points
* target_data_dim2 - dimension of output target points, should be same as
*                  - target_data_dim2
*/
void closest_k_test_target(int k, int cluster_id, float* closest,
                           int closest_dim1, int closest_dim2, int closest_dim3,
                           float* train_data, int train_data_dim1,
                           int train_data_dim2, float* target_data,
                           int target_data_dim1, int target_data_dim2,
                           float** test, int* test_dim1, int* test_dim2,
                           float** target, int* target_dim1, int* target_dim2);

void predictions_to_images(std::vector<int>& order, float* test, int test_dim1,
                           int test_dim2, float* predictions,
                           int predictions_dim1, int predictions_dim2,
                           float* predicted_images, int predicted_images_dim1,
                           int predicted_images_dim2, int predicted_images_dim3,
                           int predicted_images_dim4);
