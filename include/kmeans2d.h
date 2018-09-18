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
                          int* train_labels_dim2, float** average,
                          int* average_dim1, int* average_dim2,
                          int* average_dim3);

void closest_n(int width, int height, int n, std::vector<float>& centers,
               int** closest, int* dim1, int* dim2, int* dim3);

// k - get the k closest pixels
// cluster_id - centroid identifier
// closest - contains closest_dim3 centroids for each (x, y)
//- in test_dim1 x test_dim2 pixels.
// closest_dim1 - height (y)
// closest_dim2 - width  (x)
// closest_dim3 - ensemble_size (usually 5)
// train_data - train data
// train_data_dim1 - number of input train points
// train_data_dim2 - dimension of train points
// target_data - true values corresponding to the train points
// target_dim1 - numer of input target points, should be same as train_data_dim1
// target_dim2 - dimension of target points
// test - output test data
// test_data_dim1 - output test points
// test_data_dim2 - dimension of output test points, should be same as
//- test_data_dim2
// target - output target data
// target_data_dim1 - output target points
// target_data_dim2 - dimension of output target points, should be same as
//- target_data_dim2
void closest_k_test_target(int k, int cluster_id, int* closest,
                           int closest_dim1, int closest_dim2, int closest_dim3,
                           float* train_data, int train_data_dim1,
                           int train_data_dim2, float* target_data,
                           int target_data_dim1, int target_data_dim2,
                           float** test, int* test_dim1, int* test_dim2,
                           float** target, int* target_dim1, int* target_dim2);

// order - specifies order of each image for purposes of sampling input dataset
//      - if an image is not chosen, the value of order[i] = -1
//      - if an image is chosen, then the value is order[i] = n
//      - where i is now the n-th image number that was chosen
// test  - test data for the neural network ensembles
//      - of the form: xyirgb  xyirgb xyirgb ...
//      - where xy are the image pixel coordinates normalied
//      - i is the light location, in this case just the image number
//      - rgb is the average pixel rgb value.
// test_dim1 - number of test data elements
// test_dim2 - should always be 6
// predictions - image predictions to be made using the input test data
// predictions_dim1 - number of predictions, should match test_dim1
//                 - of the form: rgb rgb rgb...
// predictions_dim2 - should be 3
void predictions_to_errors(std::vector<int>& order, int ensemble_size,
                           float* test, int test_dim1, int test_dim2,
                           float* target, int target_dim1, int target_dim2,
                           float* predictions, int predictions_dim1,
                           int predictions_dim2, float* errors, int errors_dim1,
                           int errors_dim2);

// assignment_data_to_test_data
//
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
// Ensemble data: [[L, i, s, n], ... ]
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
    int* test_data_dim2, int** ensemble_data, int* ensemble_data_dim1,
    int* ensemble_data_dim2);

// predictions_to_image
//
//  image_out - image out
//  image_out_dim1 - width
//  image_out_dim2 - height
//  image_out_dim3 - channels
//  test - test data
//  test_dim1 - num test values
//  test_dim2 - test value size
//  predictions - predictions to put into image out
//  predictions_dim1 - num predictions
//  predictions_dim2 - prediction size
void predictions_to_image(float* image_out, int image_out_dim1,
                          int image_out_dim2, int image_out_dim3, float* test,
                          int test_dim1, int test_dim2, float* predictions,
                          int predictions_dim1, int predictions_dim2);

void train_network(const string& save_file, float* train_data,
                   int train_data_dim1, int train_data_dim2,
                   float* train_labels, int train_labels_dim1,
                   int train_labels_dim2, int num_hidden_nodes);
