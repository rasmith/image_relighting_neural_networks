#include <utility>
#include <queue>
#include <vector>
#include <iostream>

#include "kmeans2d.h"

// void predictions_to_image(float* image_out, int image_out_dim1,
// int image_out_dim2, int image_out_dim3, float* test,
// int test_dim1, int test_dim2, float* predictions,
// int predictions_dim1, int predictions_dim2);
void GenerateRandomImage(int width, int height, int channels,
                         std::vector<float>& image) {
  for (int i = 0; i < width * height * channels; ++i)
    image[i] = std::rand() / static_cast<float>(RAND_MAX);
}

void GenerateTestData(int width, int height, int channels, int num_samples,
                      const std::vector<float>& image,
                      std::vector<float>& test) {
  // Pair int-int to use to generate random samples.
  typedef std::pair<int, int> IntInt;
  auto cmp = [](IntInt a, IntInt b) { return a.second < b.second; };
  // Priority queue to generate random samples into.
  std::priority_queue<IntInt, std::vector<IntInt>, decltype(cmp)> q(cmp);
  // Generate the samples.
  for (int i = 0; i < width * height; ++i)
    q.push(std::make_pair(i, std::rand()));
  // Create test data (x, y, *, *, *, *) values.  Mainly for the x-y
  // values.
  int test_data_size = channels + 3;
  test.resize(num_samples * test_data_size);
  for (int i = 0; i < num_samples * test_data_size; i += test_data_size) {
    int k = q.top().first;
    int x = k % width;
    int y = k / width;
    q.pop();
    test[i] = x / static_cast<float>(width);
    test[i + 1] = y / static_cast<float>(height);
  }
}

void TestPredictionsToImage(int width, int height, int channels,
                            int num_samples, uint8_t* bytes) {
  // Generate a random image to test against.
  std::vector<float> image;
  GenerateRandomImage(width, height, channels, image);
  // Make some test data
  std::vector<float> test;
  GenerateTestData(width, height, channels, num_samples, image, test);
  // Generate the "predictions".
  int test_data_size = channels + 3;
  std::vector<float> predictions(num_samples * channels);
  for (int i = 0; i < num_samples * test_data_size; i += test_data_size) {
    int x = test[i];
    int y = test[i + 1];
    for (int j = 0; j < channels; ++j)
      predictions[i + j] = image[channels * (y * width + x) + j];
  }
  // Run the target function to test.
  std::vector<float> image_out;
  predictions_to_image(&image_out[0], width, height, channels, &test[0],
                       num_samples, channels + 3, &predictions[0], num_samples,
                       channels);
}

int main(int argc, char** argv) { return 0; }
