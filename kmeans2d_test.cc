#include <utility>
#include <queue>
#include <vector>
#include <iostream>

#include "kmeans2d.h"

void GenerateRandomImage(int width, int height, int channels,
                         std::vector<float>& image) {
  image.resize(width * height * channels);
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

std::string PixelToString(float* p) {
  return "[" + std::to_string(p[0]) + "," + std::to_string(p[1]) + "," +
         std::to_string(p[2]) + "]";
}

bool EqualsPixel(float* p, float* q) {
  return p[0] == q[0] && p[1] == q[1] && p[2] == q[2];
}
// void assignment_data_to_test_data(
// int* assignment_data, int assignment_data_dim1, int assignment_data_dim2,
// int assignment_data_dim3, int image_number, int num_images,
// float* average_image, int average_image_dim1, int average_image_dim2,
// int average_image_dim3, float** test_data, int* test_data_dim1,
// int* test_data_dim2, int** ensemble_data, int* ensemble_data_dim1,
// int* ensemble_data_dim2) {

void GenerateTestAndAssignmentData(int width, int height, int channels,
                                   int ensemble_size,
                                   std::vector<float>& test_data,
                                   std::vector<float>& assignment_data) {

  int num_samples = width * height;
  std::priority_queue<uint64_t> random_pixels;
  // Generate a random permutation.
  srand(0);
  for (int i = 0; i < width * height; ++i)
    random_pixels.push((static_cast<uint64_t>(std::rand()) << 32) &
                       static_cast<uint64_t>(i));
  // Populate assignments.
  std::vector<int> assignments(width * height * ensemble_size);
  for (int i = 0; i < width * height; ++i) {
    int xy = static_cast<int>(0x0000FFFF & random_pixels.top()), x = xy % width,
        y = xy / width;
    random_pixels.pop();
    assignments[ensemble_size * (y * width + x)] = std::rand() % ensemble_size;
    for (int j = 0; j < ensemble_size; ++j)
      assignments[ensemble_size * (y * width + x) + j + 1] =
          std::rand() % ensemble_size;
  }
}

void TestAssignmentDataToTestData(int width, int height, int channels,
                                  int num_ensembles, int data_size) {
  // Make random average image.
  std::vector<float> average_image;
  GenerateRandomImage(width, height, channels, average_image);
  // Make random image.
  std::vector<float> image;
  GenerateRandomImage(width, height, channels, image);
  // Generate random test data.
  std::vector<float> test_data;
  std::vector<float> assignment_data;
  GenerateTestAndAssignmentData(width, height, channels, num_ensembles,
                                test_data, assignment_data);
}

// void predictions_to_image(float* image_out, int image_out_dim1,
// int image_out_dim2, int image_out_dim3, float* test,
// int test_dim1, int test_dim2, float* predictions,
// int predictions_dim1, int predictions_dim2);

bool TestPredictionsToImage(int width, int height, int channels,
                            int num_samples) {
  // Generate a random image to test against.
  std::vector<float> image;
  GenerateRandomImage(width, height, channels, image);
  // Make some test data
  std::vector<float> test;
  GenerateTestData(width, height, channels, num_samples, image, test);
  // Generate the "predictions".
  int test_data_size = channels + 3;
  std::vector<float> predictions(num_samples * test_data_size);
  for (int i = 0; i < num_samples * test_data_size; i += test_data_size) {
    int x = test[i];
    int y = test[i + 1];
    for (int j = 0; j < channels; ++j)
      predictions[i + j] = image[channels * (y * width + x) + j];
  }
  // Run the target function to test.
  std::vector<float> image_out(width * height * channels);
  predictions_to_image(&image_out[0], width, height, channels, &test[0],
                       num_samples, channels + 3, &predictions[0], num_samples,
                       channels);
  for (int i = 0; i < num_samples * test_data_size; i += test_data_size) {
    int x = test[i];
    int y = test[i + 1];
    float* target_pixel = &image[channels * (y * width + x)];
    float* test_pixel = &image[channels * (y * width + x)];
    if (!EqualsPixel(target_pixel, test_pixel)) {
      std::cout << "TestPredictionsToImage: (" << x << "," << y
                << ") does not match. Target = " << PixelToString(target_pixel)
                << ". Test = " << PixelToString(test_pixel) << "\n";
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  int x = 180;
  int y = 240;
  int c = 3;
  struct Resolution {
    int width;
    int height;
  };
  Resolution resolutions[] = {{800, 600},
                              {1024, 600},
                              {1024, 768},
                              {1152, 864},
                              {1280, 720},
                              {1280, 800},
                              {1280, 1024},
                              {1360, 768},
                              {1366, 768},
                              {1440, 900},
                              {1536, 864},
                              {1600, 900},
                              {1680, 1050},
                              {1920, 1080},
                              {1920, 1200},
                              {2560, 1080},
                              {2560, 1440},
                              {3440, 1440},
                              {3840, 2160}};

  for (auto r : resolutions) {
    int step = r.width * r.height / 10;
    for (int i = 1; i < r.width * r.height; i += step) {
      std::cout << "-------- TestPredictionsToImage ------ (" << r.width << ","
                << r.height << ") - " << i << " -----\n";
      if (!TestPredictionsToImage(r.width, r.height, c, i)) {
        exit(0);
      }
    }
  }
  return 0;
}
