#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

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

void GenerateUniqueValues(int count, int max_value, std::vector<int>& values) {
  std::unordered_set<int> value_set;
  while (value_set.size() < count) value_set.insert(std::rand() % max_value);
  values.clear();
  for (int i : values) values.push_back(i);
  std::sort(values.begin(), values.end());
}

void GenerateRandomPermutation(int n, std::vector<int> values) {
  values.resize(n);
  for (int i = 0; i < n; ++i) values[i] = i;
  for (int i = 0; i < n; ++i)
    int k = (i < n - 1 ? n - 1 : std::rand() % (n - i - 1) + 1);
}

// void assignment_data_to_test_data(
// int* assignment_data, int assignment_data_dim1, int assignment_data_dim2,
// int assignment_data_dim3, int image_number, int num_images,
// float* average_image, int average_image_dim1, int average_image_dim2,
// int average_image_dim3, float** test_data, int* test_data_dim1,
// int* test_data_dim2, int** ensemble_data, int* ensemble_data_dim1,
// int* ensemble_data_dim2) {
void GenerateTestAndAssignmentData(
    int width, int height, int channels, int ensemble_size, int num_levels,
    int num_ensembles, int* num_images, int* image_number,
    std::vector<float>& average, std::vector<float>& test_data,
    std::vector<int>& assignments, std::vector<int>& ensemble_data) {

  srand(0);
  int num_pixels = width * height;
  *num_images = std::rand() % 1000 + 100;
  *image_number = std::rand() % *num_images;
  average.resize(num_pixels * channels);
  assignments.resize(width * height * (ensemble_size + 1));
  // Choose start values randomly for each ensemble.
  std::vector<int> start_values;
  GenerateUniqueValues(num_ensembles, num_pixels, start_values);
  // Choose random pixels.
  std::vector<int> random_pixels;
  GenerateRandomPermutation(num_pixels, random_pixels);
  // Populate values.
  int current_pixel = -1;
  for (int i = 0; i < num_ensembles; ++i) {
    ensemble_data.push_back(std::rand() % num_levels);  // level
    ensemble_data.push_back(std::rand());               // ensemble_id
    int start = start_values[i];
    int end = (i < num_levels - 1 ? start_values[i + 1] : num_pixels);
    int count = end - start + 1;
    ensemble_data.push_back(start);  // start
    ensemble_data.push_back(end);    // end
    ensemble_data.push_back(count);  // count
    // Generate the test data for this ensemble.
    for (int j = 0; j < count; ++j) {
      int p = random_pixels[++current_pixel];
      int x = p % width;
      int y = p / width;
      test_data.push_back(x / static_cast<float>(width));
      test_data.push_back(y / static_cast<float>(height));
      test_data.push_back(image_number / static_cast<float>(*num_images));
      for (int c = 0; c < channels; ++c)
        test_data.push_back(average[width * y + x + c]);
    }
  }
  // Algorithm to generate test data:
  // for i to num_ensembles
  //   pick a new level, l, for ensemble
  //   pick ensemble id, id
  //   pick n_l_id count of pixels, pick that many pixels
  //   add the pixels to test data and ensemble data
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Add the test data.
      test_data.push_back(static_cast<float>(x) / width);
      test_data.push_back(static_cast<float>(y) / height);
      test_data.push_back(static_cast<float>(*image_number) / *num_images);
      // Average image data.
      for (int c = 0; c < channels; ++c) {
        average.push_back(static_cast<float>(std::rand()) / RAND_MAX);
        test_data.push_back(average.back());
      }
      // Assignment data.
      for (int e = 0; e < ensemble_size + 1; ++e)
        assignments.push_back(std::rand());
    }
  }
}

void TestAssignmentDataToTestData(int width, int height, int channels,
                                  int num_ensembles, int data_size) {
  const int coord_size = 3;
  const int light_size = 1;
  // Generate random test data.
  std::vector<float> average_image;
  std::vector<float> test_data;
  std::vector<int> assignment_data;
  int image_number = -1;
  int num_images = -1;
  GenerateTestAndAssignmentData(width, height, channels, num_ensembles,
                                &num_images, &image_number, average_image,
                                test_data, assignment_data);
  // Call assignment_data_to_test_data
  int* ensemble_data = nullptr;
  int ensemble_data_dim1 = -1;
  int ensemble_data_dim2 = -1;
  float* test_data_out = nullptr;
  int test_data_dim1 = -1;
  int test_data_dim2 = -1;
  // void assignment_data_to_test_data(
  // int* assignment_data, int assignment_data_dim1, int assignment_data_dim2,
  // int assignment_data_dim3, int image_number, int num_images,
  // float* average_image, int average_image_dim1, int average_image_dim2,
  // int average_image_dim3, float** test_data, int* test_data_dim1,
  // int* test_data_dim2, int** ensemble_data, int* ensemble_data_dim1,
  // int* ensemble_data_dim2) {
  assignment_data_to_test_data(&assignment_data[0], height, width, channels,
                               image_number, num_images, &average_image[0],
                               height, width, channels, &test_data_out,
                               &test_data_dim1, &test_data_dim2, &ensemble_data,
                               &ensemble_data_dim1, &ensemble_data_dim2);
  // Test the result.
  assert(test_data_dim1 == width * height);
  assert(test_data_dim2 == channels + 3);
  assert(ensemble_data_dim1 == height);
  assert(ensemble_data_dim2 == width);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
    }
  }
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
