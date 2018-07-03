#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kmeans2d.h"

struct TestData {
  int a[6];
};

struct CompareTestData {
  bool operator()(const TestData& a, const TestData& b) {
    for (int i = 0; i < 6; ++i)
      if (a.a[i] < b.a[i]) return true;
    return false;
  }
};

struct NetworkData {
  int level;
  int id;
  int start;
  int count;
  bool operator==(const NetworkData& a) {
    return (this == &a) || (level == a.level && id == a.id &&
                            start == a.start && count == a.count);
  }
  bool operator!=(const NetworkData& a) { return !((*this) == a); }
};

static_assert(sizeof(NetworkData) == 4 * sizeof(int),
              "NetworkData size is too large.");

std::ostream& operator<<(std::ostream& out, const NetworkData& d) {
  std::cout << "{level:" << d.level << ", id:" << d.id << ", start:" << d.start
            << ", count:" << d.count << "}\n";
  return out;
}

struct CompareNetworkData {
  bool operator()(const NetworkData& a, const NetworkData& b) {
    if (a.level < b.level) return true;
    if (a.id < b.id) return true;
    if (a.start < b.start) return true;
    if (a.count < b.count) return true;
    return false;
  }
};

void CopyNetworkData(const int* to_pos, std::vector<NetworkData>& data) {
  const int step_size = sizeof(NetworkData) / sizeof(int);
  const int* pos = to_pos;
  for (int i = 0; i < data.size(); ++i) {
    data[i] = *reinterpret_cast<const NetworkData*>(pos);
    pos += step_size;
  }
}

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

void GenerateTestAndAssignmentData(
    int width, int height, int channels, int ensemble_size, int num_levels,
    int num_networks, int* num_images, int* image_number,
    std::vector<float>& average, std::vector<float>& test_data,
    std::vector<int>& assignments, std::vector<int>& network_data) {
  const int pixel_size = 3;                                    // set pixel size
  const int light_size = 1;                                    // set light size
  const int coord_size = 2;                                    // set coord size
  const int data_size = pixel_size + light_size + coord_size;  // set data size
  const int network_data_size = 4;  // level, id, start, count
  const int assignment_data_size = network_data_size + 1;
  int num_pixels = width * height;  // number of pixels

  // Get a random order of pixels.
  std::vector<int> pixels;
  GenerateRandomPermutation(num_pixels, pixels);

  // Allocate test_data, network_data, and assignments.
  test_data.resize(num_pixels * ensemble_size * data_size);  // allocate
  network_data.resize(num_networks * network_data_size);
  assignments.resize(num_pixels * assignment_data_size);

  // Set the num_images and image_number randomly.
  *num_images = std::rand() % 1000 + 100;     // set number of images randomly
  *image_number = std::rand() % *num_images;  // set image number randomly

  int networks_left = num_networks;
  float* test_pos = &test_data[0];
  int* network_data_pos = &network_data[0];
  for (int level = 0; level < num_levels; ++level) {
    // 0. Compute number of neural networks, pixels to use, and limits on
    // maximum pixels any neural network can be assigned.
    int networks_per_level =
        std::min(num_networks / num_levels + 1, networks_left);
    int pixels_per_level =
        std::min(num_pixels / num_levels + 1, static_cast<int>(pixels.size()));
    int pixels_per_network = std::min(pixels_per_level / networks_per_level + 1,
                                      static_cast<int>(pixels.size()));
    int max_pixels =
        std::ceil(static_cast<float>(network_data_size * pixels_per_level) /
                  networks_per_level);
    networks_left -= networks_per_level;
    for (int i = 0; i < networks_per_level; ++i) {
      int offset = test_pos - &test_data[0];
      network_data[network_data_size * i] = level;
      network_data[network_data_size * i + 1] = i;
      network_data[network_data_size * i + 2] =
          offset + pixels_per_network * networks_per_level;
      network_data[network_data_size * i + 3] = 0;
    }
    // 1. Assign pixels to this level.
    std::vector<int> local_pixels;
    std::copy(pixels.end() - pixels_per_level, pixels.end(),
              std::back_inserter(local_pixels));
    pixels.erase(pixels.end() - pixels_per_level, pixels.end());
    // 2. Assign networks to pixels.
    std::vector<int> network_counts(networks_per_level, 0);
    int current_network = 0;
    for (int i = 0; i < local_pixels.size(); ++i) {
      int pixel_index = local_pixels[i];
      int x = pixel_index % width;
      int y = pixel_index / width;
      assignments[assignment_data_size * (width * x + y)] = level;
      for (int j = 0; j < ensemble_size; ++j) {
        int offset =
            (current_network * pixels_per_network +
             network_data_pos[current_network * network_data_size + 3]) *
            data_size;
        test_pos[offset] = static_cast<float>(x) / width;
        test_pos[offset + 1] = static_cast<float>(y) / height;
        test_pos[offset + 2] = static_cast<float>(*image_number) / *num_images;
        test_pos[offset + 3] = average[width * y + x];
        test_pos[offset + 4] = average[width * y + x + 1];
        test_pos[offset + 5] = average[width * y + x + 2];
        ++network_counts[current_network * network_data_size + 3];
        assignments[assignment_data_size * (width * x + y) + j + 1] =
            current_network;
        current_network = (current_network + 1) % networks_per_level;
      }
    }
    test_pos += data_size * pixels_per_level;
    network_data_pos += network_data_size * networks_per_level;
  }
}

void TestAssignmentDataToTestData(int width, int height, int channels,
                                  int num_levels, int num_networks,
                                  int ensemble_size, int data_size) {
  const int coord_size = 3;
  const int light_size = 1;
  // Generate random test data.
  std::vector<float> average_image;
  std::vector<float> test_data;
  std::vector<int> network_data;
  std::vector<int> assignment_data;
  int image_number = -1;
  int num_images = -1;
  // Call assignment_data_to_test_data
  int* network_data_out = nullptr;
  int network_data_dim1 = -1;
  int network_data_dim2 = -1;
  float* test_data_out = nullptr;
  int test_data_dim1 = -1;
  int test_data_dim2 = -1;

  GenerateTestAndAssignmentData(width, height, channels, ensemble_size,
                                num_levels, num_networks, &num_images,
                                &image_number, average_image, test_data,
                                assignment_data, network_data);

  assignment_data_to_test_data(
      &assignment_data[0], height, width, channels, image_number, num_images,
      &average_image[0], height, width, channels, &test_data_out,
      &test_data_dim1, &test_data_dim2, &network_data_out, &network_data_dim1,
      &network_data_dim2);

  // Test the result.
  assert(test_data_dim1 == width * height * 4);
  assert(test_data_dim2 == channels + 3);
  assert(network_data_dim1 == height);
  assert(network_data_dim2 == width);

  std::vector<NetworkData> network_data_check(network_data_dim1);
  std::vector<NetworkData> network_data_test(network_data_dim1);

  // Copy in data.
  CopyNetworkData(network_data_out, network_data_test);
  CopyNetworkData(&network_data[0], network_data_check);

  // Sort so a comparison can be made.
  std::sort(network_data_check.begin(), network_data_check.end(),
            CompareNetworkData());
  std::sort(network_data_test.begin(), network_data_test.end(),
            CompareNetworkData());

  assert(network_data_check.size() == network_data_test.size());
  for (int i = 0; i < network_data_test.size(); ++i) {
    if (network_data_test[i] != network_data_check[i]) {
      std::cout << "Test value " << i << " does not match check value "
                << network_data_test[i] << " " << network_data_check[i] << "\n";
      assert(network_data_test[i] == network_data_check[i]);
    }
  }
}

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
