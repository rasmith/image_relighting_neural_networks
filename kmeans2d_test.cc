#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>
#include "kmeans2d.h"

#define CHECK_BOUNDS(x, i, n)                                            \
  {                                                                      \
    auto& d = (x)[(i)];                                                  \
    int extent = d.start + d.count;                                      \
    bool bounds_check = extent < (n);                                    \
    if (!bounds_check) {                                                 \
      std::cout << __LINE__ << ": Bounds check failed at " << (i)        \
                << ". Limit is " << (n) << " but got " << extent << "."; \
    }                                                                    \
  }

struct TestData {
  float x;
  float y;
  float i;
  float r;
  float g;
  float b;
  TestData() : x(0.0f), y(0.0f), i(0.0f), r(0.0f), g(0.0f), b(0.0f) {}
  TestData(int ix, int iy, int in, const float* rgb, int width, int height,
           int num_images)
      : x(static_cast<float>(ix) / width),
        y(static_cast<float>(iy) / height),
        i(static_cast<float>(in) / num_images),
        r(rgb[0]),
        g(rgb[1]),
        b(rgb[2]) {}
  bool operator==(const TestData& a) {
    return (this == &a) || (x == a.x && y == a.y && i == a.i && r == a.r &&
                            g == a.g && b == a.b);
  }
  bool operator!=(const TestData& a) { return !((*this) == a); }
};

struct CompareTestData {
  bool operator()(const TestData& a, const TestData& b) {
    const float* aa = &(a.x);
    const float* ab = &(b.x);
    for (int i = 0; i < 6; ++i)
      if (aa[i] < ab[i]) return true;
    return false;
  }
};

struct NetworkData {
  int level;
  int id;
  int start;
  int count;
  NetworkData() : level(0), id(0), start(0), count(0) {}
  NetworkData(int l, int i, int s, int c)
      : level(l), id(i), start(s), count(c) {}
  bool EqualsLevelIdCount(const NetworkData& d) {
    return (this == &d) || (level == d.level && id == d.id && count == d.count);
  }
  bool operator==(const NetworkData& a) {
    return (this == &a) || (level == a.level && id == a.id &&
                            start == a.start && count == a.count);
  }
  bool operator!=(const NetworkData& a) { return !((*this) == a); }
};

std::ostream& operator<<(std::ostream& out, const NetworkData& d) {
  std::cout << "{level:" << d.level << ", id:" << d.id << ", start:" << d.start
            << ", count:" << d.count << "}\n";
  return out;
}

std::ostream& operator<<(std::ostream& out, const TestData& d) {
  std::cout << "{x:" << d.x << ", y:" << d.y << ", i:" << d.i << ", r:" << d.r
            << ", g:" << d.g << ", b:" << d.b << "}\n";
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

template <typename PrimitiveType, typename StructType, int StructMultipleSize>
void CopyFromPrimitiveArrayToStructArray(const PrimitiveType* to_pos,
                                         std::vector<StructType>& data) {
  constexpr int sizeof_struct = sizeof(StructType);
  constexpr int sizeof_mult = StructMultipleSize * sizeof(PrimitiveType);
  static_assert(sizeof_struct == sizeof_mult, "StructType size is incorrect.");
  const int step_size = sizeof(StructType) / sizeof(PrimitiveType);
  const PrimitiveType* pos = to_pos;
  for (int i = 0; i < data.size(); ++i) {
    data[i] = *reinterpret_cast<const StructType*>(pos);
    pos += step_size;
  }
}

void CopyNetworkData(const int* to_pos, std::vector<NetworkData>& data) {
  CopyFromPrimitiveArrayToStructArray<int, NetworkData, 4>(to_pos, data);
}

void CopyTestData(const float* to_pos, std::vector<TestData>& data) {
  CopyFromPrimitiveArrayToStructArray<float, TestData, 6>(to_pos, data);
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
    std::vector<float>& average, std::vector<TestData>& test_data,
    std::vector<int>& assignments, std::vector<NetworkData>& network_data) {
  std::cout << "GenerateTestAndAssignmentData\n";
  const int assignment_data_size = ensemble_size + 1;
  int num_pixels = width * height;  // number of pixels

  // Get a random order of pixels.
  std::vector<int> pixels;
  GenerateRandomPermutation(num_pixels, pixels);

  // Allocate test_data, network_data, and assignments.
  test_data.clear();
  network_data.resize(num_networks);
  assignments.resize(num_pixels * assignment_data_size);

  // Set the num_images and image_number randomly.
  *num_images = std::rand() % 1000 + 100;     // set number of images randomly
  *image_number = std::rand() % *num_images;  // set image number randomly

  int networks_left = num_networks;
  NetworkData* network_pos = &network_data[0];
  TestData* test_pos = &test_data[0];
  for (int level = 0; level < num_levels; ++level) {
    // 0. Compute number of neural networks, pixels to use, and limits on
    // maximum pixels any neural network can be assigned.
    int networks_at_level =
        std::min(num_networks / num_levels + 1, networks_left);
    int pixels_at_level =
        std::min(num_pixels / num_levels + 1, static_cast<int>(pixels.size()));
    int pixels_per_network =
        std::ceil(static_cast<float>(ensemble_size * pixels_at_level) /
                  networks_at_level);
    networks_left -= networks_at_level;
    std::fill_n(std::back_inserter(test_data),
                pixels_per_network * networks_at_level, TestData());
    int offset = test_pos - &test_pos[0];
    for (int i = 0; i < networks_at_level; ++i)
      network_pos[i] =
          NetworkData(level, i, offset + pixels_per_network * i, 0);
    // 1. Assign pixels to this level.
    std::vector<int> local_pixels;
    std::copy(pixels.end() - pixels_at_level, pixels.end(),
              std::back_inserter(local_pixels));
    pixels.erase(pixels.end() - pixels_at_level, pixels.end());
    // 2. Assign networks to pixels.
    int current_network = 0;
    for (int i = 0; i < local_pixels.size(); ++i) {
      int pixel_index = local_pixels[i], x = pixel_index % width,
          y = pixel_index / width;
      assignments[assignment_data_size * pixel_index] = level;
      for (int j = 0; j < ensemble_size; ++j) {
        NetworkData* network = &network_pos[current_network];
        test_pos[current_network * pixels_per_network + network->count] =
            TestData(x, y, *image_number, &average[pixel_index], width, height,
                     *num_images);
        ++network->count;
        assignments[assignment_data_size * pixel_index + j + 1] =
            current_network;
        current_network = (current_network + 1) % networks_at_level;
      }
    }
    test_pos += networks_at_level * pixels_per_network;
    network_pos += networks_at_level;
  }
}

void TestAssignmentDataToTestData(int width, int height, int channels,
                                  int num_levels, int num_networks,
                                  int ensemble_size) {
  const int coord_size = 3;
  const int light_size = 1;
  // Generate random test data.
  std::vector<float> average_image;
  std::vector<TestData> test_data;
  std::vector<NetworkData> network_data;
  std::vector<int> assignment_data;
  int image_number = -1;
  int num_images = -1;
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

  // Call assignment_data_to_test_data
  assignment_data_to_test_data(
      &assignment_data[0], height, width, channels, image_number, num_images,
      &average_image[0], height, width, channels, &test_data_out,
      &test_data_dim1, &test_data_dim2, &network_data_out, &network_data_dim1,
      &network_data_dim2);

  // Test network data results.
  assert(network_data_dim1 == network_data.size());
  assert(network_data_dim2 == 4);

  std::vector<NetworkData> network_data_check(network_data_dim1);
  std::vector<NetworkData> network_data_test(network_data_dim1);

  // Copy in data.
  CopyNetworkData(network_data_out, network_data_test);
  CopyNetworkData(reinterpret_cast<int*>(&network_data[0]), network_data_check);

  // Sort so a comparison can be made.
  std::sort(network_data_check.begin(), network_data_check.end(),
            CompareNetworkData());
  std::sort(network_data_test.begin(), network_data_test.end(),
            CompareNetworkData());

  assert(network_data_check.size() == network_data_test.size());
  for (int i = 0; i < network_data_test.size(); ++i) {
    bool equals_level_id_count =
        network_data_check[i].EqualsLevelIdCount(network_data_check[i]);
    if (!equals_level_id_count) {
      std::cout << "Test value " << i
                << " does not match check value for level, id, and count"
                << network_data_test[i] << " " << network_data_check[i] << "\n";
      assert(equals_level_id_count);
    }
  }

  // Test "test data" results.
  assert(test_data_dim1 == width * height * 4);
  assert(test_data_dim2 == channels + 3);

  for (int i = 0; i < network_data_check.size(); ++i) {
    CHECK_BOUNDS(network_data_check, i, test_data.size() / test_data_dim2);
    CHECK_BOUNDS(network_data_test, i, test_data_dim1);
    const NetworkData& data_check = network_data_check[i];
    const NetworkData& data_test = network_data_test[i];
    std::vector<TestData> test(data_check.count);
    std::vector<TestData> check(data_check.count);
    for (int i = 0; i < network_data_check.size(); ++i) {
      CopyTestData(&test_data_out[data_test.start], test);
      std::copy(test_data.begin() + data_check.start,
                test_data.begin() + data_check.start + data_check.count,
                check.begin());
      std::sort(test.begin(), test.end(), CompareTestData());
      std::sort(check.begin(), check.end(), CompareTestData());
      for (int j = 0; j < test.size(); ++j) {
        if (test[j] != check[j]) {
          std::cout << "Match failed at j = " << j << "\n";
          std::cout << "Test = " << test[j] << "\n";
          std::cout << "Check = " << check[j] << "\n";
        }
      }
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

struct Resolution {
  int width;
  int height;
};

enum CommandLineOption {
  kRunAll = 0,
  kTestPredictionsToImage,
  kTestAssignmentDataToTestData
};

int main(int argc, char** argv) {
  int x = 180;
  int y = 240;
  int c = 3;
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

  CommandLineOption option = kRunAll;
  if (argc > 1) {
    char c = argv[1][0];
    if (c == 'a') c = '0';
    option = static_cast<CommandLineOption>(c - '0');
  }

  switch (option) {
    case kTestPredictionsToImage:
      for (auto r : resolutions) {
        int step = r.width * r.height / 10;
        for (int i = 1; i < r.width * r.height; i += step) {
          std::cout << "-------- TestPredictionsToImage ------ (" << r.width
                    << "," << r.height << ") - " << i << " -----\n";
          if (!TestPredictionsToImage(r.width, r.height, c, i)) {
            exit(0);
          }
        }
      }
      break;
    case kTestAssignmentDataToTestData:
      for (auto r : resolutions) {
        int num_levels = 5;
        int num_networks = 900;
        int ensemble_size = 5;
        std::cout << "-------- TestAssignmentDataToTestData ------ (" << r.width
                  << "," << r.height << ") - "
                  << " num_levels = " << num_levels
                  << " num_networks = " << num_networks << " -----\n";
        TestAssignmentDataToTestData(r.width, r.height, c, num_levels,
                                     num_networks, ensemble_size);
      }
      break;
    default:
      std::cout << "Invalid option " << argv[1] << "\n";
  }

  return 0;
}
