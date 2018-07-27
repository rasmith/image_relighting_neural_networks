#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kmeans2d.h"
#include "logger.h"
#include "types.h"

// Ensure x[i].start + x[i].count <= n
#define CHECK_BOUNDS(x, i, n)                                             \
  {                                                                       \
    auto& d = (x)[(i)];                                                   \
    int extent = d.start + d.count;                                       \
    bool bounds_check = extent <= (n);                                    \
    if (!bounds_check) {                                                  \
      LOG(ERROR) << __LINE__ << ": Bounds check failed at " << (i)        \
                 << ". Limit is " << (n) << " but got " << extent << "."; \
      assert(bounds_check);                                               \
    }                                                                     \
  }

bool ContainsDuplicates(const NetworkData* network_data, int count) {
  std::unordered_set<NetworkData, HashNetworkData, CompareNetworkData>
      network_set;
  for (int i = 0; i < count; ++i) {
    if (network_set.find(network_data[i]) != network_set.end()) return true;
    network_set.insert(network_data[i]);
  }
  return false;
}

template <class Iterator>
void PrintValues(Iterator begin, Iterator end, std::ostream& out) {
  int j = 0;
  for (Iterator i = begin; i != end; ++i, ++j) out << j << " " << *i << "\n";
}

void GenerateRandomImage(int width, int height, int channels,
                         std::vector<float>& image) {
  image.resize(width * height * channels);
  for (int i = 0; i < width * height * channels; ++i) {
    image[i] = std::rand() / static_cast<float>(RAND_MAX);
    assert(i >= 0 && i < width * height * channels);
  }
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
  TestData* pos = reinterpret_cast<TestData*>(&test[0]);
  test.resize(num_samples * test_data_size);
  for (int i = 0; i < num_samples * test_data_size; i += test_data_size) {
    int k = q.top().first;
    int x = k % width;
    int y = k / width;
    q.pop();
    test[i] = x / static_cast<float>(width - 1);
    test[i + 1] = y / static_cast<float>(height - 1);
  }
}

std::string PixelToString(float* p) {
  return "[" + std::to_string(p[0]) + "," + std::to_string(p[1]) + "," +
         std::to_string(p[2]) + "]";
}

bool EqualsPixel(float* p, float* q) {
  return p[0] == q[0] && p[1] == q[1] && p[2] == q[2];
}

void GenerateRandomPermutation(int n, std::vector<int>& values) {
  LOG(DEBUG) << "GenerateRandomPermutation\n";
  values.resize(n);
  for (int i = 0; i < n; ++i) values[i] = i;
  for (int i = 0; i < n - 1; ++i) {
    int j = std::rand() % (i + 1);
    assert(j >= 0 && j < n);
    std::swap(values[i], values[j]);
  }
  for (int i = 0; i < n; ++i)
    assert(values[i] >= 0 && values[i] < values.size());
}

void GenerateTestAndAssignmentData(
    int width, int height, int channels, int ensemble_size, int num_levels,
    int num_networks, int* num_images, int* image_number,
    std::vector<float>& average, std::vector<TestData>& test_data,
    std::vector<int>& assignments, std::vector<NetworkData>& network_data) {
  LOG(DEBUG) << "GenerateTestAndAssignmentData\n";
  const int assignment_data_size = ensemble_size + 1;
  int num_pixels = width * height;  // number of pixels
  LOG(DEBUG) << "GenerateTestAndAssignmentData: assignment_data_size = "
             << assignment_data_size << " num_pixels = " << num_pixels
             << " width = " << width << " height = " << height << "\n";

  // Get a random order of pixels.
  std::vector<int> pixels;
  GenerateRandomPermutation(num_pixels, pixels);

  // Allocate test_data, network_data, and assignments.
  LOG(DEBUG) << "GenerateTestAndAssignmentData: allocate\n";
  test_data.clear();
  network_data.resize(num_networks);
  std::fill(network_data.begin(), network_data.end(), NetworkData());
  assignments.resize(num_pixels * assignment_data_size);

  // Generate random average image.
  GenerateRandomImage(width, height, channels, average);
  LOG(DEBUG) << "GenerateTestAndAssignmentData: average.size = "
             << average.size() << "\n";

  // Set the num_images and image_number randomly.
  LOG(DEBUG) << "GenerateTestAndAssignmentData: set random num_images and "
                "image_number\n";
  *num_images = std::rand() % 1000 + 100;     // set number of images randomly
  *image_number = std::rand() % *num_images;  // set image number randomly

  LOG(DEBUG) << "GenerateTestAndAssignmentData: initialize for main loop\n";
  int networks_left = num_networks;
  NetworkData* network_pos = &network_data[0];
  TestData* test_pos = &test_data[0];
  int total = 0;
  for (int level = 0; level < num_levels; ++level) {
    // LOG(DEBUG) << " ---------------------------------------------- \n";
    // LOG(DEBUG) << "GenerateTestAndAssignmentData: level = " << level << "\n";
    // 0. Compute number of neural networks, pixels to use, and limits on
    // maximum pixels any neural network can be assigned.
    int networks_at_level =
        std::min(num_networks / num_levels + 1, networks_left);
    int pixels_at_level =
        std::min(num_pixels / num_levels + 1, static_cast<int>(pixels.size()));
    int pixels_per_network =
        std::ceil(static_cast<float>(ensemble_size * pixels_at_level) /
                  networks_at_level);
    int current_network = 0;
    // LOG(DEBUG)
    //<< "GenerateTestAndAssignmentData: num_pixels / num_levels + 1 = "
    //<< num_pixels / num_levels + 1 << " pixels.size() = " << pixels.size()
    //<< "\n";
    LOG(STATUS) << "GenerateTestAndAssignmentData: networks_at_level= "
                << networks_at_level << " pixels_at_level = " << pixels_at_level
                << " pixels_per_network = " << pixels_per_network
        //<< " max_count = " << *max_count
                << " pixels.size=" << pixels.size() << "\n";
    // LOG(DEBUG) << " GenerateTestAndAssignmentData: networks_left = "
    //<< networks_left << "\n";
    networks_left -= networks_at_level;
    assert(networks_left >= 0);
    // LOG(DEBUG) << " GenerateTestAndAssignmentData: fill\n";
    assert(test_pos >= &test_data[0]);
    assert(test_pos - &test_data[0] == test_data.size());
    int offset = test_pos - &test_data[0];
    std::fill_n(std::back_inserter(test_data),
                networks_at_level * pixels_per_network, TestData());
    test_pos = &test_data[0] + offset;
    assert(test_pos >= &test_data[0]);
    assert(test_pos - &test_data[0] < test_data.size());
    // LOG(DEBUG) << " GenerateTestAndAssignmentData: populate network data\n";
    for (int i = 0; i < networks_at_level; ++i) {
      assert(network_pos >= &network_data[0]);
      assert((network_pos - &network_data[0]) + i < network_data.size());
      network_pos[i] =
          NetworkData(level, i, offset + pixels_per_network * i, 0);
      assert(network_pos[i].start + network_pos[i].count < test_data.size());
      assert(network_pos[i].start >= 0 && network_pos[i].count >= 0);
    }
    // 1. Assign pixels to this level.
    std::vector<int> local_pixels;
    assert(pixels_at_level >= 0);
    assert(pixels_at_level <= pixels.size());
    for (int i = 0; i < pixels_at_level; ++i) {
      local_pixels.push_back(pixels.back());
      pixels.pop_back();
    }
    assert(local_pixels.size() == pixels_at_level);
    // 2. Assign networks to pixels.
    current_network = 0;
    LOG(DEBUG) << " GenerateTestAndAssignmentData: assign networks to pixels\n";
    for (int i = 0; i < local_pixels.size(); ++i) {
      int pixel_index = local_pixels[i], x = pixel_index % width,
          y = pixel_index / width;
      assert(x >= 0 && y >= 0);
      assert(x < width && y < height);
      assert(assignment_data_size * pixel_index >= 0);
      assert(assignment_data_size * pixel_index <
             assignment_data_size * width * height);
      assignments[assignment_data_size * pixel_index] = level;
      // LOG(DEBUG) << "GenerateTestAndAssignmentData: assign networks to (" <<
      // x
      //<< "," << y << ")\n";
      float* rgb = &average[pixel_index * channels];
      TestData data(x, y, *image_number, rgb, width, height, *num_images);
      // if (pixel_index <= 10)
      // std::cout << "GenerateTestAndAssignmentData: level = " << level
      //<< " pixel_index = " << pixel_index << " " << data << "\n";
      for (int j = 0; j < ensemble_size; ++j) {
        // LOG(DEBUG) << "GenerateTestAndAssignmentData: assign network "
        //<< current_network << " to (" << x << "," << y << ")\n";
        assert(network_pos >= &network_data[0] &&
               ((network_pos - &network_data[0]) + current_network <
                network_data.size()));
        assert(current_network >= 0 && current_network < networks_at_level);
        NetworkData* network = &network_pos[current_network];
        int index = current_network * pixels_per_network + network->count;
        // LOG(DEBUG) << "GenerateTestAndAssignmentData: add to test data\n";
        // LOG(DEBUG) << "GenerateTestAndAssignmentData: current_network = "
        //<< current_network
        //<< " pixels_per_network = " << pixels_per_network
        //<< " network->count = " << network->count
        //<< " index = " << index << " pixel_index = " << pixel_index
        //<< "\n";
        assert(pixel_index >= 0 && pixel_index < num_pixels);
        assert(pixel_index * channels <= width * height * channels);
        assert(pixel_index >= 0);
        // LOG(DEBUG) << "GenerateTestAndAssignmentData: get rgb values: r = "
        //<< rgb[0] << " g = " << rgb[1] << " b = " << rgb[2] << "\n";
        //// LOG(DEBUG) << "GenerateTestAndAssignmentData: assign to
        /// test_pos\n";
        // assert(index >= 0);
        // assert(test_pos >= &test_data[0] &&
        //((test_pos - &test_data[0]) + index < test_data.size()));
        test_pos[index] = data;
        //++network->count;
        ++network->count;
        ++total;
        if (!(network->count >= 0 && network->count <= pixels_per_network)) {
          LOG(ERROR) << "network->count is bad, " << network->count
                     << " and limit is " << pixels_per_network << "\n";
        }
        assert(network->count >= 0 && network->count <= pixels_per_network);
        // LOG(DEBUG) << "GenerateTestAndAssignmentData: add assignment\n";
        assert(assignment_data_size * pixel_index + j + 1 <=
               width * height * assignment_data_size);
        assignments[assignment_data_size * pixel_index + j + 1] =
            current_network;
        // LOG(DEBUG) << "GenerateTestAndAssignmentData: next
        // current_network\n";
        current_network = (current_network + 1) % networks_at_level;
        assert(current_network >= 0 && current_network < networks_at_level);
      }
    }
    test_pos += networks_at_level * pixels_per_network;
    network_pos += networks_at_level;
  }
  assert(network_data.size() == num_networks);
  assert(pixels.size() == 0);
  // return;
  // bool contains_duplicates =
  // ContainsDuplicates(&network_data[0], network_data.size());
  // if (contains_duplicates) {
  // LOG(ERROR)
  //<< "GenerateTestAndAssignmentData: network_data contains duplicates.\n";
  // assert(!contains_duplicates);
  //}
  if (total != num_pixels * ensemble_size) {
    LOG(ERROR) << "Expected " << num_pixels * ensemble_size
               << " test entries, but got  " << total << " test entries.\n";
    assert(total == num_pixels * ensemble_size);
  }
}

// void predictions_to_errors(std::vector<int>& order, int ensemble_size,
// float* test, int test_dim1, int test_dim2,
// float* target, int target_dim1, int target_dim2,
// float* predictions, int predictions_dim1,
// int predictions_dim2, float* errors, int errors_dim1,
// int errors_dim2) {
//

void TestPredictionsToErrors(int width, int height, int channels,
                             int ensemble_size) {

  int num_pixels = width * height;
  std::vector<float> errors;
  GenerateRandomImage(width, height, channels, errors);
  std::vector<float> target;
  GenerateRandomImage(width, height, channels, target);
  std::vector<float> predictions(num_pixels);
  for (int i = 0; i < num_pixels; ++i) predictions[i] += errors[i];
  predictions_to_errors(
      order, ensemble_size, &test[0], num_pixels,
      sizeof(TestData) / sizeof(float), &target[0], num_pixels,
      sizeof(TestData) / sizeof(float), &predictions[0], num_pixels,
      sizeof(TestData) / sizeof(float), errors, height, width);
}

void TestAssignmentDataToTestData(int width, int height, int channels,
                                  int num_levels, int num_networks,
                                  int ensemble_size) {
  // Generate random test data.
  std::vector<float> average_image;
  std::vector<TestData> test_data;
  std::vector<NetworkData> network_data;
  std::vector<int> assignment_data;
  int num_pixels = width * height;
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
  LOG(STATUS) << "TestAssignmentDataToTestData: assignment_data_to_test_data\n";
  assignment_data_to_test_data(
      &assignment_data[0], height, width, ensemble_size + 1, image_number,
      num_images, &average_image[0], height, width, channels, &test_data_out,
      &test_data_dim1, &test_data_dim2, &network_data_out, &network_data_dim1,
      &network_data_dim2);

  // const TestData* test_data_pos =
  // reinterpret_cast<const TestData*>(&test_data[0]);
  // const TestData* location = std::find_if(
  // test_data_pos, test_data_pos + test_data_dim1,
  //[](const TestData & t)->bool { return t.equalsXy(0.0f, 0.0f); });
  // if (location == test_data_pos + test_data_dim1) {
  // LOG(ERROR) << "Did not find (0, 0, 0) anywwhere!";
  //} else {
  // LOG(ERROR) << "Found (0, 0, 0) at " << location << "\n";
  //}

  LOG(STATUS) << "TestAssignmentDataToTestData: contains_duplicates\n";
  bool contains_duplicates = ContainsDuplicates(
      reinterpret_cast<NetworkData*>(network_data_out), network_data_dim1);
  if (contains_duplicates) {
    LOG(ERROR)
        << "GenerateTestAndAssignmentData: network_data contains duplicates.\n";
    assert(!contains_duplicates);
  }

  // Test network data results.
  if (network_data_dim1 != network_data.size()) {
    LOG(ERROR) << "TestAssignmentDataToTestData: network_data_dim1 = "
               << network_data_dim1
               << " and network_data.size() = " << network_data.size()
               << " do not match.\n";
  }
  assert(network_data_dim1 == network_data.size());
  assert(network_data_dim2 == sizeof(NetworkData) / sizeof(int));

  std::vector<NetworkData> network_data_check(network_data_dim1);
  std::vector<NetworkData> network_data_test(network_data_dim1);

  // Copy in data.
  const NetworkData* network_data_out_pos =
      reinterpret_cast<const NetworkData*>(network_data_out);
  std::copy(network_data_out_pos, network_data_out_pos + network_data_dim1,
            network_data_test.begin());
  std::copy(network_data.begin(), network_data.end(),
            network_data_check.begin());

  assert(network_data_check.size() == network_data_test.size());
  // Sort so a comparison can be made.
  CompareNetworkData cmp;
  std::sort(network_data_check.begin(), network_data_check.end(), cmp);
  std::sort(network_data_test.begin(), network_data_test.end(), cmp);

  LOG(STATUS) << "TestAssignmentDataToTestData: check network data\n";
  for (int i = 0; i < network_data_test.size(); ++i) {
    bool equals_level_id_count =
        network_data_check[i].EqualsLevelIdCount(network_data_test[i]);
    if (!equals_level_id_count) {
      LOG(ERROR) << "Test value " << i
                 << " does not match check value for level, id, and count"
                 << network_data_test[i] << " " << network_data_check[i]
                 << "\n";
      PrintValues(network_data_test.begin(), network_data_test.begin() + 10,
                  LOG(ERROR));
      PrintValues(network_data_check.begin(), network_data_check.begin() + 10,
                  LOG(ERROR));
      assert(equals_level_id_count);
    }
  }
  LOG(STATUS) << "TestAssignmentDataToTestData: check assignments\n";
  const AssignmentData* assignments_check =
      reinterpret_cast<const AssignmentData*>(&assignment_data[0]);
  std::vector<AssignmentData> assignments_test(width * height,
                                               AssignmentData());
  std::vector<int> assignment_counts(width * height, -1);
  for (int j = 0; j < network_data_test.size(); ++j) {
    const NetworkData& n = network_data_test[j];
    for (int i = n.start; i < n.start + n.count; ++i) {
      const TestData* t = reinterpret_cast<const TestData*>(test_data_out) + i;
      int x = std::round(t->x * (width - 1)),
          y = std::round(t->y * (height - 1));
      if (!(y >= 0 && y < height)) {
        LOG(ERROR) << " t->x = " << t->x << " t->y = " << t->y
                   << " t->x * width = " << t->x * (width - 1)
                   << " t->y * height = " << t->y * (height - 1) << "\n";
        LOG(ERROR) << " x = " << x << " y = " << y << " width = " << width
                   << " height = " << height << "\n";
      }
      assert(x >= 0 && x < width);
      assert(y >= 0 && y < height);
      int index = y * width + x;
      if (!(n.level == assignments_check[index].level)) {
        LOG(ERROR) << "n.level = " << n.level << " assignments_check[" << index
                   << "].level = " << assignments_check[index].level
                   << " j = " << j << " x = " << x << " y = " << y << "\n";
      }
      assert(n.level == assignments_check[index].level);
      AssignmentData& a = assignments_test[index];
      if (assignment_counts[index] == -1) {
        a.level = n.level;
        ++assignment_counts[index];
      }
      if (!(a.level == n.level)) {
        LOG(ERROR) << "a.level = " << a.level << " n.level = " << n.level
                   << " assignments_check = " << assignments_check[index].level
                   << "\n";
      }
      assert(a.level == n.level);
      a[++assignment_counts[index] - 1] = n.id;
      if (!(assignment_counts[index] >= 1 &&
            assignment_counts[index] <= ensemble_size)) {
        LOG(ERROR) << "assignment_counts[" << index
                   << "]=" << assignment_counts[index] << ".\n";
      }
      assert(assignment_counts[index] >= 1 &&
             assignment_counts[index] <= ensemble_size);
    }
  }

  for (int i = 0; i < num_pixels; ++i) {
    if (assignment_counts[i] != ensemble_size) {
      LOG(ERROR) << "assignment_counts[" << i << "]=" << assignment_counts[i]
                 << "\n";
    }
    assert(assignment_counts[i] == ensemble_size);
  }

  const AssignmentData* a_pos = assignments_check;
  const AssignmentData* b_pos = &assignments_test[0];
  for (int i = 0; i < num_pixels; ++i) {
    if (*a_pos != *b_pos) {
      LOG(ERROR) << "a_pos = " << *a_pos << " b_pos = " << *b_pos << "\n";
    }
    assert(*a_pos == *b_pos);
    ++a_pos;
    ++b_pos;
  }

  LOG(STATUS) << "TestAssignmentDataToTestData: check test data\n";
  // Test "test data" results.
  assert(test_data_dim1 == width * height * ensemble_size);
  assert(test_data_dim2 == sizeof(TestData) / sizeof(int));

  for (int i = 0; i < network_data_check.size(); ++i) {
    CHECK_BOUNDS(network_data_check, i, test_data.size());
    CHECK_BOUNDS(network_data_test, i, test_data_dim1);
    const NetworkData& data_check = network_data_check[i];
    const NetworkData& data_test = network_data_test[i];
    assert(data_test.EqualsLevelIdCount(data_check));
    std::vector<TestData> test(data_check.count);
    std::vector<TestData> check(data_check.count);
    const TestData* test_data_out_pos =
        reinterpret_cast<const TestData*>(test_data_out) + data_test.start;
    std::copy(test_data_out_pos, test_data_out_pos + data_test.count,
              test.begin());
    std::copy(test_data.begin() + data_check.start,
              test_data.begin() + data_check.start + data_check.count,
              check.begin());
    std::sort(test.begin(), test.end(), CompareTestData());
    std::sort(check.begin(), check.end(), CompareTestData());
    for (int j = 0; j < test.size(); ++j) {
      if (test[j] != check[j]) {
        PrintValues(test.begin(), test.begin() + 10, LOG(ERROR));
        PrintValues(check.begin(), check.begin() + 10, LOG(ERROR));
        LOG(ERROR) << "Match failed at i = " << i << " j = " << j << "\n";
        LOG(ERROR) << "Test = " << test[j] << "\n";
        LOG(ERROR) << "Check = " << check[j] << "\n";
        assert(test[j] == check[j]);
      }
    }
  }
  LOG(ERROR) << "TestAssignmentDataToTestData: finished\n";
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
      LOG(DEBUG) << "TestPredictionsToImage: (" << x << "," << y
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

  CommandLineOption option;
  if (argc > 1) {
    char c = argv[1][0];
    LOG(DEBUG) << "c=" << c << "\n";
    option = static_cast<CommandLineOption>(c - '0');
  }

  switch (option) {
    case kTestPredictionsToImage:
      for (auto r : resolutions) {
        int step = r.width * r.height / 10;
        for (int i = 1; i < r.width * r.height; i += step) {
          LOG(STATUS) << "TestPredictionsToImage: (" << r.width << ","
                      << r.height << ") - " << i << "\n";
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
        LOG(STATUS) << "TestAssignmentDataToTestData: (" << r.width << ","
                    << r.height << ") - "
                    << " num_levels = " << num_levels
                    << " num_networks = " << num_networks << "\n";
        TestAssignmentDataToTestData(r.width, r.height, c, num_levels,
                                     num_networks, ensemble_size);
      }
      break;
    default:
      LOG(STATUS) << "Invalid option " << argv[1] << "\n";
  }

  return 0;
}
