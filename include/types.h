#pragma once

#include <tuple>

struct AssignmentData {
  int level;
  int a0;
  int a1;
  int a2;
  int a3;
  int a4;
  inline int& operator[](int i) { return (&a0)[i]; }
  inline int operator[](int i) const { return (&a0)[i]; }
};
static_assert(sizeof(AssignmentData) == 6 * sizeof(int),
              "AssignmentData size should be  6 * sizeof(int) but was not.");

// TestData data(x, y, *image_number, rgb, width, height, *num_images);
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
  bool equalsXy(float xx, float yy) const {
    return xx == x && yy == y;
  }
};
static_assert(sizeof(TestData) == 6 * sizeof(float),
              "TestData size should be  6 * sizeof(float) but was not.");

struct CompareTestData {
  bool operator()(const TestData& a, const TestData& b) const {
    return std::tie(a.x, a.y, a.i, a.r, a.g, a.b) <
           std::tie(b.x, b.y, b.i, b.r, b.g, b.b);
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
  NetworkData(int l, int i) : level(l), id(i), start(0), count(0) {}
  bool EqualsLevelIdCount(const NetworkData& d) const {
    return (this == &d) || (level == d.level && id == d.id && count == d.count);
  }
  bool operator==(const NetworkData& a) const {
    return (this == &a) || (level == a.level && id == a.id &&
                            start == a.start && count == a.count);
  }
  bool operator!=(const NetworkData& a) { return !((*this) == a); }
};
static_assert(sizeof(NetworkData) == 4 * sizeof(int),
              "Expected sizeof(NetworkData) = 4 * sizeof(int))");

struct HashNetworkData {
  size_t operator()(const NetworkData& data) const {
    std::hash<std::string> hash;
    return hash(std::to_string(data.level) + "-" + std::to_string(data.id));
  }
};

struct CompareNetworkData {
  bool operator()(const NetworkData& a, const NetworkData& b) const {
    return std::tie(a.level, a.id, a.count, a.start) <
           std::tie(b.level, b.id, b.count, b.start);
  }
};

inline std::ostream& operator<<(std::ostream& out, const NetworkData& d) {
  out << "{level:" << d.level << ", id:" << d.id << ", start:" << d.start
      << ", count:" << d.count << "}\n";
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const TestData& d) {
  out << "{x:" << d.x << ", y:" << d.y << ", i:" << d.i << ", r:" << d.r
      << ", g:" << d.g << ", b:" << d.b << "}\n";
  return out;
}
