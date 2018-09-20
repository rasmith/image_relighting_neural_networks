#pragma once

#include "image.h"

#include <iostream>
#include <algorithm>
#include <tuple>

#include <Eigen/Core>
#include <OpenANN/OpenANN>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <glm/glm.hpp>

class ImageDataSet : public OpenANN::DataSet {
 public:
  ImageDataSet(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& outputs)
      : in_(inputs), out_(outputs), data_set_(&in_, &out_) {}
  virtual int samples() { return data_set_.samples(); }
  virtual int inputs() { return data_set_.inputs(); }
  virtual int outputs() { return data_set_.outputs(); }
  virtual Eigen::VectorXd& getInstance(int i) {
    return data_set_.getInstance(i);
  }
  virtual Eigen::VectorXd& getTarget(int i) { return data_set_.getTarget(i); }
  virtual void finishIteration(OpenANN::Learner& learner) {}

 private:
  Eigen::MatrixXd in_, out_;
  OpenANN::DirectStorageDataSet data_set_;
};

struct CoordinateData {
  double x;
  double y;
  CoordinateData(int xx, int yy, int width, int height)
      : x(xx / static_cast<double>(width)),
        y(yy / static_cast<double>(height)) {}
  CoordinateData(double xx, double yy) : x(xx), y(yy) {}
  double& operator[](int i) { return (&x)[i]; }
  double operator[](int i) const { return (&x)[i]; }
};

struct PixelConversion {
  enum ConversionType {
    kZeroToPositiveOne,
    kNegativeOneToPositiveOne
  };
  static double Convert(uint8_t x, PixelConversion::ConversionType conversion) {
    return (conversion == kZeroToPositiveOne
                ? glm::clamp(x / 255.0, 0.0, 1.0)
                : glm::clamp(2.0 * (x / 255.0) - 1.0, -1.0, 1.0));
  }
  static double Convert(double x, PixelConversion::ConversionType conversion) {
    return (conversion == kZeroToPositiveOne ? x : glm::clamp(2.0 * x - 1.0,
                                                              -1.0, 1.0));
  }
  static double Unconvert(double x,
                          PixelConversion::ConversionType conversion) {
    return (conversion == kZeroToPositiveOne
                ? glm::clamp(x, 0.0, 1.0)
                : glm::clamp(0.5 * (x + 1.0), 0.0, 1.0));
  }
  static PixelConversion::ConversionType DefaultConversion() {
    return kZeroToPositiveOne;
  }
};

struct PixelData {
  double r;
  double g;
  double b;
  PixelData() : r(0.0), g(0.0), b(0.0) {}
  PixelData(double rr, double gg, double bb) : r(rr), g(gg), b(bb) {}
  PixelData(double rr, double gg, double bb,
            ::PixelConversion::ConversionType conversion)
      : r(::PixelConversion::Convert(rr, conversion)),
        g(::PixelConversion::Convert(gg, conversion)),
        b(::PixelConversion::Convert(bb, conversion)) {}
  PixelData(const image::Pixel& p, ::PixelConversion::ConversionType conversion)
      : r(::PixelConversion::Convert(p.r, conversion)),
        g(::PixelConversion::Convert(p.g, conversion)),
        b(::PixelConversion::Convert(p.b, conversion)) {}
  double& operator[](int i) { return (&r)[i]; }
  double operator[](int i) const { return (&r)[i]; }
  bool operator==(const PixelData& p) const {
    return r == p.r && g == p.g && b == p.b;
  }
  bool operator!=(const PixelData& p) const { return !(*this == p); }
};
static_assert(sizeof(PixelData) == 3 * sizeof(double),
              "PixelData size should be  3 * sizeof(double) but was not.");

struct AssignmentData {
  int level;
  int a0;
  int a1;
  int a2;
  int a3;
  int a4;
  AssignmentData() : level(0), a0(0), a1(0), a2(0), a3(0), a4(0) {}
  int& operator[](int i) { return (&a0)[i]; }
  int operator[](int i) const { return (&a0)[i]; }
  bool operator==(const AssignmentData& other) const {
    constexpr int count = sizeof(AssignmentData) / sizeof(int) - 1;
    if (this == &other) return true;
    AssignmentData a = *this, b = other;
    std::sort(&a.a0, &a.a0 + count);
    std::sort(&b.a0, &b.a0 + count);
    return a.level == b.level && a.a0 == b.a0 && a.a1 == b.a1 && a.a2 == b.a2 &&
           a.a3 == b.a3 && a.a4 == b.a4;
  }
  bool operator!=(const AssignmentData& other) const {
    return !(*this == other);
  }
};
static_assert(sizeof(AssignmentData) == 6 * sizeof(int),
              "AssignmentData size should be  6 * sizeof(int) but was not.");

struct TestData {
  enum PixelConversion {
    kZeroToPositiveOne,
    kNegativeOneToPositiveOne
  };
  double x;
  double y;
  double i;
  double r;
  double g;
  double b;
  TestData() : x(0.0), y(0.0), i(0.0), r(0.0), g(0.0), b(0.0) {}
  TestData(int ix, int iy, int in, const double* rgb, int width, int height,
           int num_images)
      : x(static_cast<double>(ix) / (std::max(width - 1, 1))),
        y(static_cast<double>(iy) / (std::max(height - 1, 1))),
        i(static_cast<double>(in) / (std::max(num_images - 1, 1))),
        r(rgb[0]),
        g(rgb[1]),
        b(rgb[2]) {}
  TestData(int ix, int iy, int in, const double* rgb, int width, int height,
           int num_images, ::PixelConversion::ConversionType conversion)
      : x(static_cast<double>(ix) / (std::max(width - 1, 1))),
        y(static_cast<double>(iy) / (std::max(height - 1, 1))),
        i(static_cast<double>(in) / (std::max(num_images - 1, 1))),
        r(::PixelConversion::Convert(rgb[0], conversion)),
        g(::PixelConversion::Convert(rgb[1], conversion)),
        b(::PixelConversion::Convert(rgb[2], conversion)) {}
  TestData(int ix, int iy, int in, const image::Pixel& p, int width, int height,
           int num_images, ::PixelConversion::ConversionType conversion)
      : x(static_cast<double>(ix) / (std::max(width - 1, 1))),
        y(static_cast<double>(iy) / (std::max(height - 1, 1))),
        i(static_cast<double>(in) / (std::max(num_images - 1, 1))),
        r(::PixelConversion::Convert(p.r, conversion)),
        g(::PixelConversion::Convert(p.g, conversion)),
        b(::PixelConversion::Convert(p.b, conversion)) {}
  bool operator==(const TestData& a) {
    return (this == &a) || (x == a.x && y == a.y && i == a.i && r == a.r &&
                            g == a.g && b == a.b);
  }
  bool operator!=(const TestData& a) { return !((*this) == a); }
  bool equalsXy(double xx, double yy) const { return xx == x && yy == y; }
};
static_assert(sizeof(TestData) == 6 * sizeof(double),
              "TestData size should be  6 * sizeof(double) but was not.");

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

inline std::ostream& operator<<(std::ostream& out, const AssignmentData& a) {
  out << "{level:" << a.level << ", a0:" << a.a0 << ", a1:" << a.a1
      << ", a2:" << a.a2 << ", a3:" << a.a3 << ", a4:" << a.a4 << "}\n";
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const PixelData& p) {
  out << "{r:" << p.r << ", g:" << p.g << ", b:" << p.b << "}\n";
  return out;
}
