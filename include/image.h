#pragma once

#include <cstdlib>
#include <string>
#include <vector>

namespace image {

struct Pixel {
  Pixel() {}
  Pixel(const Pixel& p) : r(p.r), g(p.g), b(p.b) {}
  Pixel(uint8_t rr, uint8_t gg, uint8_t bb) : r(rr), g(gg), b(bb) {}
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

class Image {
 public:
  Image() {}
  Image(uint32_t width, uint32_t height) : width_(width), height_(height) {
    pixels_.resize(width_ * height_);
  }
  Image(const Image& im);
  inline Pixel& operator()(uint32_t x, uint32_t y) {
    return pixels_[y * width_ + x];
  }
  inline const Pixel& operator()(uint32_t x, uint32_t y) const {
    return pixels_[y * width_ + x];
  }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  void ToPpm(std::string& output) const;

 private:
  std::vector<Pixel> pixels_;
  uint32_t width_;
  uint32_t height_;
};

}  // namespace image
