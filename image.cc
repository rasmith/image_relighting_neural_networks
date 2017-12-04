#include "image.h"

#include <string>

namespace image {

Image::Image(const Image& im) : width_(im.width_), height_(im.height_) {
  pixels_.resize(im.pixels_.size());
  std::copy(im.pixels_.begin(), im.pixels_.end(), pixels_.begin());
}

void Image::SetDimensions(uint32_t width, uint32_t height) {
  width_ = width;
  height_ = height;
  pixels_.resize(width * height);
}

void Image::ToPpm(std::string& result) const {
  std::string output = "P3\n";
  output += std::to_string(width_) + " " + std::to_string(height_) + "\n";
  output += "255\n";
  const Pixel* current = (const Pixel*)(&pixels_[0]);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      output += std::to_string(static_cast<int>(current->r)) + " ";
      output += std::to_string(static_cast<int>(current->g)) + " ";
      output += std::to_string(static_cast<int>(current->b));
      ++current;
      if (x < width_ - 1) output += " ";
    }
    output += "\n";
  }
  result = output;
}

}  // namespace image
