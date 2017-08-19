#include "kdtree.h"

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

namespace kdtree {

void KdTree::AddPoints(const std::vector<glm::vec2>& input_points) {
  points_.resize(input_points.size());
  std::copy(input_points.begin(), input_points.end(), points_.begin());
}

void KdTree::ComputeBounds() {
  min_ = glm::vec2(std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max());
  max_ = glm::vec2(-std::numeric_limits<float>::max(),
                   -std::numeric_limits<float>::max());
  for (auto& point : points_) {
    min_.x = std::min(point.x, min_.x);
    min_.y = std::min(point.y, min_.y);
    max_.x = std::max(point.x, max_.x);
    max_.y = std::max(point.y, max_.y);
  }
}

glm::vec2 KdTree::SelectMedian(int first, int last, int dim) {
  int index = (first + last) / 2;
  int size = last - first;
  return (dim == 0 ? quickselect::QuickSelect<glm::vec2, XComparator>(
                         &points_[first], size, index)
                   : quickselect::QuickSelect<glm::vec2, YComparator>(
                         &points_[first], size, index));
}

void KdTree::RecursiveBuild(int first, int last, int dim) {
  KdNode node;
  if (last - first < 1) return;
  if (last - first == 1) {
    node.type = KdNode::kLeaf;
    node.index = first;
  } else {
    int median_index = (first + last - 1) / 2;
    SelectMedian(first, last, dim);
    RecursiveBuild(first, median_index, (dim + 1) % 2);
    RecursiveBuild(median_index, last, (dim + 1) % 2);
  }
}

}  // namespace kdtree
