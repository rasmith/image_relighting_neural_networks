#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "quickselect.h"

namespace kdtree {

template <typename GlmType, int DimensionNumber>
struct GlmComparator {
  GlmComparator() {}
  bool operator()(const GlmType& a, const GlmType& b) {
    return a[DimensionNumber] < b[DimensionNumber];
  }
};

struct KdNode {
  enum Type { kInternal, kLeaf };

  KdNode()
      : type(kInternal), split_dimension(0), split_value(0.0f), index(-1) {}

  KdNode(const KdNode& node)
      : type(node.type),
	split_dimension(node.split_dimension),
	split_value(node.split_value),
	index(node.index) {}
  Type type;
  int split_dimension;
  float split_value;
  int index;
};

class KdTree {
 public:
  typedef GlmComparator<glm::vec2, 0> XComparator;
  typedef GlmComparator<glm::vec2, 1> YComparator;

  KdTree() {}

  void AddPoints(const std::vector<glm::vec2>& input_points);
  void Build() { RecursiveBuild(0, points_.size(), 0); }
  int NearestNeighbor(const glm::vec2 query);

 protected:
  void ComputeBounds();
  glm::vec2 SelectMedian(int first, int last, int dim);
  void RecursiveBuild(int first, int last, int dim);
  int RecursiveNearestNeighbor(const glm::vec2 query, const KdNode& node) {
    return 0;
  }

  glm::vec2 min_;
  glm::vec2 max_;
  std::vector<glm::vec2> points_;
  std::vector<KdNode> nodes_;
};

}  // namespace kdtree
