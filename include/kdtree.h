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
  enum Type { kInternal = 0, kLeaf = 1 };

  KdNode()
      : type(kInternal),
        split_dimension(0),
        split_value(0.0f),
        left(-1),
        right(-1) {}

  KdNode(const KdNode& node)
      : type(node.type),
        split_dimension(node.split_dimension),
        split_value(node.split_value),
        left(node.left),
        right(node.right) {}
  unsigned char type : 1;
  unsigned char split_dimension : 3;
  int left;
  int right;
  float split_value;
};

std::ostream& operator<<(std::ostream& out, const KdNode& node);

class KdTree {
 public:
  typedef GlmComparator<glm::vec2, 0> XComparator;
  typedef GlmComparator<glm::vec2, 1> YComparator;

  KdTree() {}

  void AssignPoints(const std::vector<glm::vec2>& input_points);
  const std::vector<glm::vec2>& points() const { return points_; }
  const std::vector<KdNode>& nodes() const { return nodes_; }
  void SetMaxDepth(int depth) { max_depth_ = depth; }
  void SetMaxLeafSize(int size) { max_leaf_size_ = size; }

  void Build() {
    nodes_.push_back(KdNode());
    RecursiveBuild(0, 0, points_.size(), 0, 0);
  }
  int NearestNeighbor(const glm::vec2 query) const {
    return RecursiveNearestNeighbor(0, query, 0);
  }

 protected:
  glm::vec2 SelectMedian(int first, int last, unsigned char dim);
  void RecursiveBuild(int id, int first, int last, unsigned char dim,
                      int depth);
  int RecursiveNearestNeighbor(int id, const glm::vec2& query, int depth) const;
  inline int NaiveNearestNeighbor(int first, int last,
                                  const glm::vec2& query) const;

  std::vector<glm::vec2> points_;
  std::vector<KdNode> nodes_;
  int max_depth_;
  int max_leaf_size_;
};

}  // namespace kdtree
