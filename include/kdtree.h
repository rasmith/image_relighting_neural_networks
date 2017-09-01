#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "quickselect.h"

namespace kdtree {

template <typename PointType, int DimensionNumber>
struct PointComparator {
  PointComparator() {}
  bool operator()(const PointType& a, const PointType& b) {
    return a[DimensionNumber] < b[DimensionNumber];
  }
};

struct PointWrapper {
  PointWrapper() {}
  PointWrapper(const PointWrapper& p)
      : position(p.position), location(p.location) {}
  PointWrapper(const glm::vec2& p, int i) : position(p), location(i) {}
  float operator[](int i) const { return position[i]; }
  glm::vec2 position;
  int location;
};

struct KdNode {
  enum Type { kInternal = 0, kLeaf = 1 };

  KdNode() : type(kInternal), split_dimension(0) {}

  KdNode(const KdNode& node)
      : type(node.type), split_dimension(node.split_dimension) {
    if (node.type == kInternal)
      info.internal = node.info.internal;
    else
      info.leaf = node.info.leaf;
  }

  unsigned char type : 1;
  unsigned char split_dimension : 3;
  union NodeInfo {
    NodeInfo() {}
    struct InternalInfo {
      InternalInfo() {}
      int left;
      int right;
      float split_value;
    } internal;
    struct LeafInfo {
      LeafInfo() {}
      glm::vec2 position;
      int location;
    } leaf;
  } info;
};

std::ostream& operator<<(std::ostream& out, const KdNode& node);

class KdTree {
 public:
  typedef PointComparator<PointWrapper, 0> XComparator;
  typedef PointComparator<PointWrapper, 1> YComparator;

  KdTree() :debug_(false){}

  void AssignPoints(const std::vector<glm::vec2>& input_points);
  const std::vector<glm::vec2>& points() const { return points_; }
  const KdNode* nodes() const { return nodes_; }
  void SetMaxDepth(int depth) { max_depth_ = depth; }
  void SetMaxLeafSize(int size) { max_leaf_size_ = size; }
  void SetDebug(bool v) { debug_ = true; }

  void Build() {
    build_nodes_.clear();
    build_nodes_.push_back(KdNode());
    point_wrappers_.resize(points_.size());
    for (int i = 0; i < points_.size(); ++i)
      point_wrappers_[i] = PointWrapper(points_[i], i);
    RecursiveBuild(0, 0, points_.size(), 0, 0);
    nodes_ = new KdNode[build_nodes_.size()];
    std::copy(build_nodes_.begin(), build_nodes_.end(), nodes_);
    build_nodes_.clear();
  }
  void NearestNeighbor(const glm::vec2 query, int* best,
                       float* best_distance) const {
    *best = -1;
    *best_distance = std::numeric_limits<float>::max();
    //IterativeNearestNeighbor(0, query, best, best_distance);
    RecursiveNearestNeighbor(0, query, 0, best, best_distance);
  }

 protected:
  PointWrapper SelectMedian(int first, int last, unsigned char dim);
  void RecursiveBuild(int id, int first, int last, unsigned char dim,
                      int depth);
  void RecursiveNearestNeighbor(int id, const glm::vec2& query, int depth,
                                int* best, float* best_distance) const;
  inline void IterativeNearestNeighbor(int id, const glm::vec2& query,
                                       int* best, float* best_distance) const {
    static int far_nodes[128];
    static float split_distances[128];
    int back = 0;
    far_nodes[back] = id;
    split_distances[back] = std::numeric_limits<float>::max();
    while (back >= 0) {
      assert(back < 128);
      const KdNode* current = &nodes_[far_nodes[back]];
      float split_distance = split_distances[back];
      --back;
      if (split_distance < *best_distance) {
        while (current->type != KdNode::kLeaf) {
          bool is_left = (query[current->split_dimension] <
                          current->info.internal.split_value);
          int near = (is_left && current->info.internal.left != -1
                          ? current->info.internal.left
                          : current->info.internal.right);
          int far = (is_left && current->info.internal.left != -1
                         ? current->info.internal.right
                         : current->info.internal.left);
          if (far != -1) {
            ++back;
            far_nodes[back] = far;
            split_distances[back] = fabs(query[current->split_dimension] -
                                         current->info.internal.split_value);
          }
          current = &nodes_[near];
        }
        float distance = glm::distance2(query, current->info.leaf.position);
        if (distance < *best_distance) {
          *best = current->info.leaf.location;
          *best_distance = distance;
        }
      }
    }
  }

  std::vector<glm::vec2> points_;
  KdNode* nodes_;
  std::vector<PointWrapper> point_wrappers_;
  std::vector<KdNode> build_nodes_;
  int max_depth_;
  int max_leaf_size_;
  bool debug_;
};

}  // namespace kdtree
