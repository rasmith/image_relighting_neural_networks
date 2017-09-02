#include "kdtree.h"

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include <iostream>

namespace kdtree {

std::ostream& operator<<(std::ostream& out, const glm::vec2& p) {
  out << "(" << p.x << "," << p.y << ")";
  return out;
}

std::ostream& operator<<(std::ostream& out, const KdNode& node) {
  if (node.type == KdNode::kInternal) {
    out << "[I L:" << node.info.internal.left
        << " R:" << node.info.internal.right
        << " D:" << (int)node.split_dimension
        << " V:" << node.info.internal.split_value << "]";
  } else {
    out << "[L I:" << node.info.leaf.location
        << " P: " << node.info.leaf.position << "]";
  }
  return out;
}

void KdTree::AssignPoints(const std::vector<glm::vec2>& input_points) {
  points_.resize(input_points.size());
  std::copy(input_points.begin(), input_points.end(), points_.begin());
}

void KdTree::RecursiveNearestNeighbor(int id, const glm::vec2& query, int depth,
                                      int* best, float* best_distance) const {
  const KdNode& node = nodes_[id];
  if (node.type == KdNode::kLeaf) {
    float distance = glm::distance2(query, node.info.leaf.position);
    if (distance < *best_distance) {
      *best = node.info.leaf.location;
      *best_distance = distance;
    }
    return;
  }
  bool is_left = (query[node.split_dimension] < node.info.internal.split_value);
  int near =
      (is_left && node.info.internal.left != -1 ? node.info.internal.left
                                                : node.info.internal.right);
  int far =
      (is_left && node.info.internal.left != -1 ? node.info.internal.right
                                                : node.info.internal.left);
  RecursiveNearestNeighbor(near, query, depth + 1, best, best_distance);
  float split_distance =
      fabs(query[node.split_dimension] - node.info.internal.split_value);
  if (split_distance * split_distance < *best_distance && far != -1)
    RecursiveNearestNeighbor(far, query, depth + 1, best, best_distance);
}

PointWrapper KdTree::SelectMedian(int first, int last, unsigned char dim) {
  int size = last - first;
  int index = (size - 1) / 2;
  return (dim == 0 ? quickselect::QuickSelect<PointWrapper, XComparator>(
                         &point_wrappers_[first], size, index)
                   : quickselect::QuickSelect<PointWrapper, YComparator>(
                         &point_wrappers_[first], size, index));
}

void KdTree::RecursiveBuild(int node_index, int first, int last,
                            unsigned char dim, int depth) {
  KdNode node;

  if (last - first == 1) {
    node.type = KdNode::kLeaf;
    // node.info.leaf.position = points_[first];
    // node.info.leaf.location = first;
    node.info.leaf.position = point_wrappers_[first].position;
    node.info.leaf.location = point_wrappers_[first].location;
  } else {
    int median_index = first + (last - first - 1) / 2;
    PointWrapper median_point = SelectMedian(first, last, dim);
    // std::cout << "first = " << first << " last = " << last <<
    //" median_index = " << median_index << "\n";
    // std::cout << "median = " << median_point << "\n";
    node.type = KdNode::kInternal;
    node.split_dimension = dim;
    node.info.internal.split_value = median_point[dim];
    node.info.internal.left = -1;
    node.info.internal.right = -1;
    if (median_index + 1 - first >= 1) {
      node.info.internal.left = build_nodes_.size();
      build_nodes_.push_back(KdNode());
    }
    if (last - median_index - 1 >= 1) {
      node.info.internal.right = build_nodes_.size();
      build_nodes_.push_back(KdNode());
    }
    if (median_index + 1 - first >= 1) {
      RecursiveBuild(node.info.internal.left, first, median_index + 1,
                     (dim + 1) % 2, depth + 1);
    }
    if (last - median_index - 1 >= 1) {
      RecursiveBuild(node.info.internal.right, median_index + 1, last,
                     (dim + 1) % 2, depth + 1);
    }
  }
  build_nodes_[node_index] = node;
}

}  // namespace kdtree
