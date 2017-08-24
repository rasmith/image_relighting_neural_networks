#include "kdtree.h"

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include <iostream>

namespace kdtree {

std::ostream& operator<<(std::ostream& out, const KdNode& node) {
  if (node.type == KdNode::kInternal) {
    out << "[I L:" << node.left << " R:" << node.right
	<< " D:" << node.split_dimension << " V:" << node.split_value << "]";
  } else {
    out << "[L P:" << node.left << " N: " << node.right - node.left << "]";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const glm::vec2& p) {
  out << "(" << p.x << "," << p.y << ")";
  return out;
}

void KdTree::AssignPoints(const std::vector<glm::vec2>& input_points) {
  points_.resize(input_points.size());
  std::copy(input_points.begin(), input_points.end(), points_.begin());
}

inline int KdTree::NaiveNearestNeighbor(int first, int last,
				 const glm::vec2& query) const {
  int best = -1;
  float best_distance = std::numeric_limits<float>::max();
  for (int i = first; i < last; ++i) {
    float distance = glm::distance2(query, points_[i]);
    if (distance < best_distance) {
      best_distance = distance;
      best = i;
    }
  }
  return best;
}

int KdTree::RecursiveNearestNeighbor(int id, const glm::vec2& query,
				     int depth) const {
  const KdNode& node = nodes_[id];
  // for (int i = 0; i < depth; ++i) std::cout << " ";
  // std::cout << "node[" << id << "] = " << node << "\n";
  if (node.type == KdNode::kLeaf)
    return NaiveNearestNeighbor(node.left, node.right, query);

  bool is_left = (query[node.split_dimension] < node.split_value);
  int near = (is_left && node.left != -1 ? node.left : node.right);
  int far = (is_left && node.left != -1 ? node.right : node.left);
  int best_index = RecursiveNearestNeighbor(near, query, depth + 1);
  glm::vec2 best_point = points_[best_index];
  float best_distance = glm::distance2(best_point, query);
  float split_distance = fabs(query[node.split_dimension] - node.split_value);
  if (split_distance < best_distance && far != -1) {
    int found_index = RecursiveNearestNeighbor(far, query, depth + 1);
    glm::vec2 found_point = points_[found_index];
    float found_distance = glm::distance2(found_point, query);
    if (found_distance < best_distance) best_index = found_index;
  }
  return best_index;
}

glm::vec2 KdTree::SelectMedian(int first, int last, unsigned char dim) {
  int size = last - first;
  int index = (size - 1) / 2;
  return (dim == 0 ? quickselect::QuickSelect<glm::vec2, XComparator>(
			 &points_[first], size, index)
		   : quickselect::QuickSelect<glm::vec2, YComparator>(
			 &points_[first], size, index));
}

void KdTree::RecursiveBuild(int node_index, int first, int last, unsigned char dim,
			    int depth) {
  KdNode node;

  // std::cout << "RecursiveBuild: " << depth <<  " dim = " << dim <<"\n";
  // for (int i = first; i < last; ++i) {
  // std::cout << points_[i] << "\n";
  //}

  if (last - first == 1 || depth >= max_depth_ ||
      last - first <= max_leaf_size_) {
    node.type = KdNode::kLeaf;
    node.left = first;
    node.right = last;
  } else {
    int median_index = first + (last - first - 1) / 2;
    glm::vec2 median_point = SelectMedian(first, last, dim);
    // std::cout << "first = " << first << " last = " << last <<
    //" median_index = " << median_index << "\n";
    // std::cout << "median = " << median_point << "\n";
    node.type = KdNode::kInternal;
    node.split_dimension = dim;
    node.split_value = median_point[dim];
    node.left = -1;
    node.right = -1;
    if (median_index + 1 - first >= 1) {
      node.left = nodes_.size();
      nodes_.push_back(KdNode());
      RecursiveBuild(node.left, first, median_index + 1, (dim + 1) % 2,
		     depth + 1);
    }
    if (last - median_index - 1 >= 1) {
      node.right = nodes_.size();
      nodes_.push_back(KdNode());
      RecursiveBuild(node.right, median_index + 1, last, (dim + 1) % 2,
		     depth + 1);
    }
  }
  nodes_[node_index] = node;
}

}  // namespace kdtree
