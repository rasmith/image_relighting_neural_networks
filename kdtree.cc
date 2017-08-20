#include "kdtree.h"

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

namespace kdtree {

void KdTree::SetPoints(const std::vector<glm::vec2>& input_points) {
  points_.resize(input_points.size());
  std::copy(input_points.begin(), input_points.end(), points_.begin());
}

int KdTree::RecursiveNearestNeighbor(const KdNode& node,
				     const glm::vec2& query) {
  if (node.type == KdNode::kLeaf) return node.index;
  int near = (query[node.split_dimension] < node.split_value && node.left != -1
		  ? node.left
		  : node.right);
  int far =
      (query[node.split_dimension] < node.split_value ? node.right : node.left);
  int best_index = RecursiveNearestNeighbor(nodes_[near], query);
  glm::vec2 best_point = points_[best_index];
  float best_distance = glm::distance(best_point, query);
  float split_distance = fabs(query[node.split_dimension] - node.split_value);
  if (split_distance < best_distance && far != -1) {
    int found_index = RecursiveNearestNeighbor(nodes_[far], query);
    glm::vec2 found_point = points_[found_index];
    float found_distance = glm::distance(found_point, query);
    if (found_distance < best_distance) best_index = found_index;
  }
  return best_index;
}

glm::vec2 KdTree::SelectMedian(int first, int last, int dim) {
  int index = (first + last) / 2;
  int size = last - first;
  return (dim == 0 ? quickselect::QuickSelect<glm::vec2, XComparator>(
			 &points_[first], size, index)
		   : quickselect::QuickSelect<glm::vec2, YComparator>(
			 &points_[first], size, index));
}

int KdTree::RecursiveBuild(int first, int last, int dim) {
  if (last - first < 1) return -1;
  int result = nodes_.size();
  nodes_.push_back(KdNode());
  KdNode& node = nodes_.back();
  if (last - first == 1) {
    node.type = KdNode::kLeaf;
    node.index = first;
  } else {
    int median_index = (first + last - 1) / 2;
    glm::vec2 median_point = SelectMedian(first, last, dim);
    node.type = KdNode::kInternal;
    node.split_dimension = dim;
    node.split_value = median_point[dim];
    node.left = RecursiveBuild(first, median_index, (dim + 1) % 2);
    node.right = RecursiveBuild(median_index, last, (dim + 1) % 2);
  }
  return result;
}

}  // namespace kdtree
