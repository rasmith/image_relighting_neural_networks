#include "kdtree.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

float RandomFloat(float min_value, float max_value) {
  return std::rand() / static_cast<float>(RAND_MAX) * (max_value - min_value) +
         min_value;
}

glm::vec2 RandomPoint(const glm::vec2* extents) {
  return glm::vec2(RandomFloat(extents[0][0], extents[0][1]),
                   RandomFloat(extents[1][0], extents[1][1]));
}

int NearestNeighborNaive(const std::vector<glm::vec2>& points,
                         const glm::vec2& query) {
  int best = -1;
  float best_distance = std::numeric_limits<float>::max();
  for (int i = 0; i < points.size(); ++i) {
    float distance = glm::distance2(query, points[i]);
    if (distance < best_distance) {
      best_distance = distance;
      best = i;
    }
  }
  return best;
}

std::ostream& operator<<(std::ostream& out, const glm::vec2& p) {
  out << "(" << p.x << "," << p.y << ")";
  return out;
}

int main(int argc, char** argv) {
  std::vector<glm::vec2> points;
  glm::vec2 extents[2] = {glm::vec2(0.0f, 1023.0f), glm::vec2(0.0f, 1023.0f)};
  glm::vec2 extents2[2] = {glm::vec2(0.0f, 16000.0f), glm::vec2(0.0f, 16000.0f)};
  for (int i = 0; i < 1024; ++i) {
    for (int j = 0; j < 1024; ++j) {
      points.push_back(glm::vec2(static_cast<float>(i), static_cast<float>(j)));
    }
  }
  // for (int i = 0; i < size; ++i)
  // std::cout << "points[" << i << "]=(" << points[i].x << ", " << points[i].y
  //<< ")\n";

  int trials = 1000;
  kdtree::KdTree tree;
  // tree.SetMaxLeafSize(std::ceil(std::log2(points.size())));
  tree.AssignPoints(points);
  auto start_build = std::chrono::high_resolution_clock::now();
  tree.Build();
  auto end_build = std::chrono::high_resolution_clock::now();
  std::cout << "points.size() = " << tree.points().size() << "\n";
  std::chrono::duration<double> elapsed_build = end_build - start_build;
  std::cout << "build time = " << elapsed_build.count() << "\n";
  // for (int i = 0; i < size; ++i)
  // std::cout << "nodes.points[" << i << "]=(" << tree.points()[i].x << ", "
  //<< tree.points()[i].y << ")\n";
  // for (int i = 0; i < tree.nodes().size(); ++i) {
  // if (tree.nodes()[i].type == kdtree::KdNode::kLeaf)
  // std::cout << "leaf node @" << tree.nodes()[i].index << "\n";
  //}
  // for (int i = 0; i < tree.nodes().size(); ++i)
  // std::cout << tree.nodes()[i] << "\n";
  kdtree::KdNode node;
  std::cout << "size of KdNode = " << sizeof(node) << "\n";

  auto start_naive = std::chrono::high_resolution_clock::now();
  std::srand(0);
  for (int i = 0; i < trials; ++i) {
    glm::vec2 query = RandomPoint(extents);
    int naive_closest = NearestNeighborNaive(points, query);
  }
  auto end_naive = std::chrono::high_resolution_clock::now();
  auto elapsed_naive = end_naive - start_naive;
  std::cout << "Naive: " << elapsed_naive.count() << "\n";

  auto start_kd = std::chrono::high_resolution_clock::now();
  std::srand(0);
  int tree_closest= -1;
  float best_distance = std::numeric_limits<float>::max();
  for (int i = 0; i < trials; ++i) {
    glm::vec2 query = RandomPoint(extents2);
    tree.NearestNeighbor(query, &tree_closest, &best_distance);
#if 0
    float tree_distance = glm::distance(points[tree_closest], query);
    int naive_closest = NearestNeighborNaive(points, query);
    float naive_distance = glm::distance(points[naive_closest], query);
    bool passed = (fabs(naive_distance - tree_distance) < 1e-5);
    if (!passed) {
      std::cout << "query = " << query << "\n";
      std::cout << i << "/" << trials << "\n";
      std::cout << "naive_closest = " << naive_closest << " " << naive_distance
                << " " << points[naive_closest] << "\n"
                << "tree_closest  = " << tree_closest << " "
                << tree_distance << " " << points[tree_closest] << "\n"
                << "passed = " << passed << "\n";
    tree.SetDebug(true);
    tree.NearestNeighbor(query, &tree_closest, &best_distance);
      break;
    }
#endif
  }
  auto end_kd = std::chrono::high_resolution_clock::now();
  auto elapsed_kd = end_kd - start_kd;
  std::cout << "Kd-Tree: " << elapsed_kd.count() << "\n";
  return 0;
}
