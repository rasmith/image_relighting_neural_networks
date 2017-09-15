#pragma once

#include <vector>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>

void kmeans(int width, int height, std::vector<glm::vec2>& centers,
            std::vector<int>& labels);
