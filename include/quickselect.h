#pragma once

#include <cstdlib>
#include <ostream>
#include <vector>

namespace quickselect {

template <typename ValueType>
std::ostream& operator<<(std::ostream& out,
                         const std::vector<ValueType>& values) {
  for (const auto& v : values) out << v << " ";
  return out;
}

template <typename ValueType, typename Comparator>
int Partition(ValueType* values, int first, int last, int pivot) {
  Comparator comparator;
  ValueType pivot_value = values[pivot];
  int left = first + 1;
  int right = last;
  int count = 0;
  std::swap(values[first], values[pivot]);
  while (left < right) {
    if (comparator(values[left], pivot_value)) {
      ++count;
      ++left;
    } else {
      std::swap(values[right - 1], values[left]);
      --right;
    }
  }
  std::swap(values[first], values[left - 1]);
  return count;
}

template <typename ValueType, typename Comparator>
ValueType QuickSelectRecursive(ValueType* values, int first, int last,
                               int rank) {
  if (last - first <= 1) return values[first];
  int pivot = std::rand() % (last - first) + first;
  ValueType value = values[pivot];
  int count = Partition<ValueType, Comparator>(values, first, last, pivot);
  if (first + count < rank)
    return QuickSelectRecursive<ValueType, Comparator>(
        values, first + count + 1, last, rank);
  else if (first + count > rank)
    return QuickSelectRecursive<ValueType, Comparator>(values, first,
                                                       first + count, rank);
  return value;
}

template <typename ValueType, typename Comparator>
ValueType QuickSelect(ValueType* values, int size, int rank) {
  return QuickSelectRecursive<ValueType, Comparator>(values, 0, size, rank);
}

}  // namespace quickselect
