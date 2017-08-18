#pragma once

#include <cstdlib>
#include <vector>
#include <ostream>

namespace quickselect {

template <typename ValueType>
std::ostream& operator<<(std::ostream& out,
                         const std::vector<ValueType>& values) {
  for (const auto& v : values) out << v << " ";
  return out;
}

template <typename ValueType, typename Comparator>
int Partition(std::vector<ValueType>& values, int first, int last, int pivot) {
  Comparator comparator;
  ValueType pivot_value = values[pivot];
  int left = first;
  int right = last;
  int count = 0;
  int i = first;
  while (left < right) {
    if (comparator(values[i], pivot_value)) {
      ++left;
      ++count;
      ++i;
    } else {
      std::swap(values[right - 1], values[i]);
      --right;
    }
  }
  return count;
}

template <typename ValueType, typename Comparator>
ValueType QuickSelectRecursive(std::vector<ValueType>& values, int first,
                               int last, int rank) {
  if (first == last - 1) return values[first];

  int pivot = std::rand() % (last - first) + first;
  ValueType value = values[pivot];
  std::cout << "values = " << values << " pivot = " << pivot << " \n";
  int count = Partition<ValueType, Comparator>(values, first, last, pivot);
  std::cout << "partitioned values = " << values << "\n";
  std::cout << "rank = " << rank << " first = " << first << " last = " << last
            << " value = " << value << " count = " << count << "\n";

  if (count < rank)
    return QuickSelectRecursive<ValueType, Comparator>(values, first + count,
                                                       last, rank - count);
  else if (count > rank)
    return QuickSelectRecursive<ValueType, Comparator>(values, first,
                                                       first + count, rank);
  return value;
}

template <typename ValueType, typename Comparator>
ValueType QuickSelect(std::vector<ValueType>& values, int rank) {
  return QuickSelectRecursive<ValueType, Comparator>(values, 0, values.size(),
                                                     rank);
}

template <typename ValueType, typename Comparator>
ValueType qselect(std::vector<ValueType>& pArray, int k, int li, int hi) {
  Comparator cmp;
  if (hi - li <= 1) return pArray[k];
  int j = li;
  std::swap(pArray[j], pArray[k]);
  for (int i = j = li + 1; i < hi; i++)
    if (cmp(pArray[i], pArray[li])) std::swap(pArray[j++], pArray[i]);
  std::swap(pArray[--j], pArray[li]);
  if (k < j) return qselect<ValueType, Comparator>(pArray, k, li, j);
  if (k > j) return qselect<ValueType, Comparator>(pArray, k - j, j + 1, hi);
  return pArray[j];
}

}  // namespace quickselect
