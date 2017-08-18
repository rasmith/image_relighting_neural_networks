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
int Partition(std::vector<ValueType>& values, int first, int last, int pivot) {
  Comparator comparator;
  ValueType pivot_value = values[pivot];
  int left = first + 1;
  int right = last;
  int count = 0;
  int i = left;
  std::swap(values[first], values[pivot]);
  while (left < right) {
    if (comparator(values[left], pivot_value)) {
      ++count;
      ++left;
    } else {
      std::swap(values[right - 1], values[left]);
      --right;
    }
    //std::cout << "left = " << left << " right = " << right
              //<< " values = " << values << "\n ";
    // if (comparator(values[i], pivot_value)) {
    //++left;
    //++count;
    //++i;
    //} else {
    // std::swap(values[right - 1], values[i]);
    //--right;
    //}
  }
  std::cout << "(1) values = " << values << "\n";
  std::swap(values[first], values[left - 1]);
  std::cout << "(2) values = " << values << "\n";
  return count;
}

template <typename ValueType, typename Comparator>
ValueType QuickSelectRecursive(std::vector<ValueType>& values, int first,
                               int last, int rank) {
  if (last - first <= 1) return values[first];

  int pivot = std::rand() % (last - first) + first;
  ValueType value = values[pivot];
  std::vector<char> markers(values.size(), ' ');
  markers[pivot] = '*';
  std::cout << "         " << markers << "\n";
  std::cout << "values = " << values << " pivot = " << pivot
            << " pivot_value = " << value << "\n";
  int count = Partition<ValueType, Comparator>(values, first, last, pivot);
  std::cout << "                     " << markers << "\n";
  std::cout << "partitioned values = " << values << "\n";
  std::cout << "rank = " << rank << " first = " << first << " last = " << last
            << " count = " << count << "\n";

  if (first + count < rank)
    return QuickSelectRecursive<ValueType, Comparator>(
        values, first + count + 1, last, rank);
  else if (first + count > rank)
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
