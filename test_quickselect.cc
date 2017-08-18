#include <iostream>
#include <vector>

#include "quickselect.h"

template <typename ValueType>
struct ValueComparator {
  ValueComparator() {}
  bool operator()(const ValueType& a, const ValueType& b) { return a < b; }
};

int main(int argc, char** argv) {
  std::vector<int> a1 = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  std::vector<int> test(a1.size());

  std::cout << "original = ";
  quickselect::operator<<(std::cout, a1);
  std::cout << "\n";

  for (int i = 0; i < a1.size(); ++i) {
    // int i = 1;
    std::copy(a1.begin(), a1.end(), test.begin());
    // std::cout << "QuickSelect(test, " << i << ") = "
    //<< quickselect::QuickSelect<int, ValueComparator<int>>(test, i)
    //<< "\n";
    std::cout << "QuickSelect(test, " << i
              << ") = " << quickselect::qselect<int, ValueComparator<int>>(
                               test, i, 0, a1.size() - 1) << "\n";
  }

  // i = 1;
  // std::copy(a1.begin(), a1.end(), test.begin());
  // std::cout << "QuickSelect(test, " << i << ") = "
  //<< quickselect::QuickSelect<int, ValueComparator<int>>(test, i)
  //<< "\n";
  // for (int i = 1; i <= 10; ++i) {
  // std::cout << "value = " << a1[i] << "\n";
  // int count =
  // quickselect::Partition<int, ValueComparator<int>>(a1, 0, a1.size(), i);
  // quickselect::operator<<(std::cout, a1);
  // std::cout << " count = " << count << "\n";
  //}
  return 0;
}
