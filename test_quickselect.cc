#include <iostream>
#include <vector>

#include "quickselect.h"

template <typename ValueType>
struct ValueComparator {
  ValueComparator() {}
  bool operator()(const ValueType& a, const ValueType& b) { return a < b; }
};

template <int TestNumber>
void TestDuplicatesInArray() {
  std::cout << "================================================\n";
  std::cout << "  TestDuplicatesInArray     " << TestNumber << "\n";
  std::cout << "------------------------------------------------\n";
  std::vector<int> test_values = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  int result = quickselect::QuickSelect<int, ValueComparator<int>>(test_values,
                                                                   TestNumber);
  std::cout << "QuickSelect(test, " << TestNumber << ") = " << result << "\n";
  std::cout << "================================================\n";
}

template <int TestNumber>
void TestDistinctArray() {
  std::cout << "================================================\n";
  std::cout << "  TestDistinctArray        " << TestNumber << "\n";
  std::cout << "------------------------------------------------\n";
  std::vector<int> test_values = {9, 1, 3, 4, 0, 8, 11, 7, 10, 6, 5, 2};
  int result = quickselect::QuickSelect<int, ValueComparator<int>>(test_values,
                                                                   TestNumber);
  std::cout << "QuickSelect(test, " << TestNumber << ") = " << result << "\n";
  std::cout << "================================================\n";
}

template <int TestNumber>
void TestDistinctArray2() {
  std::cout << "================================================\n";
  std::cout << "  TestDistinctArray2        " << TestNumber << "\n";
  std::cout << "------------------------------------------------\n";
  std::vector<int> test_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int result = quickselect::QuickSelect<int, ValueComparator<int>>(test_values,
                                                                   TestNumber);
  std::cout << "QuickSelect(test, " << TestNumber << ") = " << result << "\n";
  std::cout << "================================================\n";
}

template <int TestNumber>
void TestDuplicatesInArray2() {
  std::cout << "================================================\n";
  std::cout << "  TestDuplicatesInArray2    " << TestNumber << "\n";
  std::cout << "------------------------------------------------\n";
  std::vector<int> test_values = {1, 3, 1, 4, 2, 2, 5, 6, 6, 12, 7, 9};
  int result = quickselect::QuickSelect<int, ValueComparator<int>>(test_values,
                                                                   TestNumber);
  std::cout << "QuickSelect(test, " << TestNumber << ") = " << result << "\n";
  std::cout << "================================================\n";
}

int main(int argc, char** argv) {
  TestDuplicatesInArray<0>();
  TestDuplicatesInArray<1>();
  TestDuplicatesInArray<2>();
  TestDuplicatesInArray<3>();
  TestDuplicatesInArray<4>();
  TestDuplicatesInArray<5>();
  TestDuplicatesInArray<6>();
  TestDuplicatesInArray<7>();
  TestDuplicatesInArray<8>();
  TestDuplicatesInArray<9>();
  TestDuplicatesInArray<10>();
  TestDuplicatesInArray<11>();
  TestDuplicatesInArray2<0>();
  TestDuplicatesInArray2<1>();
  TestDuplicatesInArray2<2>();
  TestDuplicatesInArray2<3>();
  TestDuplicatesInArray2<4>();
  TestDuplicatesInArray2<5>();
  TestDuplicatesInArray2<6>();
  TestDuplicatesInArray2<7>();
  TestDuplicatesInArray2<8>();
  TestDuplicatesInArray2<9>();
  TestDuplicatesInArray2<10>();
  TestDuplicatesInArray2<11>();
  TestDistinctArray<0>();
  TestDistinctArray<1>();
  TestDistinctArray<2>();
  TestDistinctArray<3>();
  TestDistinctArray<4>();
  TestDistinctArray<5>();
  TestDistinctArray<6>();
  TestDistinctArray<7>();
  TestDistinctArray<8>();
  TestDistinctArray<9>();
  TestDistinctArray<10>();
  TestDistinctArray<11>();
  TestDistinctArray2<0>();
  TestDistinctArray2<1>();
  TestDistinctArray2<2>();
  TestDistinctArray2<3>();
  TestDistinctArray2<4>();
  TestDistinctArray2<5>();
  TestDistinctArray2<6>();
  TestDistinctArray2<7>();
  TestDistinctArray2<8>();
  TestDistinctArray2<9>();
  TestDistinctArray2<10>();
  TestDistinctArray2<11>();
  return 0;
}
