module;

#include <vector>
#include <iostream>

export module vector_utils;

export void print_vector(std::vector<double> vec) {
  for (const auto& element : vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}
