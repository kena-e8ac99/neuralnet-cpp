#include "neuralnet/layer/input_layer.hpp"

#include <cstddef>

int main() {
  static_assert(
      std::same_as<neuralnet::input_layer<float, 1>::real_type, float>);
  static_assert(
      std::same_as<neuralnet::input_layer<double, 1>::real_type, double>);
  static_assert(
      std::same_as<neuralnet::input_layer<float, 1>::size_type, std::size_t>);
  static_assert(
      std::same_as<neuralnet::input_layer<double, 1>::size_type, std::size_t>);
  static_assert(neuralnet::input_layer<float, 1>::input_size == 1);
  static_assert(neuralnet::input_layer<float, 2>::input_size == 2);
  static_assert(neuralnet::input_layer<double, 1>::input_size == 1);
  static_assert(neuralnet::input_layer<double, 2>::input_size == 2);
  static_assert(
      std::same_as<neuralnet::input_layer<float, 1, 2>::real_type, float>);
  static_assert(
      std::same_as<neuralnet::input_layer<double, 1, 2>::real_type, double>);
  static_assert(std::same_as<neuralnet::input_layer<float, 1, 2>::size_type,
                             std::array<std::size_t, 2>>);
  static_assert(std::same_as<neuralnet::input_layer<double, 1, 2>::size_type,
                             std::array<std::size_t, 2>>);
  static_assert(neuralnet::input_layer<float, 1, 2>::input_size ==
                std::array<std::size_t, 2>{1, 2});
  static_assert(neuralnet::input_layer<double, 1, 2>::input_size ==
                std::array<std::size_t, 2>{1, 2});
}
