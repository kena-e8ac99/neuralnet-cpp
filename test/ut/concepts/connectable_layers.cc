#include "neuralnet/concepts/connectable_layers.hpp"

#include <concepts>
#include <cstdint>

namespace mock {

template <std::floating_point F, std::size_t I, std::size_t O>
struct layer {
  using real_type = F;
  using size_type = std::size_t;
  static constexpr size_type input_size = I;
  static constexpr size_type output_size = O;
};

}  // namespace mock

int main() {
  static_assert(neuralnet::connectable_layers<mock::layer<float, 1, 2>,
                                              mock::layer<float, 2, 3>>);
  static_assert(!neuralnet::connectable_layers<mock::layer<float, 1, 2>,
                                               mock::layer<float, 1, 3>>);
  static_assert(neuralnet::connectable_layers<mock::layer<double, 1, 2>,
                                              mock::layer<double, 2, 3>>);
  static_assert(!neuralnet::connectable_layers<mock::layer<double, 1, 2>,
                                               mock::layer<double, 1, 2>>);
  static_assert(!neuralnet::connectable_layers<mock::layer<float, 1, 2>,
                                               mock::layer<double, 2, 3>>);
  static_assert(!neuralnet::connectable_layers<mock::layer<double, 1, 2>,
                                               mock::layer<float, 2, 3>>);
}
