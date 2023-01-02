#include "neuralnet/type_traits/connect_layers.hpp"

#include <concepts>
#include <cstdint>

namespace mock {

template <std::floating_point F, std::size_t I, std::size_t O>
struct initialized_layer {
  using real_type = F;
  using size_type = std::size_t;
  static constexpr size_type input_size = I;
  static constexpr size_type output_size = O;
};

template <std::size_t O>
struct uninitialized_layer {
  using size_type = std::size_t;
  static constexpr size_type output_size = O;
  template <class T>
  using type = initialized_layer<typename T::real_type, T::output_size, O>;
};

}  // namespace mock

int main() {
  static_assert(
      std::same_as<
          neuralnet::connect_layers_t<mock::initialized_layer<float, 1, 2>,
                                      mock::initialized_layer<float, 2, 3>,
                                      mock::initialized_layer<float, 3, 4>>,
          std::tuple<mock::initialized_layer<float, 1, 2>,
                     mock::initialized_layer<float, 2, 3>,
                     mock::initialized_layer<float, 3, 4>>>);
  static_assert(
      std::same_as<
          neuralnet::connect_layers_t<mock::initialized_layer<float, 1, 2>,
                                      mock::initialized_layer<float, 2, 3>,
                                      mock::uninitialized_layer<4>>,
          std::tuple<mock::initialized_layer<float, 1, 2>,
                     mock::initialized_layer<float, 2, 3>,
                     mock::initialized_layer<float, 3, 4>>>);
  static_assert(
      std::same_as<
          neuralnet::connect_layers_t<mock::initialized_layer<float, 1, 2>,
                                      mock::uninitialized_layer<3>,
                                      mock::uninitialized_layer<4>>,
          std::tuple<mock::initialized_layer<float, 1, 2>,
                     mock::initialized_layer<float, 2, 3>,
                     mock::initialized_layer<float, 3, 4>>>);
  static_assert(
      std::same_as<
          neuralnet::connect_layers_t<mock::initialized_layer<float, 1, 2>,
                                      mock::uninitialized_layer<3>,
                                      mock::initialized_layer<float, 3, 4>>,
          std::tuple<mock::initialized_layer<float, 1, 2>,
                     mock::initialized_layer<float, 2, 3>,
                     mock::initialized_layer<float, 3, 4>>>);
}
