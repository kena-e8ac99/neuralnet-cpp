#include "neuralnet/type_traits/remove_input_layer.hpp"

#include <concepts>
#include <cstdint>

#include "neuralnet/layer/input_layer.hpp"

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
  static_assert(
      std::same_as<
          neuralnet::remove_input_layer_t<std::tuple<
              neuralnet::input_layer<float, 1>, mock::layer<float, 1, 2>>>,
          std::tuple<mock::layer<float, 1, 2>>>);
  static_assert(
      std::same_as<
          neuralnet::remove_input_layer_t<
              std::tuple<neuralnet::input_layer<float, 1>,
                         mock::layer<float, 1, 2>, mock::layer<float, 2, 3>>>,
          std::tuple<mock::layer<float, 1, 2>, mock::layer<float, 2, 3>>>);
}
