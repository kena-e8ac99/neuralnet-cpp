#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace neuralnet {

template <std::floating_point F, std::size_t... N>
requires (sizeof...(N) > 0)
struct input_layer final {
  // Type Definition
  using size_type = std::conditional_t<sizeof...(N) == 1, std::size_t,
                                       std::array<std::size_t, sizeof...(N)>>;
  using real_type = F;

  // Static Member
  static constexpr size_type input_size = size_type{N...};
  static constexpr size_type output_size = size_type{N...};
};

}  // namespace neuralnet
