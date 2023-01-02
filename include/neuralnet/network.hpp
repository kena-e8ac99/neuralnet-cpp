#pragma once

#include <array>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

#include "neuralnet/layer/input_layer.hpp"
#include "neuralnet/type_traits/connect_layers.hpp"
#include "neuralnet/type_traits/remove_input_layer.hpp"

namespace neuralnet {

template <class... T>
class network;

template <std::floating_point F, std::size_t... N, class... Layers>
requires (sizeof...(Layers) >= 2)
class network<input_layer<F, N...>, Layers...> final {
 public:
  // Type Definition
  using real_type = F;

  using layers_type =
      remove_input_layer_t<connect_layers_t<input_layer<F, N...>, Layers...>>;

  using value_type = decltype([]<std::size_t... I>(std::index_sequence<I...>) {
    return std::tuple{
        typename std::tuple_element_t<I, layers_type>::value_type{}...};
  }(std::make_index_sequence<sizeof...(Layers)>{}));

  template <std::size_t I>
  using layer_type = std::tuple_element_t<I, layers_type>;

  using input_layer_type = input_layer<F, N...>;

  using output_layer_type =
      std::tuple_element_t<sizeof...(Layers) - 1, layers_type>;

  using output_type = std::array<real_type, output_layer_type::output_size>;

  // Static Member
  static constexpr std::size_t size = sizeof...(Layers);

  // Constructor
  network() = default;

  explicit constexpr network([[maybe_unused]] input_layer_type input_layer,
                             const Layers&... layers) noexcept
      : layers_(layers...) {}

  explicit constexpr network([[maybe_unused]] input_layer_type input_layer,
                             Layers&&... layers) noexcept
      : layers_(std::move(layers)...) {}

  // Public Method
  constexpr value_type value() const noexcept {
    return [this]<std::size_t... I>(std::index_sequence<I...>) {
      return std::tuple{this->layer<I>().value()...};
    } (std::make_index_sequence<size>{});
  }

 private:
  // Private Member
  layers_type layers_{};

  // Private Method
  template <std::size_t I>
  requires(I < size)
  constexpr const layer_type<I>& layer() const noexcept {
    return std::get<I>(layers_);
  }
};

// Deduction Guide
template <std::floating_point F, std::size_t N, class... Layers>
network(input_layer<F, N>, const Layers&...)
    -> network<input_layer<F, N>, Layers...>;

template <std::floating_point F, std::size_t N, class... Layers>
network(input_layer<F, N>, Layers&&...)
    -> network<input_layer<F, N>, Layers...>;

}  // namespace neuralnet
