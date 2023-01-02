#pragma once

#include <concepts>
#include <tuple>

#include "neuralnet/layer/input_layer.hpp"

namespace neuralnet {

template <class Layers>
struct remove_input_layer;

template <class Layers>
using remove_input_layer_t = typename remove_input_layer<Layers>::type;

template <std::floating_point F, std::size_t N, class... Layers>
struct remove_input_layer<std::tuple<input_layer<F, N>, Layers...>> final {
  using type = std::tuple<Layers...>;
};

template <class Layer, class... Layers>
struct remove_input_layer<std::tuple<Layer, Layers...>> final {
  using type = std::tuple<Layer, Layers...>;
};

}  // namespace neuralnet
