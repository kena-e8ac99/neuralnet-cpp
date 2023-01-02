#pragma once

#include <concepts>

#include "neuralnet/concepts/layer.hpp"

namespace neuralnet {

template <class T, class U>
concept connectable_layers = layer<T> && layer<U> &&
    std::same_as<typename T::real_type, typename U::real_type> &&
    (T::output_size == U::input_size);

}  // namespace neuralnet
