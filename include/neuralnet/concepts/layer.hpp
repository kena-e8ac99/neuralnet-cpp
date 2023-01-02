#pragma once

#include <concepts>

namespace neuralnet {

template <class T>
concept layer = std::floating_point<typename T::real_type> &&
                std::same_as<std::remove_cvref_t<decltype(T::input_size)>,
                             typename T::size_type> &&
                std::same_as<std::remove_cvref_t<decltype(T::output_size)>,
                             typename T::size_type>;

}  // namespace neuralnet
