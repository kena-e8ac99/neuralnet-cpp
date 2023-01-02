#pragma once

#include <concepts>
#include <tuple>

#include "neuralnet/concepts/connectable_layers.hpp"

namespace neuralnet {

template <class... T>
struct connect_layers;

template <class... T>
using connect_layers_t = typename connect_layers<T...>::type;

template <class T, class U, class... V>
requires connectable_layers<T, U>
struct connect_layers<T, U, V...> final {
  using type = typename connect_layers<std::tuple<T>, U, V...>::type;
};

template <class T, class U, class... V>
requires connectable_layers<T, typename U::template type<T>>
struct connect_layers<T, U, V...> final {
  using type =
      typename connect_layers<std::tuple<T>, typename U::template type<T>,
                              V...>::type;
};

template <class T, class U, class... V>
requires connectable_layers<T, U>
struct connect_layers<std::tuple<V...>, T, U> final {
  using type = std::tuple<V..., T, U>;
};

template <class T, class U, class... V>
requires connectable_layers<T, typename U::template type<T>>
struct connect_layers<std::tuple<V...>, T, U> final {
  using type = std::tuple<V..., T, typename U::template type<T>>;
};

}  // namespace neuralnet
