#include "neuralnet/network.hpp"

#include <boost/ut.hpp>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <utility>

namespace mock {

template <std::size_t O, class... T>
requires(sizeof...(T) <= 1)
  struct layer;

template <std::size_t O, class T>
struct layer<O, T> {
  using real_type = typename T::real_type;
  using size_type = std::size_t;
  using value_type = real_type;
  static constexpr size_type input_size = T::output_size;
  static constexpr size_type output_size = O;
  constexpr real_type value() const noexcept { return value_; }
  real_type value_{};
};

template <std::size_t O>
struct layer<O> {
  template <class T>
  using type = layer<O, T>;

  using size_type = std::size_t;
  static constexpr size_type output_size = O;
};

}  // namespace mock

int main() {
  using namespace boost::ut;

  "default constructor"_test = []<std::floating_point F>() {
    constexpr neuralnet::network<neuralnet::input_layer<F, 1>, mock::layer<2>,
                                 mock::layer<3>>
        network{};

    constexpr auto value = network.value();
    static_assert(std::get<0>(value) == F{});
    static_assert(std::get<1>(value) == F{});
  } | std::tuple<float, double>{};

  "constructor(const&)"_test = []<std::floating_point F>() {
    constexpr auto input_layer = neuralnet::input_layer<F, 1>{};
    constexpr auto hiden_layer = mock::layer<2, decltype(input_layer)>{F{1.0}};
    constexpr auto output_layer =
        mock::layer<3, decltype(hiden_layer)>{F{-1.0}};
    constexpr auto network =
        neuralnet::network{input_layer, hiden_layer, output_layer};

    constexpr auto value = network.value();
    static_assert(std::get<0>(value) == F{1.0});
    static_assert(std::get<1>(value) == F{-1.0});
  } | std::tuple<float, double>{};

  "constructor(&&)"_test = []<std::floating_point F>() {
    using input_layer_t = neuralnet::input_layer<F, 1>;
    using hiden_layer_t = mock::layer<2, input_layer_t>;
    using output_layer_t = mock::layer<3, hiden_layer_t>;
    constexpr auto network = neuralnet::network{
        input_layer_t{}, hiden_layer_t{F{1.0}}, output_layer_t{F{-1.0}}};

    constexpr auto value = network.value();
    static_assert(std::get<0>(value) == F{1.0});
    static_assert(std::get<1>(value) == F{-1.0});
  } | std::tuple<float, double>{};
}
