#pragma once

#include <concepts>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

namespace skippy {

// Pooling modes the embedding path supports. Values mirror llama's enum so the
// backend's translation is a cast plus a range check; the backend maps
// anything else (unspecified, rank) to UnsupportedWorkload before it gets here.
enum class Pooling : std::int8_t {
    None = 0, // per-token outputs; one token is selected
    Mean = 1,
    Cls = 2,
    Last = 3,
};

template <Pooling P> using PoolingTag = std::integral_constant<Pooling, P>;

namespace detail {

template <typename Visitor>
using pooling_visit_result_t = std::invoke_result_t<Visitor, PoolingTag<Pooling::None>>;

template <typename Visitor>
concept PoolingVisitor = std::invocable<Visitor, PoolingTag<Pooling::None>> &&
                         std::invocable<Visitor, PoolingTag<Pooling::Mean>> &&
                         std::invocable<Visitor, PoolingTag<Pooling::Cls>> &&
                         std::invocable<Visitor, PoolingTag<Pooling::Last>> &&
                         !std::is_void_v<pooling_visit_result_t<Visitor>> &&
                         std::same_as<pooling_visit_result_t<Visitor>,
                                      std::invoke_result_t<Visitor, PoolingTag<Pooling::Mean>>> &&
                         std::same_as<pooling_visit_result_t<Visitor>,
                                      std::invoke_result_t<Visitor, PoolingTag<Pooling::Cls>>> &&
                         std::same_as<pooling_visit_result_t<Visitor>,
                                      std::invoke_result_t<Visitor, PoolingTag<Pooling::Last>>>;

} // namespace detail

// Runtime -> compile-time dispatch. `visit` is instantiated once per mode and
// receives the mode as a PoolingTag, so `decltype(tag)::value` is a constant
// expression inside it. Every mode is handled; there is no fallthrough.
template <detail::PoolingVisitor Visitor>
[[nodiscard]] constexpr detail::pooling_visit_result_t<Visitor> with_pooling(Pooling pooling,
                                                                             Visitor&& visit) {
    switch (pooling) {
    case Pooling::None:
        return std::invoke(std::forward<Visitor>(visit), PoolingTag<Pooling::None>{});
    case Pooling::Mean:
        return std::invoke(std::forward<Visitor>(visit), PoolingTag<Pooling::Mean>{});
    case Pooling::Cls: return std::invoke(std::forward<Visitor>(visit), PoolingTag<Pooling::Cls>{});
    case Pooling::Last:
        return std::invoke(std::forward<Visitor>(visit), PoolingTag<Pooling::Last>{});
    }
    std::unreachable();
}

} // namespace skippy
