// T6 (docs/TYPE_DESIGN.md): the only file that spells static_cast between
// integer types. A width change that can lose value is an optional the caller
// must handle, never a silent cast.
#pragma once

#include <concepts>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace skippy {

template <std::integral To, std::integral From>
[[nodiscard]] constexpr std::optional<To> narrow(From value) noexcept {
    return std::in_range<To>(value) ? std::optional<To>{static_cast<To>(value)} : std::nullopt;
}

template <std::unsigned_integral T>
[[nodiscard]] constexpr std::optional<T> checked_mul(T a, T b) noexcept {
#if __has_builtin(__builtin_mul_overflow)
    T out{};
    return __builtin_mul_overflow(a, b, &out) ? std::nullopt : std::optional<T>{out};
#else
    return (b != 0 && a > std::numeric_limits<T>::max() / b) ? std::nullopt
                                                             : std::optional<T>{a * b};
#endif
}

template <std::unsigned_integral T>
[[nodiscard]] constexpr std::optional<T> checked_add(T a, T b) noexcept {
    return a > std::numeric_limits<T>::max() - b ? std::nullopt : std::optional<T>{a + b};
}

} // namespace skippy
