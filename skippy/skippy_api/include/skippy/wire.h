// T7 (docs/TYPE_DESIGN.md): bytes off a file decode through WireScalar, and
// the big-endian path exists instead of being assumed away.
#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>

namespace skippy {

template <typename T>
concept WireScalar =
    (std::integral<T> || std::floating_point<T>) && std::has_unique_object_representations_v<T>;

template <WireScalar T>
[[nodiscard]] T from_little_endian(std::span<const std::byte, sizeof(T)> bytes) noexcept {
    T value;
    std::memcpy(&value, bytes.data(), sizeof value);
    if constexpr (std::endian::native == std::endian::big && std::integral<T>) {
        value = std::byteswap(value);
    }
    return value;
}

// The GGUF parser assumes a 64-bit address space for offsets.
static_assert(sizeof(std::size_t) >= 8);

} // namespace skippy
