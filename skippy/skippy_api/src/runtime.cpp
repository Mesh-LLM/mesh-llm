#include "runtime.h"

#include <atomic>
#include <cstdint>
#include <expected>

namespace skippy::detail {

// We use 2^63-1 as our counter, this means we can avoid a CAS loop.
// once bit 63 is set, we are exhausted.
// hopefully never. Starts at 1.
std::expected<std::uint64_t, RuntimeError> next_generation() noexcept {
    static std::atomic<std::uint64_t> next{1};
    constexpr auto exhausted = std::uint64_t{1} << 63U;
    const auto value = next.fetch_add(1, std::memory_order_relaxed);
    if ((value & exhausted) != 0U) [[unlikely]] {
        return std::unexpected(RuntimeError::CapacityExceeded);
    }
    return value;
}

} // namespace skippy::detail
