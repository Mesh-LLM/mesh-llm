#ifndef SKIPPY_API_RECURRENT_ROLLBACK_H
#define SKIPPY_API_RECURRENT_ROLLBACK_H

/**
 * @file recurrent_rollback.h
 * @brief Bounded rollback depth for in-place speculative verify (patch 0015).
 *
 * A recurrent or hybrid context verifies an N-token speculative window in
 * place only while it keeps N rollback snapshots of the recurrent state per
 * sequence (native `n_rs_seq`); fewer snapshots and verify falls back to a
 * host copy, while each extra snapshot is another copy of the state. The
 * caller sizes the pool from its speculative plan.
 */

#include <algorithm>
#include <cstdint>

namespace skippy {

/// Rollback snapshot count, inside the loader's [2, 64] window by construction.
///
/// `from_planned` mirrors the clamp the patched loader applies
/// (`min(max(planned, 2), 64)`, `model_loading.cpp`), so a depth outside the
/// window cannot cross this boundary, and a planned 0 keeps the native
/// default of 2. `SKIPPY_RS_SEQ` can still override either inside the same
/// window; that override stays native.
class RollbackTokens {
public:
    constexpr RollbackTokens() noexcept = default;

    /// Clamps `planned` into the window exactly as the native loader does.
    [[nodiscard]] static constexpr RollbackTokens from_planned(std::uint32_t planned) noexcept {
        return RollbackTokens{std::min(std::max(planned, kLeast), kMost)};
    }

    [[nodiscard]] constexpr std::uint32_t value() const noexcept { return value_; }

    [[nodiscard]] friend constexpr bool operator==(const RollbackTokens& lhs,
                                                   const RollbackTokens& rhs) noexcept {
        return lhs.value_ == rhs.value_;
    }

private:
    explicit constexpr RollbackTokens(std::uint32_t value) noexcept : value_{value} {}

    static constexpr std::uint32_t kLeast = 2;
    static constexpr std::uint32_t kMost = 64;

    std::uint32_t value_ = kLeast;
};

} // namespace skippy

#endif // SKIPPY_API_RECURRENT_ROLLBACK_H
