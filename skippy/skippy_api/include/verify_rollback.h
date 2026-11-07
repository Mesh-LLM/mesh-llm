#ifndef SKIPPY_API_VERIFY_ROLLBACK_H
#define SKIPPY_API_VERIFY_ROLLBACK_H

/**
 * @file verify_rollback.h
 * @brief How far a verify window may rewind recurrent state in place (patch 0015).
 *
 * A recurrent or hybrid context verifies an N-token speculative window in
 * place only while it keeps N rollback snapshots of the recurrent state per
 * sequence (native `n_rs_seq`); fewer snapshots and verify falls back to a
 * checkpoint restore. Each snapshot is another copy of the state that every
 * decode step writes, so a session that never verifies drafts reserves none.
 */

#include <cstdint>
#include <expected>
#include <optional>
#include <string>

namespace skippy {

/// Verify-window rewind policy: `none`, or up to `tokens` trailing tokens
/// rewound in place.
///
/// The two states are distinct, never a magic 0 at this boundary: `in_place`
/// rejects 0 outright, and `none` is its own constructor. Only `tokens()`,
/// which crosses into the native `skippy_runtime_config.verify_rollback_tokens`
/// field, encodes `none` as 0 — the same encoding the native ABI defines.
/// The decision belongs to the caller that knows the speculative plan; the
/// façade never derives it from model metadata.
class VerifyRollback {
public:
    /// The session never verifies drafts: no snapshots, verify windows
    /// restore a checkpoint instead.
    [[nodiscard]] static constexpr VerifyRollback none() noexcept { return VerifyRollback{0}; }

    /// Verify windows of up to `tokens` trailing tokens rewind in place.
    /// Zero is not a policy: it is rejected here, not silently mapped.
    [[nodiscard]] static std::expected<VerifyRollback, std::string> in_place(std::uint32_t tokens) {
        if (tokens == 0) {
            return std::unexpected(
                "in-place verify rollback needs at least one token; use VerifyRollback::none() "
                "when the session never verifies drafts");
        }
        return VerifyRollback{tokens};
    }

    /// Native `SKIPPY_VERIFY_ROLLBACK_TOKENS_DEFAULT`: the two snapshots a
    /// load that does not state a policy keeps (patch 0015's native default).
    [[nodiscard]] static constexpr VerifyRollback native_default() noexcept {
        return VerifyRollback{kNativeDefault};
    }

    /// The value for `skippy_runtime_config.verify_rollback_tokens`: 0 when
    /// `none`, the in-place window otherwise.
    [[nodiscard]] constexpr std::uint32_t tokens() const noexcept { return tokens_; }

    /// True when this is the no-snapshots policy.
    [[nodiscard]] constexpr bool is_none() const noexcept { return tokens_ == 0; }

    [[nodiscard]] friend constexpr bool operator==(const VerifyRollback& lhs,
                                                   const VerifyRollback& rhs) noexcept {
        return lhs.tokens_ == rhs.tokens_;
    }

private:
    explicit constexpr VerifyRollback(std::uint32_t tokens) noexcept : tokens_{tokens} {}
    static constexpr std::uint32_t kNativeDefault = 2;
    std::uint32_t tokens_;
};

/// Resolves the required staged-open policy to the native field value.
/// An omitted policy is a caller bug, rejected here at the pure boundary so
/// `StagedModel::open` never sees a spec without one; `none` is the native
/// 0 (no snapshots, checkpoint restore on rewind).
[[nodiscard]] inline std::expected<std::uint32_t, std::string>
verify_rollback_tokens_of(const std::optional<VerifyRollback>& policy) {
    if (!policy) {
        return std::unexpected(std::string{
            "verify_rollback is required: state VerifyRollback::none() when the session never "
            "verifies drafts, or VerifyRollback::in_place(n) for the window"});
    }
    return policy->tokens();
}

} // namespace skippy

#endif // SKIPPY_API_VERIFY_ROLLBACK_H
