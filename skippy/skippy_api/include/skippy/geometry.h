// T2 (docs/TYPE_DESIGN.md): a session request proven by make().
// The product max_tokens * sequences is formed in one constructor, so a
// geometry that exceeds llama's int32 batch counts cannot exist.
#pragma once

#include <compare>
#include <cstdint>
#include <expected>
#include <utility>

#include "runtime.h"

namespace skippy {

struct TokensPerSequence {
    std::uint32_t value = 0;
    constexpr auto operator<=>(const TokensPerSequence&) const = default;
};

struct SequenceCount {
    std::uint32_t value = 0;
    constexpr auto operator<=>(const SequenceCount&) const = default;
};

class SessionGeometry {
public:
    static constexpr std::uint32_t kMaxSequences = 64;
    static constexpr std::uint32_t kMaxTotalTokens = 0x7FFF'FFFF; // llama batch counts are int32

    [[nodiscard]] static constexpr std::expected<SessionGeometry, RuntimeError>
    make(TokensPerSequence tokens, SequenceCount sequences) noexcept {
        if (tokens.value == 0 || sequences.value == 0 || sequences.value > kMaxSequences)
            return std::unexpected(RuntimeError::UnsupportedConfiguration);
        if (tokens.value > kMaxTotalTokens / sequences.value)
            return std::unexpected(RuntimeError::UnsupportedConfiguration);
        return SessionGeometry{tokens, sequences};
    }

    [[nodiscard]] constexpr TokensPerSequence tokens() const noexcept { return tokens_; }
    [[nodiscard]] constexpr SequenceCount sequences() const noexcept { return sequences_; }
    // proven by make(): the product cannot wrap or exceed kMaxTotalTokens
    [[nodiscard]] constexpr std::uint32_t total_tokens() const noexcept {
        return tokens_.value * sequences_.value;
    }

private:
    constexpr SessionGeometry(TokensPerSequence tokens, SequenceCount sequences) noexcept
        : tokens_(tokens), sequences_(sequences) {}
    TokensPerSequence tokens_;
    SequenceCount sequences_;
};

} // namespace skippy
