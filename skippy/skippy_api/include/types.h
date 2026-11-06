#pragma once

// Strong identifier types shared by the generic layer. Each is a scoped enum
// over the native int32 it stands in for: same size and alignment, so a
// buffer of them is layout-compatible with a Rust repr(transparent) newtype,
// but no arithmetic and no silent mixing of a token with a position or a
// sequence id. The llama backend converts at its boundary with static_cast.
//
// Nothing here depends on llama.cpp.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace skippy {

// A vocabulary token id.
enum class Token : std::int32_t {};

// A sequence within a native context; pooled embeddings are looked up by it.
enum class SequenceId : std::int32_t {};

// A position within a batch; per-token embeddings are looked up by it.
enum class TokenIndex : std::int32_t {};

static_assert(sizeof(Token) == sizeof(std::int32_t) && alignof(Token) == alignof(std::int32_t));
static_assert(sizeof(SequenceId) == sizeof(std::int32_t));
static_assert(sizeof(TokenIndex) == sizeof(std::int32_t));

namespace detail {

// Views a native int32 token buffer as Token without copying where the
// standard library can start the Token objects' lifetime in place; copies
// into `scratch` otherwise. The result borrows `native` or `scratch`
// accordingly, until either is next written.
#if defined(__cpp_lib_start_lifetime_as) && __cpp_lib_start_lifetime_as >= 202207L
inline constexpr bool token_view_is_zero_copy = true;
inline std::span<const Token> tokens_from_native(std::span<std::int32_t> native,
                                                 std::vector<Token>& /*scratch*/) noexcept {
    return {std::start_lifetime_as_array<Token>(native.data(), native.size()), native.size()};
}
#else
inline constexpr bool token_view_is_zero_copy = false;
inline std::span<const Token> tokens_from_native(std::span<const std::int32_t> native,
                                                 std::vector<Token>& scratch) {
    scratch.resize(native.size());
    std::ranges::transform(native, scratch.begin(), [](std::int32_t id) { return Token{id}; });
    return scratch;
}
#endif

// Borrowed output from the native layer, before normalisation or conversion.
// Valid until the next native decode on the same context.
using NativeEmbeddingView = std::span<const float>;

// A native lookup that may find nothing (the token or sequence produced no output).
using NativeEmbeddingResult = std::optional<NativeEmbeddingView>;

} // namespace detail
} // namespace skippy
