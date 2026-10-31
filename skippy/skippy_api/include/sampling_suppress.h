#pragma once

/// Vocab-declared token suppression, ported from llama.cpp patch 0012
/// (skippy: honor vocab suppress tokens in sampling).
///
/// A GGUF may declare `tokenizer.ggml.suppress_tokens`: ids the model
/// never wants sampled, whatever the caller asks for. The native patch
/// teaches the runtime's samplers to honor that list — suppression
/// overrides caller logit biases, greedy selection skips suppressed
/// tokens, and exhausting the vocabulary is representable.
///
/// This library never links a sampler (see stage_sink.h): it hands the
/// caller logits and the caller decides. What it can own is the *policy*:
/// the merge and selection rules above, as pure functions over caller
/// data, plus the model accessor that gathers the declared set once at
/// load. A caller composes the two and gets exactly the patch's behavior
/// without this library sampling anything.
///
/// EOG handling composes the same way: the native patch's `ignore_eog`
/// arm is "skip these ids too", so a caller that wants it appends the
/// model's EOG ids to the suppress list before merging.

#include "types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace skippy {

/// A caller's per-token logit bias, mirroring the native llama_logit_bias.
struct LogitBias {
    Token token;
    float bias;
};

/// True when `token` is in the declared suppress set. This is the cheap
/// single-candidate check the native patch's accelerated path performs.
[[nodiscard]] inline bool is_suppressed(Token token, std::span<const Token> suppress) noexcept {
    return std::ranges::find(suppress, token) != suppress.end();
}

/// Vocab suppression overrides caller biases: every caller entry on a
/// suppressed token is dropped, then one −∞ entry per suppressed token is
/// appended, in declaration order. A model without suppress tokens (every
/// fixture in this suite) leaves the caller's biases untouched.
[[nodiscard]] inline std::vector<LogitBias> apply_vocab_suppress(std::vector<LogitBias> biases,
                                                                 std::span<const Token> suppress) {
    if (suppress.empty()) return biases;
    const auto overridden = [suppress](Token token) { return is_suppressed(token, suppress); };
    std::erase_if(biases, [&overridden](const LogitBias& bias) { return overridden(bias.token); });
    biases.reserve(biases.size() + suppress.size());
    constexpr float blocked = -std::numeric_limits<float>::infinity();
    for (const Token token : suppress) {
        biases.push_back(LogitBias{.token = token, .bias = blocked});
    }
    return biases;
}

/// Greedy selection over the allowed logits: suppressed tokens are
/// skipped, the first token wins ties, and ids outside the vocabulary are
/// ignored rather than fatal. `nullopt` means every token was suppressed
/// (or there were no logits) — the native patch surfaces the same case as
/// LLAMA_TOKEN_NULL and lets the caller decide what exhaustion means.
[[nodiscard]] inline std::optional<Token> greedy_among_allowed(std::span<const float> logits,
                                                               std::span<const Token> suppress) {
    if (logits.empty()) return std::nullopt;
    std::vector<std::uint8_t> blocked(logits.size(), 0);
    for (const Token token : suppress) {
        const auto id = std::to_underlying(token);
        if (id >= 0 && static_cast<std::uint64_t>(id) < logits.size()) {
            blocked[static_cast<std::size_t>(id)] = 1;
        }
    }
    std::optional<Token> best;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (blocked[index] != 0) continue;
        if (!best.has_value() || logits[index] > best_logit) {
            best_logit = logits[index];
            best = Token{static_cast<std::int32_t>(index)};
        }
    }
    return best;
}

/// Why a selection cannot become a sampled token. Ported from llama.cpp
/// patch 0013 (skippy: reject exhausted vocab sampling): exhausting the
/// vocabulary under suppression is a caller-visible failure, never a
/// sentinel id that leaks into caller outputs or subsequent decode inputs.
enum class SamplingFailure {
    /// Every candidate was suppressed, or there were no logits to pick from.
    ExhaustedVocabulary,
    /// A raw id below zero arrived where a token was required — the native
    /// sampler's "nothing" spelling, whatever sentinel produced it.
    SentinelToken,
};

/// Patch 0013's boundary rule for callers of `greedy_among_allowed`: the
/// selection is either a usable token or the typed reason it is not. An
/// engaged selection always passes — id 0 is a real token, not a sentinel —
/// and an exhausted one is a rejection the caller stops on, exactly as the
/// native patch turns `LLAMA_TOKEN_NULL` into a runtime error at every
/// sampling boundary instead of storing it.
[[nodiscard]] constexpr std::expected<Token, SamplingFailure>
eligible_token(std::optional<Token> selection) noexcept {
    if (!selection.has_value()) return std::unexpected(SamplingFailure::ExhaustedVocabulary);
    return *selection;
}

/// Patch 0014's store rule for ids that arrive as raw native tokens: any
/// negative id — `LLAMA_TOKEN_NULL` today, whatever sentinel a future
/// sampler returns — is a rejection, and id 0 is a token. The native helper
/// this mirrors also guarantees a rejected store never touches the caller's
/// destination; here that property is structural, because the only way to
/// write the result is through a successful `expected`.
[[nodiscard]] constexpr std::expected<Token, SamplingFailure>
eligible_token(std::int32_t native_id) noexcept {
    if (native_id < 0) return std::unexpected(SamplingFailure::SentinelToken);
    return Token{native_id};
}

/// Patch 0014's batch boundary: a sampled batch is published only when every
/// selection is a token. One exhausted selection rejects the whole batch —
/// the native iteration paths zero their output count and publish nothing
/// partial, and so does this rule.
[[nodiscard]] inline std::expected<std::vector<Token>, SamplingFailure>
eligible_tokens(std::span<const std::optional<Token>> sampled) {
    std::vector<Token> tokens;
    tokens.reserve(sampled.size());
    for (const std::optional<Token>& selection : sampled) {
        if (!selection.has_value()) {
            return std::unexpected(SamplingFailure::ExhaustedVocabulary);
        }
        tokens.push_back(*selection);
    }
    return tokens;
}

} // namespace skippy
