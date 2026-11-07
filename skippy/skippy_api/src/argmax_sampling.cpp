#include "argmax_sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace skippy {
namespace {

/// Mirrors the native per-sampler preserves-argmax rule exactly; a sampler
/// kind outside the domain preserves nothing.
[[nodiscard]] bool sampler_preserves_argmax(SamplerKind kind, const ArgmaxEligibility& knobs,
                                            bool default_order) noexcept {
    switch (kind) {
    case SamplerKind::penalties: {
        // The chain maps a zero window to -1, which disables the penalties
        // sampler; a zero repeat penalty maps to the neutral 1.0.
        const float repeat_penalty = knobs.repeat_penalty == 0.0F ? 1.0F : knobs.repeat_penalty;
        return knobs.penalty_last_n <= 0 ||
               (repeat_penalty == 1.0F && knobs.frequency_penalty == 0.0F &&
                knobs.presence_penalty == 0.0F);
    }
    case SamplerKind::dry:
        return knobs.dry_multiplier == 0.0F || knobs.dry_base < 1.0F ||
               knobs.dry_penalty_last_n <= 0;
    case SamplerKind::typical_p:
        // Typical sampling can drop the top token; the default order skips
        // it outside (0, 1).
        return knobs.typical_p >= 1.0F || (default_order && !(knobs.typical_p > 0.0F));
    case SamplerKind::xtc:
        // XTC removes top tokens unless it returns early.
        return knobs.xtc_probability <= 0.0F || knobs.xtc_threshold > 0.5F;
    case SamplerKind::top_n_sigma:
    case SamplerKind::top_k:
    case SamplerKind::top_p:
    case SamplerKind::min_p:
    case SamplerKind::temperature: return true;
    }
    return false;
}

} // namespace

bool selects_argmax(const ArgmaxEligibility& knobs, std::span<const SamplerKind> order) noexcept {
    if (!(knobs.temperature <= 0.0F) || knobs.dynatemp_range > 0.0F || knobs.mirostat_mode == 1 ||
        knobs.mirostat_mode == 2) {
        return false;
    }
    const std::span<const SamplerKind> chain =
        order.empty() ? std::span<const SamplerKind>{default_sampler_order} : order;
    bool has_temperature = false;
    for (const SamplerKind kind : chain) {
        if (!sampler_preserves_argmax(kind, knobs, order.empty())) {
            return false;
        }
        has_temperature = has_temperature || kind == SamplerKind::temperature;
    }
    // Without the temperature sampler the draw spans the distribution.
    return has_temperature;
}

ArgmaxPlan::ArgmaxPlan(const std::span<const LogitBias> biases, const std::int32_t n_vocab)
    : n_vocab_(n_vocab > 0 ? n_vocab : 0) {
    if (biases.empty() || n_vocab_ == 0) {
        return;
    }
    const auto vocab = static_cast<std::size_t>(n_vocab_);
    biased_.assign(vocab, 0);
    std::vector<std::int32_t> slot(vocab, -1);
    // Pass one groups repeated biases per token; pass two lays each token's
    // values contiguously in the arena, because interleaved pushes would
    // leave a token's values non-adjacent.
    for (const LogitBias& bias : biases) {
        if (bias.token < Token{0} || bias.token >= Token{n_vocab_}) {
            continue;
        }
        const auto token = static_cast<std::size_t>(bias.token);
        if (slot[token] < 0) {
            slot[token] = static_cast<std::int32_t>(tokens_.size());
            biased_[token] = 1;
            tokens_.push_back(Biased{.token = bias.token, .first = 0, .count = 0});
        }
        tokens_[static_cast<std::size_t>(slot[token])].count += 1;
    }
    std::uint32_t cursor = 0;
    for (Biased& entry : tokens_) {
        entry.first = cursor;
        cursor += entry.count;
    }
    values_.assign(cursor, 0.0F);
    std::vector<std::uint32_t> filled(tokens_.size(), 0);
    for (const LogitBias& bias : biases) {
        if (bias.token < Token{0} || bias.token >= Token{n_vocab_}) {
            continue;
        }
        const auto index = static_cast<std::size_t>(slot[static_cast<std::size_t>(bias.token)]);
        values_[tokens_[index].first + filled[index]] = bias.bias;
        filled[index] += 1;
    }
}

std::optional<Token> ArgmaxPlan::select_unique(const std::span<const float> logits) const noexcept {
    if (logits.empty() || n_vocab_ <= 0 || logits.size() < static_cast<std::size_t>(n_vocab_)) {
        return std::nullopt;
    }
    Token best_token{0};
    bool have_best = false;
    float best = -std::numeric_limits<float>::infinity();
    bool tied = false;
    bool has_nan = false;
    const auto consider = [&](const Token token, const float value) noexcept {
        if (std::isnan(value)) {
            has_nan = true;
        } else if (!have_best || value > best) {
            best = value;
            best_token = token;
            have_best = true;
            tied = false;
        } else if (value == best) {
            tied = true;
        }
    };
    const std::uint8_t* biased = biased_.empty() ? nullptr : biased_.data();
    const auto vocab = static_cast<std::size_t>(n_vocab_);
    for (std::size_t index = 0; index < vocab; ++index) {
        if (biased == nullptr || biased[index] == 0) {
            // Bounded against n_vocab_ just above, so the narrowing is safe.
            consider(Token{static_cast<std::int32_t>(index)}, logits[index]);
        }
    }
    for (const Biased& entry : tokens_) {
        const auto token = static_cast<std::size_t>(entry.token);
        float value = logits[token];
        for (std::uint32_t bias = 0; bias < entry.count; ++bias) {
            value += values_[entry.first + bias];
        }
        consider(entry.token, value);
    }
    if (has_nan || tied || !have_best || !std::isfinite(best)) {
        return std::nullopt;
    }
    return best_token;
}

} // namespace skippy
