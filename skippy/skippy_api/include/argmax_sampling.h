#pragma once

#include "types.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace skippy {

/// Sampler-chain step kinds, mirroring the native `skippy_sampler_type`
/// domain (patch 0014) so an eligible chain reads identically on both sides.
enum class SamplerKind : std::int32_t {
    penalties = 1,
    dry = 2,
    top_n_sigma = 3,
    top_k = 4,
    typical_p = 5,
    top_p = 6,
    min_p = 7,
    xtc = 8,
    temperature = 9,
};

/// The chain the native side runs when the caller configures none; the order
/// matters because it decides which samplers see which distribution.
inline constexpr std::array<SamplerKind, 9> default_sampler_order{
    SamplerKind::penalties, SamplerKind::dry,       SamplerKind::top_n_sigma,
    SamplerKind::top_k,     SamplerKind::typical_p, SamplerKind::top_p,
    SamplerKind::min_p,     SamplerKind::xtc,       SamplerKind::temperature,
};

/// The knobs `selects_argmax` reads, as a projection of the caller's chain
/// configuration. The zero value is the temperature-zero chain with nothing
/// else configured: eligible. Every field mirrors one native read.
struct ArgmaxEligibility {
    float temperature = 0.0F;
    float dynatemp_range = 0.0F;
    std::int32_t mirostat_mode = 0;
    std::int32_t penalty_last_n = 0;
    float repeat_penalty = 0.0F;
    float frequency_penalty = 0.0F;
    float presence_penalty = 0.0F;
    float dry_multiplier = 0.0F;
    float dry_base = 0.0F;
    std::int32_t dry_penalty_last_n = 0;
    float typical_p = 0.0F;
    float xtc_probability = 0.0F;
    float xtc_threshold = 0.0F;
};

/// True when the chain degenerates to argmax: temperature <= 0, no dynatemp
/// or mirostat, every configured sampler preserves the maximum, and the
/// temperature sampler runs (without it the draw spans the distribution).
/// An empty `order` means the default chain.
[[nodiscard]] bool selects_argmax(const ArgmaxEligibility& knobs,
                                  std::span<const SamplerKind> order = {}) noexcept;

/// One logit-bias entry: `bias` accumulates onto `token`'s logit.
struct LogitBias {
    Token token;
    float bias;
};

/// Immutable per-request bias index for the argmax fast path. Biases apply
/// in sequence onto the original logit — never pre-summed, because float
/// addition is order-sensitive and the reference chain applies them in order.
/// Biases outside `[0, n_vocab)` are skipped, mirroring the native chain.
class ArgmaxPlan {
public:
    ArgmaxPlan() = default;

    /// Builds the index for a vocabulary of `n_vocab` tokens. The membership
    /// mask is one byte per token, so `n_vocab` must be the model's real
    /// vocabulary size, never a file-supplied or caller-inflated count.
    explicit ArgmaxPlan(std::span<const LogitBias> biases, std::int32_t n_vocab);

    /// The unique biased argmax, or nullopt when the maximum is tied, NaN,
    /// non-finite, or the logits span is shorter than the vocabulary (the
    /// native `LLAMA_TOKEN_NULL` cases).
    [[nodiscard]] std::optional<Token> select_unique(std::span<const float> logits) const noexcept;

    [[nodiscard]] bool empty() const noexcept { return tokens_.empty(); }

private:
    /// Position of one biased token's values in the flat `values_` arena;
    /// positions, never pointers, so the arena can never dangle.
    struct Biased {
        Token token;
        std::uint32_t first;
        std::uint32_t count;
    };

    std::vector<std::uint8_t> biased_;
    std::vector<Biased> tokens_;
    std::vector<float> values_;
    std::int32_t n_vocab_ = 0;
};

} // namespace skippy
