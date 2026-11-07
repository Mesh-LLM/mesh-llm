#ifndef SKIPPY_API_LLAMA_CAUSAL_READOUT_H
#define SKIPPY_API_LLAMA_CAUSAL_READOUT_H

/// Causal token-score readouts: the façade half of llama.cpp patch 0011.
///
/// `read_selected_log_probs` feeds `tokens` through the session and reads
/// bounded, unsampled conditional log probabilities for `selected` from the
/// final row's full vocabulary distribution — before temperature, penalties
/// or grammar. The read advances positions and records tokens like any
/// iteration, but never samples and never changes sampler history. Only
/// terminal stages write scores; `supports_causal_readout` says whether the
/// loaded graph offers the capability at all.

#include "llama/staged.h"
#include "selected_log_probs.h"
#include "types.h"

#include <expected>
#include <span>
#include <string>

namespace skippy::llama {

/// Why a readout failed.
struct ReadoutFailure {
    /// Status the native call returned.
    int status = 0;
    /// The backend's own explanation, when it produced one.
    std::string message;
};

/// True when the model's graph supports causal next-token score reads
/// (decoder-only, causal attention, not a diffusion model). A model that
/// fails this never accepted a readout; gating on it avoids treating a
/// missing capability as a runtime error.
[[nodiscard]] bool supports_causal_readout(const StagedModel& model) noexcept;

/// Writes `selected.size()` log probabilities into `out_scores`.
/// `out_scores.size()` must be at least `selected.size()`, and the selection
/// must respect `skippy::detail::max_selected_tokens`. Fed tokens must fit
/// the session's own batch width, which this boundary reads from the
/// session — a caller cannot supply or bypass the bound. The backend
/// validates tokens against the vocabulary; rejections arrive as
/// `ReadoutFailure`.
[[nodiscard]] std::expected<void, ReadoutFailure>
read_selected_log_probs(StagedSession& session, std::span<const skippy::Token> tokens,
                        std::span<const skippy::Token> selected, std::span<float> out_scores);

} // namespace skippy::llama

#endif
