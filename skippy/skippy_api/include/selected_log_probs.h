#pragma once

// Selected log probabilities from one full-vocabulary logits row: the pure
// half of the causal readout capability (llama.cpp patch 0011). Computed in
// double precision over the unmodified vocabulary distribution — before
// temperature, penalties or grammar. Masked (-infinity) logits are valid row
// entries, but a selected masked token, a NaN, +infinity or an entirely
// masked vocabulary is not usable decision evidence and is rejected.
//
// Never publishes a partial result: `output` is untouched on rejection.
// No dependency on any native library.

#include <cstddef>
#include <span>

#include "types.h"

namespace skippy::detail {

// Selected output stays bounded even when the vocabulary is very large
// (patch 0011: SKIPPY_MAX_SELECTED_TOKENS).
inline constexpr std::size_t max_selected_tokens = 256;

// Writes log P(token) for each token in `selected`, read from `logits`,
// into `output`. `selected.size()` entries are written on success;
// `output.size()` must be at least `selected.size()`. Rejections: null or
// empty inputs, `selected.size() > max_selected_tokens`, a token outside the
// vocabulary, a non-finite or NaN row entry that the scores cannot trust, an
// entirely masked row, or a selected token whose score is not finite.
bool selected_log_probabilities(std::span<const float> logits, std::span<const Token> selected,
                                std::span<float> output) noexcept;

} // namespace skippy::detail
