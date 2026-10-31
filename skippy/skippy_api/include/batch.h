#pragma once

// Fills a native token batch for an embedding pass. The batch type is a
// concept rather than llama_batch so this layer, and its tests, never see
// llama.h; llama_batch satisfies TokenBatch field for field. The backend
// allocates (llama_batch_init) and this fills.
//
// Nothing here depends on llama.cpp.

#include "types.h"

#include <concepts>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>

namespace skippy::detail {

template <typename B>
concept TokenBatch = requires(B& batch, std::int32_t i) {
    batch.n_tokens = std::int32_t{};
    batch.token[i] = std::int32_t{};
    batch.pos[i] = std::int32_t{};
    batch.n_seq_id[i] = std::int32_t{};
    batch.seq_id[i][0] = std::int32_t{};
    batch.logits[i] = std::int8_t{};
};

inline constexpr std::size_t max_batch_tokens =
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());

template <TokenBatch Batch>
constexpr void fill_embedding_batch(Batch& batch, std::span<const Token> tokens,
                                    SequenceId sequence) noexcept {
    const auto count = static_cast<std::int32_t>(tokens.size());
    batch.n_tokens = count;
    for (const auto i : std::views::iota(std::int32_t{0}, count)) {
        batch.token[i] = static_cast<std::int32_t>(tokens[static_cast<std::size_t>(i)]);
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = static_cast<std::int32_t>(sequence);
        batch.logits[i] = 1;
    }
}

// Fills one sequence per input: each starts at position 0 and owns its own
// sequence id, and only each sequence's last token carries logits — the
// pooled output for every sequence is what the caller reads back. Returns the
// batch's total token count.
template <TokenBatch Batch>
constexpr std::int32_t
fill_embedding_batches(Batch& batch, std::span<const std::span<const Token>> inputs) noexcept {
    std::int32_t at = 0;
    for (const auto sequence : std::views::iota(std::size_t{0}, inputs.size())) {
        const auto length = static_cast<std::int32_t>(inputs[sequence].size());
        for (const auto i : std::views::iota(std::int32_t{0}, length)) {
            batch.token[at] =
                static_cast<std::int32_t>(inputs[sequence][static_cast<std::size_t>(i)]);
            batch.pos[at] = i;
            batch.n_seq_id[at] = 1;
            batch.seq_id[at][0] = static_cast<std::int32_t>(sequence);
            // Every token keeps its output: mean/CLS pooling is computed over
            // the sequence, not over the last token alone.
            batch.logits[at] = 1;
            ++at;
        }
    }
    batch.n_tokens = at;
    return at;
}

} // namespace skippy::detail
