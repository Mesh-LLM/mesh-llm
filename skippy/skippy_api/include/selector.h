#pragma once

// Chooses which native lookup an embedding comes from, by pooling mode, at
// compile time. Pooled modes produce one output per sequence and are looked up
// by SequenceId; Pooling::None produces one output per token and is looked up
// by TokenIndex. The strong id types make the two impossible to confuse.
//
// Nothing here depends on llama.cpp.

#include "pooling.h"
#include "types.h"

#include <concepts>
#include <utility>

namespace skippy::detail {

template <Pooling P> struct EmbeddingSelector {
    static_assert(P == Pooling::Mean || P == Pooling::Cls || P == Pooling::Last,
                  "Pooling::None has its own specialisation");

    template <std::invocable<SequenceId> SequenceLookup>
        requires std::same_as<std::invoke_result_t<SequenceLookup, SequenceId>,
                              NativeEmbeddingResult>
    [[nodiscard]] static constexpr NativeEmbeddingResult select(SequenceId sequence,
                                                                SequenceLookup&& lookup) {
        return std::forward<SequenceLookup>(lookup)(sequence);
    }
};

template <> struct EmbeddingSelector<Pooling::None> {
    template <std::invocable<TokenIndex> TokenLookup>
        requires std::same_as<std::invoke_result_t<TokenLookup, TokenIndex>, NativeEmbeddingResult>
    [[nodiscard]] static constexpr NativeEmbeddingResult select(TokenIndex token,
                                                                TokenLookup&& lookup) {
        return std::forward<TokenLookup>(lookup)(token);
    }
};

} // namespace skippy::detail
