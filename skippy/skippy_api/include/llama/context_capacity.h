// T2 (docs/TYPE_DESIGN.md): a capacity computed from the resource, never
// supplied by the caller. Constructible from a const llama_context& and
// nothing else, so no parameter can carry a wrong label (P0-A).
//
// llama pads n_ctx to a 256 multiple but keeps the requested n_batch and
// n_ubatch (llama-context.cpp:332-334, 377-390); llama_decode GGML_ASSERTs
// when tokens exceed n_batch/n_ubatch (llama-context.cpp:1612, 1896-1898).
// The min of the three, divided by the sequence count, is the only honest label.
#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "llama.h"
#include "skippy/geometry.h"

namespace skippy::llama {

class ContextCapacity {
public:
    explicit ContextCapacity(const llama_context& context) noexcept
        : sequences_{std::max(1U, llama_n_seq_max(&context))},
          tokens_{
              std::min({llama_n_ctx(&context), llama_n_batch(&context), llama_n_ubatch(&context)}) /
                  sequences_.value,
          } {}

    [[nodiscard]] constexpr bool serves(const SessionGeometry& geometry) const noexcept {
        return tokens_ >= geometry.tokens() && sequences_ >= geometry.sequences();
    }
    [[nodiscard]] constexpr TokensPerSequence tokens() const noexcept { return tokens_; }
    [[nodiscard]] constexpr SequenceCount sequences() const noexcept { return sequences_; }

private:
    SequenceCount sequences_;
    TokensPerSequence tokens_;
};

// Pins (docs/TYPE_DESIGN.md T2): a capacity cannot be spelled from numbers.
static_assert(
    !std::is_constructible_v<skippy::llama::ContextCapacity, std::uint32_t, std::uint32_t>);
static_assert(!std::is_default_constructible_v<skippy::llama::ContextCapacity>);

} // namespace skippy::llama
