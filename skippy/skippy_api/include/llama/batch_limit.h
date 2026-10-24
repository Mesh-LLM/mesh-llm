// T3 (docs/TYPE_DESIGN.md): admission is a type. batch_limit_ was a field
// nobody had to read; the check now produces the only thing run_pass consumes,
// so a decode cannot be issued without the limit check that guards the
// GGML_ASSERT in llama_decode.
#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <type_traits>
#include <utility>

#include "llama.h"
#include "skippy/geometry.h"
#include "types.h"
#include "workload_error.h"

namespace skippy::llama {

class AdmittedBatch {
public:
    [[nodiscard]] std::span<const Token> tokens() const noexcept { return tokens_; }

private:
    friend class BatchLimit;
    explicit AdmittedBatch(std::span<const Token> tokens) noexcept : tokens_(tokens) {}
    std::span<const Token> tokens_;
};

class BatchLimit {
public:
    explicit BatchLimit(const llama_context& context) noexcept
        : ubatch_{llama_n_ubatch(&context)} {}

    // Both bounds in one place: the session's contract and the context's physical batch.
    [[nodiscard]] std::expected<AdmittedBatch, WorkloadError>
    admit(std::span<const Token> tokens, TokensPerSequence configured) const noexcept {
        if (tokens.empty()) return std::unexpected(WorkloadError::EmptyInput);
        if (tokens.size() > configured.value || tokens.size() > ubatch_)
            return std::unexpected(WorkloadError::TooManyTokens);
        return AdmittedBatch{tokens};
    }

    // The batch path: the total across sequences, against the same ubatch.
    [[nodiscard]] std::expected<AdmittedBatch, WorkloadError>
    admit_total(std::span<const Token> tokens) const noexcept {
        if (tokens.empty()) return std::unexpected(WorkloadError::EmptyInput);
        if (tokens.size() > ubatch_) return std::unexpected(WorkloadError::TooManyTokens);
        return AdmittedBatch{tokens};
    }

private:
    std::uint32_t ubatch_;
};

// Pins (docs/TYPE_DESIGN.md T3): admission is produced only by the check.
static_assert(
    !std::is_constructible_v<skippy::llama::AdmittedBatch, std::span<const skippy::Token>>);
static_assert(!std::is_default_constructible_v<skippy::llama::AdmittedBatch>);

} // namespace skippy::llama
