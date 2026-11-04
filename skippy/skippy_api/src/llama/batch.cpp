#include "llama/batch.h"

#include "batch.h"

#include <cstdint>
#include <new>

namespace skippy::llama {

static_assert(detail::TokenBatch<llama_batch>,
              "llama_batch no longer has the fields fill_embedding_batch writes");

std::expected<EmbeddingBatchPtr, WorkloadFailure> allocate_embedding_batch(std::int32_t capacity,
                                                                           std::int32_t sequences) {
    // Allocate the owner first, before llama allocates its arrays: if this
    // fails nothing needs freeing, and once it succeeds the deleter owns both.
    EmbeddingBatchPtr batch{new (std::nothrow) llama_batch{}};
    if (!batch) {
        return std::unexpected(WorkloadFailure::code_only(WorkloadError::AllocationFailed));
    }
    *batch = llama_batch_init(capacity, /*embd=*/0, sequences);
    return batch;
}

std::expected<EmbeddingBatchPtr, WorkloadFailure>
make_embedding_batch(std::span<const Token> tokens, SequenceId sequence) {
    if (tokens.empty()) {
        return std::unexpected(WorkloadFailure::code_only(WorkloadError::EmptyInput));
    }
    if (tokens.size() > detail::max_batch_tokens) {
        return std::unexpected(WorkloadFailure::code_only(WorkloadError::TooManyTokens));
    }
    return allocate_embedding_batch(static_cast<std::int32_t>(tokens.size()))
        .transform([&](EmbeddingBatchPtr batch) {
            detail::fill_embedding_batch(*batch, tokens, sequence);
            return batch;
        });
}

} // namespace skippy::llama