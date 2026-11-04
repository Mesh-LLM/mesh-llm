#pragma once

// llama.cpp-facing batch ownership. This header, and everything under
// include/llama/ and src/llama/, is the only code that includes llama.h.
// It is not covered by the C++ tests (rule: no llama in C++ tests); the Rust
// integration tests exercise it against a real model.

#include "types.h"
#include "workload_error.h"

#include "llama.h"

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <type_traits>

namespace skippy::llama {

// llama_batch_init() hands back the struct by value with heap-owned arrays;
// the owner here is heap-allocated too so the unique_ptr can hold it, and the
// deleter frees both. See allocate_embedding_batch for why the owner comes first.
inline constexpr auto embedding_batch_deleter = [](llama_batch* batch) noexcept {
    llama_batch_free(*batch);
    delete batch;
};

using EmbeddingBatchPtr =
    std::unique_ptr<llama_batch, std::remove_cv_t<decltype(embedding_batch_deleter)>>;

// Allocates an empty batch with room for `capacity` tokens across `sequences`
// sequences. Fails with AllocationFailed. Capacity is the caller's to validate.
[[nodiscard]] std::expected<EmbeddingBatchPtr, WorkloadFailure>
allocate_embedding_batch(std::int32_t capacity, std::int32_t sequences = 1);

// Allocates a batch sized for `tokens` and fills it as one sequence. Fails
// with EmptyInput, TooManyTokens (more than int32 can count) or
// AllocationFailed. Capacity against the context's n_batch is the session's
// check, since only it knows the context.
[[nodiscard]] std::expected<EmbeddingBatchPtr, WorkloadFailure>
make_embedding_batch(std::span<const Token> tokens, SequenceId sequence);

} // namespace skippy::llama