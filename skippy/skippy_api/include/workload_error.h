#pragma once

#include <string>
#include <utility>

namespace skippy {

// One error type for every non-chat workload (embed, rerank, encode), as the
// original ABI had one status code. Shared by the generic functions and the
// native backend so a failure means the same thing wherever it is raised.
// Ordered roughly by where in a pipeline it is detected.
enum class WorkloadError {
    InvalidSession,             // handle does not resolve (moved-from, released, foreign)
    UnsupportedWorkload,        // the model behind the session is not for this workload
    UnsupportedClassifierCount, // rerank needs exactly one classifier output
    EmptyInput,                 // zero tokens
    TooManyTokens,              // exceeds the session's batch capacity
    TooManySequences,           // more inputs than the session was created for
    TokenizationFailed,         // text could not be tokenized, or produced nothing
    NoEmbeddingDimensions,      // model advertises an output width of zero
    OutputTooSmall,             // caller's buffer is shorter than the output width
    AllocationFailed,           // batch or scratch allocation
    DecodeFailed,               // native decode reported failure
    EncodeFailed,               // native encode reported failure
    NoDecoderStartToken,        // encoder-decoder model has neither a decoder start token nor BOS
    NoOutput,                   // decode succeeded but produces no output
    DimensionMismatch,          // native output length differs from advertised width
};

// A workload error plus the backend's explanation when there is one. A native
// pass failure cannot be constructed without its message: two failure paths
// parked the reason in a member nobody read, so the message travels in the
// value (AGENTS, Traps; LoadFailure is the load-side spelling).
struct WorkloadFailure {
    WorkloadError code = WorkloadError::DecodeFailed;
    std::string message;

    /// Every non-native code: raised by the facade itself, where no backend
    /// text exists to carry.
    [[nodiscard]] static constexpr WorkloadFailure code_only(WorkloadError error) noexcept {
        return WorkloadFailure{.code = error, .message = {}};
    }

    /// A native pass failure with the backend's own text. Only the two native
    /// codes have a backend behind them; the clause makes any other code
    /// uncompilable (negative tests in tests/type_contracts_test.cpp).
    template <WorkloadError Code>
        requires(Code == WorkloadError::DecodeFailed || Code == WorkloadError::EncodeFailed)
    [[nodiscard]] static WorkloadFailure native(std::string text) {
        return WorkloadFailure{.code = Code, .message = std::move(text)};
    }
};

} // namespace skippy
