#pragma once

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
    NoOutput,                   // decode succeeded but produced no output
    DimensionMismatch,          // native output length differs from advertised width
};

} // namespace skippy
