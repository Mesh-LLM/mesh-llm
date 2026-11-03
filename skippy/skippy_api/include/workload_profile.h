#pragma once

/// Pure workload classification, ported from llama.cpp patch 0011
/// (skippy: configure non-chat model loading).
///
/// The native patch derives three booleans inside `skippy_finish_model_open`
/// and uses them to configure the context: embeddings on or off, RANK pooling
/// for classifier heads, and one serialized lane for encoder-decoders. Those
/// decisions belong to the model's shape, not to the loader, so they live here
/// as a decision table over observable model facts — testable without llama,
/// and consumed by the native session configuration.
///
/// The lane rule has no façade equivalent: sessions own private contexts, so
/// encoder-decoder work is isolated by construction rather than serialized.
/// `lanes_must_serialize` is still reported so a caller that does share state
/// knows the constraint exists.

#include <cstdint>

namespace skippy {

/// The pooling a GGUF declares, constrained at the boundary. Anything outside
/// the known values is `Unspecified` and falls through to the decision table.
enum class DeclaredPooling : std::uint8_t {
    None,
    Mean,
    Cls,
    Last,
    Rank,
    Unspecified,
};

/// Observable facts about a loaded model. Gathered once at load; nothing here
/// calls into llama.
struct ModelFacts {
    bool has_encoder = false;
    bool has_decoder = false;
    /// Encoder-side classifier outputs. 0 for decoder-family models:
    /// the native n_cls_out getter defaults to 1, so it is not a presence
    /// test and load_model gates it on the architecture family before
    /// feeding the table.
    std::uint32_t classifier_outputs = 0;
    DeclaredPooling declared_pooling = DeclaredPooling::Unspecified;
};

/// How a model's workloads must be configured.
struct WorkloadProfile {
    /// The model produces pooled or per-sequence embeddings: the context must
    /// carry them. Encoder-only models, classifier heads and any declared
    /// pooling all imply this; a bare causal decoder does not.
    bool embedding_workload = false;

    /// A classifier head with no declared pooling reads as a reranker: the
    /// context needs RANK pooling to expose its single score.
    bool classifier_workload = false;

    /// Both halves present: encoder output is context-global in llama.cpp, so
    /// a shared context must serialize this model's work.
    bool encoder_decoder = false;

    /// The serialization constraint above, stated for callers that share lanes.
    /// The façade never does; per-session contexts make it moot.
    bool lanes_must_serialize = false;
};

/// The decision table, mirroring patch 0011's semantics exactly:
/// - encoder-decoders serialize,
/// - a classifier head with unspecified/none pooling is a reranker,
/// - anything encoder-backed, classifier-headed, or with a declared pooling
///   (other than none) is an embedding workload — unless it is an
///   encoder-decoder, whose embeddings are consumed by the decode half.
[[nodiscard]] constexpr WorkloadProfile classify(ModelFacts facts) noexcept {
    const bool encoder_decoder = facts.has_encoder && facts.has_decoder;
    const bool classifier_workload =
        facts.classifier_outputs > 0 && (facts.declared_pooling == DeclaredPooling::None ||
                                         facts.declared_pooling == DeclaredPooling::Unspecified);
    const bool pooled = facts.declared_pooling != DeclaredPooling::None &&
                        facts.declared_pooling != DeclaredPooling::Unspecified;
    const bool embedding_workload =
        !encoder_decoder && (classifier_workload || facts.has_encoder || pooled);
    return WorkloadProfile{
        .embedding_workload = embedding_workload,
        .classifier_workload = classifier_workload,
        .encoder_decoder = encoder_decoder,
        .lanes_must_serialize = encoder_decoder,
    };
}

} // namespace skippy
