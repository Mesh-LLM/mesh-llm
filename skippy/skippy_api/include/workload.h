#pragma once

#include "pooling.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace skippy {

enum class WorkloadKind : std::int8_t {
    CausalGeneration,
    Embedding,
    Rerank,
    EncoderDecoder,
};

enum class NativePooling : std::int8_t {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
};

[[nodiscard]] constexpr std::optional<Pooling> to_pooling(NativePooling native) noexcept {
    switch (native) {
    case NativePooling::None: return Pooling::None;
    case NativePooling::Mean: return Pooling::Mean;
    case NativePooling::Cls: return Pooling::Cls;
    case NativePooling::Last: return Pooling::Last;
    case NativePooling::Unspecified:
    case NativePooling::Rank: return std::nullopt;
    }
    return std::nullopt; // a value from a newer native library
}

// ABI compatible as well
struct WorkloadProbe {
    NativePooling pooling = NativePooling::Unspecified;
    bool has_encoder = false;
    bool has_decoder = false;
    std::size_t embedding_dimensions = 0;
    std::size_t classifier_outputs = 0;
};

// Matches the original ABI
[[nodiscard]] constexpr WorkloadKind classify_workload(const WorkloadProbe& probe) noexcept {
    if (probe.classifier_outputs > 0 || probe.pooling == NativePooling::Rank) {
        return WorkloadKind::Rerank;
    }
    if (probe.has_encoder && probe.has_decoder) {
        return WorkloadKind::EncoderDecoder;
    }
    const bool pooled =
        probe.pooling != NativePooling::None && probe.pooling != NativePooling::Unspecified;
    if (probe.has_encoder || pooled) {
        return WorkloadKind::Embedding;
    }
    return WorkloadKind::CausalGeneration;
}

// should match the existing ABI
struct WorkloadInfo {
    WorkloadKind kind;
    NativePooling pooling;
    std::optional<Pooling> embedding_pooling;
    std::size_t output_dimensions;
    std::size_t classifier_outputs;
    bool has_encoder;
    bool has_decoder;
};

[[nodiscard]] constexpr WorkloadInfo make_workload_info(const WorkloadProbe& probe) noexcept {
    const WorkloadKind kind = classify_workload(probe);
    return WorkloadInfo{
        .kind = kind,
        .pooling = probe.pooling,
        .embedding_pooling =
            kind == WorkloadKind::Embedding ? to_pooling(probe.pooling) : std::nullopt,
        .output_dimensions = kind == WorkloadKind::Embedding ? probe.embedding_dimensions : 0,
        .classifier_outputs = kind == WorkloadKind::Rerank ? probe.classifier_outputs : 0,
        .has_encoder = probe.has_encoder,
        .has_decoder = probe.has_decoder,
    };
}

} // namespace skippy
