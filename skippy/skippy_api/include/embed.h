#pragma once

// The embedding workload, generic over the runtime's backend.
//
// Layers, top down:
//   embed<Format>()           this file: validation, normalisation, Format view
//   Backend::SessionResource  native decode; hands back a borrowed f32 span
//
// The native span never escapes embed(): it is consumed into the caller's
// buffer before returning, and the EmbeddingView returned borrows that buffer.
// Nothing here depends on llama.cpp.

#include "normalize.h"
#include "runtime.h"
#include "types.h"
#include "workload_error.h"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <expected>
#include <functional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

namespace skippy {

// What a backend's session must provide for embed() to compile against it.
template <typename S> using EmbedResult = std::expected<detail::NativeEmbeddingView, WorkloadError>;

template <typename S>
concept EmbeddingSession =
    requires(S& session, const S& const_session, std::span<const Token> tokens,
             std::span<const std::span<const Token>> inputs) {
        // Output width the model advertises; zero means the model cannot embed.
        { const_session.embedding_dimensions() } -> std::convertible_to<std::size_t>;
        // Runs the native decode and returns the pooled (or selected) output,
        // borrowed from the native context until the next decode.
        { session.embed(tokens) } -> std::same_as<EmbedResult<S>>;
        // Decodes several sequences in one native pass; rows are contiguous.
        { session.embed_many(inputs) } -> std::same_as<EmbedResult<S>>;
    };

// An output format: what each element is stored as, plus any per-call
// metadata the caller wants carried alongside the values.
template <typename F>
concept EmbeddingFormat = requires {
    typename F::storage_type;
    typename F::metadata_type;
} && std::is_object_v<typename F::storage_type>;

struct F32Format {
    using storage_type = float;
    struct metadata_type {}; // nothing yet; widen when a caller needs it
};

// Caller-owned destination. `values` must hold at least the model's width.
template <EmbeddingFormat Format> struct EmbeddingOutput {
    std::span<typename Format::storage_type> values;
    std::span<typename Format::metadata_type> metadata;
};

// Read-only view over the caller's buffer, trimmed to the model's width.
template <EmbeddingFormat Format> struct EmbeddingView {
    std::span<const typename Format::storage_type> values;
    std::span<const typename Format::metadata_type> metadata;
    std::size_t dimensions;
};

// Read-only view over the caller's batch buffer: `count` rows of `dimensions`.
template <EmbeddingFormat Format> struct EmbeddingBatchView {
    std::span<const typename Format::storage_type> values;
    std::span<const typename Format::metadata_type> metadata;
    std::size_t count;
    std::size_t dimensions;
};

// Computes one normalised embedding for `tokens` into `output` and returns a
// view of it. Checks that need no native call run first, so a bad request
// never reaches the backend. Currently f32 only; other storage types will
// normalise into scratch and convert.
template <EmbeddingFormat Format = F32Format, RuntimeBackend Backend>
    requires EmbeddingSession<typename Backend::SessionResource> &&
             std::same_as<typename Format::storage_type, float>
[[nodiscard]] std::expected<EmbeddingView<Format>, WorkloadError>
embed(NativeRuntime<Backend>& runtime, const Session& handle, std::span<const Token> tokens,
      EmbeddingOutput<Format> output) {
    using View = EmbeddingView<Format>;
    using Result = std::expected<View, WorkloadError>;

    if (tokens.empty()) {
        return std::unexpected(WorkloadError::EmptyInput);
    }

    const auto run = [&](Backend::SessionResource& session) -> Result {
        const std::size_t dimensions = std::as_const(session).embedding_dimensions();
        if (dimensions == 0) {
            return std::unexpected(WorkloadError::NoEmbeddingDimensions);
        }
        if (output.values.size() < dimensions) {
            return std::unexpected(WorkloadError::OutputTooSmall);
        }

        return session.embed(tokens).and_then([&](detail::NativeEmbeddingView native) -> Result {
            if (native.size() != dimensions) {
                return std::unexpected(WorkloadError::DimensionMismatch);
            }
            detail::normalize_embedding(native.data(), output.values.data(), dimensions);
            return View{output.values.first(dimensions), output.metadata, dimensions};
        });
    };

    // with_session() yields expected<Result, RuntimeError>; the only runtime
    // error it can produce is an unresolvable handle. Map it, then flatten.
    return runtime.with_session(handle, run)
        .transform_error([](RuntimeError) { return WorkloadError::InvalidSession; })
        .and_then(std::identity{});
}

// Computes one normalised embedding per input, all decoded in a single native
// pass. `output.values` must hold `inputs.size() * dimensions` elements; rows
// are written back to back. Requires a session created with
// `SessionConfig::sequences >= inputs.size()`.
template <EmbeddingFormat Format = F32Format, RuntimeBackend Backend>
    requires EmbeddingSession<typename Backend::SessionResource> &&
             std::same_as<typename Format::storage_type, float>
[[nodiscard]] std::expected<EmbeddingBatchView<Format>, WorkloadError>
embed_batch(NativeRuntime<Backend>& runtime, const Session& handle,
            std::span<const std::span<const Token>> inputs, EmbeddingOutput<Format> output) {
    using View = EmbeddingBatchView<Format>;
    using Result = std::expected<View, WorkloadError>;

    if (inputs.empty()) {
        return std::unexpected(WorkloadError::EmptyInput);
    }
    if (std::ranges::any_of(inputs, [](const auto& tokens) { return tokens.empty(); })) {
        return std::unexpected(WorkloadError::EmptyInput);
    }

    const auto run = [&](Backend::SessionResource& session) -> Result {
        const std::size_t dimensions = std::as_const(session).embedding_dimensions();
        if (dimensions == 0) {
            return std::unexpected(WorkloadError::NoEmbeddingDimensions);
        }
        if (output.values.size() < (inputs.size() * dimensions)) {
            return std::unexpected(WorkloadError::OutputTooSmall);
        }

        return session.embed_many(inputs).and_then(
            [&](detail::NativeEmbeddingView native) -> Result {
                if (native.size() != (inputs.size() * dimensions)) {
                    return std::unexpected(WorkloadError::DimensionMismatch);
                }
                for (const std::size_t row : std::views::iota(std::size_t{0}, inputs.size())) {
                    detail::normalize_embedding(native.data() + (row * dimensions),
                                                output.values.data() + (row * dimensions),
                                                dimensions);
                }
                return View{
                    .values = output.values.first(inputs.size() * dimensions),
                    .metadata = output.metadata,
                    .count = inputs.size(),
                    .dimensions = dimensions,
                };
            });
    };

    return runtime.with_session(handle, run)
        .transform_error([](RuntimeError) { return WorkloadError::InvalidSession; })
        .and_then(std::identity{});
}

} // namespace skippy
