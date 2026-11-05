#ifndef SKIPPY_API_STAGE_CHAIN_H
#define SKIPPY_API_STAGE_CHAIN_H

/// The plugin contract for chained execution.
///
/// A chain produces one payload: the activation frame crossing into its terminal
/// stage. What that payload means — an embedding to pool, logits to sample, bytes
/// to forward — is the caller's decision, so it arrives as a compile-time sink.
/// The library never guesses, and never links a sampler.

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace skippy {

/// Bytes of one activation boundary, borrowed for the duration of `consume`.
using FrameBytes = std::span<const std::byte>;

/// A consumer for the terminal frame.
///
/// Failure is a value, never a sentinel: a sink that cannot accept the frame
/// returns the reason.
template <typename Sink>
concept StageSink = requires(Sink& sink, FrameBytes frame) {
    { sink.consume(frame) } -> std::same_as<std::optional<std::string>>;
};

/// Advances the chain and keeps nothing. Useful for load tests and warmup.
struct DiscardSink {
    [[nodiscard]] std::optional<std::string> consume(FrameBytes frame) const {
        static_cast<void>(frame);
        return std::nullopt;
    }
};

/// Keeps its own copy of the terminal frame so the caller can pool it.
struct CopyFrameSink {
    std::optional<std::vector<std::byte>> frame;

    [[nodiscard]] std::optional<std::string> consume(FrameBytes source) {
        frame = std::vector<std::byte>{source.begin(), source.end()};
        return std::nullopt;
    }
};

} // namespace skippy

#endif // SKIPPY_API_STAGE_CHAIN_H
