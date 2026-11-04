#ifndef SKIPPY_API_LLAMA_STAGE_CHAIN_H
#define SKIPPY_API_LLAMA_STAGE_CHAIN_H

/// A model run as N chained stages.
///
/// Each stage owns a model and a session; the payload of one stage is the input
/// of the next. What happens to the payload crossing into the terminal stage is
/// the caller's decision, expressed as a compile-time sink: the library never
/// guesses whether that means pool it, sample it, or drop it.

#include "llama/prepared_export.h"
#include "llama/staged.h"
#include "stage_plan.h"
#include "stage_sink.h"

#include "skippy.h"

#include <concepts>
#include <cstddef>
#include <expected>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace skippy::llama {

/// Owns N stages and drives them in order.
class StageChain {
public:
    StageChain() = default;
    StageChain(StageChain&&) noexcept = default;
    StageChain& operator=(StageChain&&) noexcept = default;
    StageChain(const StageChain&) = delete;
    StageChain& operator=(const StageChain&) = delete;

    /// Plans `ranges` over `path` and opens one model and session per stage.
    [[nodiscard]] static std::expected<StageChain, StagedFailure>
    open(const std::string& path, std::span<const LayerRange> ranges);

    /// Runs `tokens` through every stage, handing the terminal frame to `sink`.
    /// `frame_hint` bounds the first reserve for each stage's export buffer;
    /// the native call still retries with its own size when the hint is low.
    template <skippy::StageSink Sink>
    [[nodiscard]] std::expected<void, StagedFailure>
    prefill(std::span<const llama_token> tokens, Sink& sink,
            std::size_t frame_hint = std::size_t{4096});

    [[nodiscard]] std::size_t stage_count() const noexcept { return stages_.size(); }

    /// Clears every stage's session state so the chain can be re-prefilled.
    /// The reset is not free, so streaming callers keep feeding the chain
    /// instead of resetting between continuation steps.
    [[nodiscard]] std::optional<StagedFailure> reset();

private:
    struct Stage {
        StagedModel model;
        StagedSession session;
        /// This stage's export frame, owned and reused across calls.
        PreparedExport out;
    };

    [[nodiscard]] std::expected<void, StagedFailure> run_stage(std::size_t index,
                                                               std::span<const llama_token> tokens,
                                                               const PreparedExport* input,
                                                               std::size_t hint);

    std::vector<Stage> stages_;
};

template <skippy::StageSink Sink>
std::expected<void, StagedFailure> StageChain::prefill(std::span<const llama_token> tokens,
                                                       Sink& sink, std::size_t frame_hint) {
    if (stages_.empty()) {
        return std::unexpected(StagedFailure{.status = 0, .message = "chain is empty"});
    }
    if (auto result = run_stage(0, tokens, nullptr, frame_hint); !result) {
        return std::unexpected(result.error());
    }
    for (const std::size_t index : std::views::iota(std::size_t{1}, stages_.size())) {
        // Hand the previous stage's buffer over directly: no copy, no new frame.
        if (auto result = run_stage(index, tokens, &stages_[index - 1].out, frame_hint); !result) {
            return std::unexpected(result.error());
        }
        // An empty export from the final stage is the model's real answer:
        // no frontier follows the last layer, and the sink must see exactly
        // that, not an earlier stage's intermediate frame. An empty export
        // anywhere else is a broken chain — the next stage would run without
        // its input — so it fails instead of silently surfacing a partial
        // frame.
        if (stages_[index].out.written().empty()) {
            if (index + 1 < stages_.size()) {
                return std::unexpected(StagedFailure{
                    .status = 0,
                    .message = "stage " + std::to_string(index) +
                               " exported nothing but the chain continues",
                });
            }
            break;
        }
    }
    const PreparedExport& terminal = stages_.back().out;
    if (auto reason = sink.consume(skippy::FrameBytes{terminal.written()}); reason.has_value()) {
        return std::unexpected(StagedFailure{.status = 0, .message = std::move(*reason)});
    }
    return {};
}

} // namespace skippy::llama

#endif // SKIPPY_API_LLAMA_STAGE_CHAIN_H
