#pragma once

// The cxx-facing facade over NativeRuntime<LlamaBackend>.
//
// Ownership:
//   Bridge            owns the runtime (pimpl, so this header never sees llama.h)
//   Model / Session   skippy's own non-forgeable tokens; Rust holds them in
//                     UniquePtr and gives them back to unload/release
// Shared enums and structs below are defined by the cxx bridge (lib.rs); the
// generated header completes them. Nothing here throws: every status is a
// return value.

#include "llama/stage_chain.h"
#include "llama/staged.h"
#include "runtime.h"

#include "rust/cxx.h"

#include <cstdint>
#include <memory>

namespace skippy::ffi {

enum class RuntimeStatus : std::uint8_t;
enum class WorkloadStatus : std::uint8_t;
struct SessionConfig;
struct ModelResult;
struct SessionResult;
struct SuppressTokensResult;
struct WorkloadResult;
struct EmbedResult;
struct RerankBatchResult;
struct RerankResult;
struct StagePlanInfo;
struct PlanStagesResult;
struct ChainValidation;
struct StagedModelResult;
struct StagedSessionResult;
struct ChainResult;
struct ChainRunResult;
struct StagedEmbedResult;
enum class ModelFamily : ::std::uint8_t;
struct EncodeResult;

/// Thin owners so the cxx bridge can hold a stage in a UniquePtr.
/// Owns a chain of stages; cxx holds it behind a Box because it is move-only.
class StageChainHandle {
public:
    StageChainHandle() = default;
    explicit StageChainHandle(skippy::llama::StageChain chain) noexcept
        : chain_(std::move(chain)) {}
    [[nodiscard]] skippy::llama::StageChain& get() noexcept { return chain_; }

private:
    skippy::llama::StageChain chain_;
};

class StagedModel {
public:
    explicit StagedModel(skippy::llama::StagedModel model) noexcept : model_(std::move(model)) {}
    [[nodiscard]] const skippy::llama::StagedModel& get() const noexcept { return model_; }
    [[nodiscard]] skippy::llama::StagedModel& get() noexcept { return model_; }

private:
    skippy::llama::StagedModel model_;
};

class StagedSession {
public:
    explicit StagedSession(skippy::llama::StagedSession session) noexcept
        : session_(std::move(session)) {}
    [[nodiscard]] skippy::llama::StagedSession& get() noexcept { return session_; }

private:
    skippy::llama::StagedSession session_;
};

class Bridge {
public:
    Bridge();
    ~Bridge();

    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    [[nodiscard]] ModelResult load_model(rust::Str path) noexcept;

    /// Plans `stage_count` contiguous stages over the model at `path`.
    /// Failure is the envelope's, never an entry's: the result carries
    /// ok/error and the stages together, so an out-of-memory entry cannot be
    /// mistaken for "a model with no plan".
    [[nodiscard]] PlanStagesResult plan_stages(rust::Str path,
                                               std::uint32_t stage_count) const noexcept;

    /// Opens stage `index` of `stages` over the model at `path`.
    [[nodiscard]] StagedModelResult load_staged_model(rust::Str path, std::uint32_t stages,
                                                      std::uint32_t index) noexcept;

    /// Opens a session on a stage.
    [[nodiscard]] StagedSessionResult create_staged_session(const StagedModel& model) noexcept;

    /// Embeds `tokens` into `output` on a stage.
    [[nodiscard]] StagedEmbedResult staged_embed(StagedSession& session,
                                                 rust::Slice<const std::int32_t> tokens,
                                                 rust::Slice<float> output) noexcept;

    /// Plans `ranges` (start/end pairs) and opens one stage per range.
    [[nodiscard]] ChainResult open_chain(rust::Str path,
                                         rust::Slice<const std::int32_t> ranges) noexcept;

    /// Runs `tokens` through every stage, returning the terminal frame if any.
    [[nodiscard]] ChainRunResult run_chain(StageChainHandle& chain,
                                           rust::Slice<const std::int32_t> tokens) noexcept;

    /// Reads the model's family from its GGUF architecture.
    [[nodiscard]] ModelFamily model_family(rust::Str path) const noexcept;

    /// Checks that a plan chain is contiguous and hands over cleanly.
    [[nodiscard]] ChainValidation validate_chain(rust::Str path,
                                                 std::uint32_t stage_count) const noexcept;
    [[nodiscard]] RuntimeStatus unload_model(Model& model) noexcept;

    [[nodiscard]] SessionResult create_session(const Model& model, SessionConfig config) noexcept;
    [[nodiscard]] RuntimeStatus release_session(Session& session) noexcept;

    [[nodiscard]] WorkloadResult workload(const Session& session) const noexcept;

    /// The model's declared vocab suppress tokens (patch 0012), as native ids.
    /// Empty when the GGUF declares none.
    [[nodiscard]] SuppressTokensResult suppress_tokens(const Model& model) const noexcept;

    // Replaces the contents of `tokens`.
    [[nodiscard]] WorkloadStatus tokenize(const Session& session, rust::Str text,
                                          rust::Vec<std::int32_t>& tokens) noexcept;

    // Writes the normalised embedding into `output`; `dimensions` is also
    // filled on OutputTooSmall so the caller can resize and retry.
    [[nodiscard]] EmbedResult embed(const Session& session, rust::Slice<const std::int32_t> tokens,
                                    rust::Slice<float> output) noexcept;

    // `tokens` holds every group concatenated; `lengths[i]` is group i's size.
    [[nodiscard]] EmbedResult embed_batch(const Session& session,
                                          rust::Slice<const std::int32_t> tokens,
                                          rust::Slice<const std::uint32_t> lengths,
                                          rust::Slice<float> output) noexcept;

    [[nodiscard]] RerankResult rerank(const Session& session, rust::Str query,
                                      rust::Str document) noexcept;

    // One score per document; the whole group is decoded in a single pass.
    [[nodiscard]] RerankBatchResult rerank_batch(const Session& session, rust::Str query,
                                                 rust::Slice<const rust::Str> documents,
                                                 rust::Slice<float> scores) noexcept;

    [[nodiscard]] EncodeResult encode(const Session& session,
                                      rust::Slice<const std::int32_t> tokens) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Null when the runtime could not be allocated.
[[nodiscard]] std::unique_ptr<Bridge> new_bridge() noexcept;

} // namespace skippy::ffi
