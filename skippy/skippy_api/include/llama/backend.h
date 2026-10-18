#pragma once

// Ownership:
//   LlamaBackend  owns nothing (held by value in the runtime)
//   LlamaModel    owns llama_model   (llama_model_ptr) and the idle contexts a
//                 released session hands back
//   LlamaSession  owns llama_context (llama_context_ptr), its batch and tokenizer scratch;
//                 borrows the model's vocab and idle list (the runtime destroys
//                 every session before any model)
// Only include/llama/* and src/llama/* include llama.h. No C++ tests by rule;
// the static_asserts in backend.cpp are the C++ test surface.

#include "embed.h"
#include "encode.h"
#include "llama/batch.h"
#include "rerank.h"
#include "runtime.h"
#include "stage_load.h"
#include "stage_plan.h"
#include "types.h"
#include "workload.h"
#include "workload_error.h"

#include "llama-cpp.h"
#include "llama.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace skippy::llama {

class LlamaBackend;

// Owned strings behind RerankPromptFormat; read off the model once at load.
struct RerankPromptSource {
    std::optional<std::string> prompt_template;
    std::string eos_piece;
    std::string sep_piece;

    [[nodiscard]] RerankPromptFormat view() const noexcept {
        return {
            .prompt_template =
                prompt_template ? std::optional<std::string_view>{*prompt_template} : std::nullopt,
            .eos_piece = eos_piece,
            .sep_piece = sep_piece,
        };
    }
};

class LlamaModel {
public:
    // Facts that cost metadata reads or string assembly. Derived once at load:
    // a session needs them, but re-deriving them per session would put model
    // introspection on the request path.
    struct Traits {
        enum llama_pooling_type declared_pooling;
        std::uint32_t classifier_outputs;
        std::optional<RerankPromptSource> rerank_prompt;
        std::optional<Token> decoder_start;
    };

    LlamaModel(llama_model_ptr model, Traits traits) noexcept;

    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;
    LlamaModel(LlamaModel&&) = delete;
    LlamaModel& operator=(LlamaModel&&) = delete;

    [[nodiscard]] bool has_encoder() const noexcept { return has_encoder_; }
    [[nodiscard]] bool has_decoder() const noexcept { return has_decoder_; }
    [[nodiscard]] std::size_t embedding_dimensions() const noexcept {
        return embedding_dimensions_;
    }
    [[nodiscard]] const Traits& traits() const noexcept { return traits_; }

private:
    friend class LlamaBackend;
    friend class LlamaSession;

    // A session's context costs milliseconds to build (K/V cache, graph
    // reservation, backend buffers) and microseconds to wrap. A released
    // session therefore hands its context back instead of destroying it, and
    // the next session with the same geometry takes it. Two slots keep a
    // request-per-session server from rebuilding without hoarding memory.
    static constexpr std::size_t kRetainedContexts = 2;

    struct IdleContext {
        std::uint32_t max_tokens;
        std::uint32_t sequences;
        llama_context_ptr context;
    };

    [[nodiscard]] llama_context_ptr take_idle_context(std::uint32_t max_tokens,
                                                      std::uint32_t sequences) noexcept;
    void recycle(llama_context_ptr context, std::uint32_t max_tokens,
                 std::uint32_t sequences) noexcept;

    [[nodiscard]] llama_model& native() noexcept { return *model_; }
    [[nodiscard]] const llama_vocab& vocab() const noexcept {
        return *llama_model_get_vocab(model_.get());
    }

    llama_model_ptr model_;
    bool has_encoder_;
    bool has_decoder_;
    std::size_t embedding_dimensions_;
    Traits traits_;
    // Declared last: destroyed before model_, which the contexts depend on.
    // Fixed slots, not a vector: releasing a session must never allocate or
    // throw, so capacity is decided when the model loads.
    std::array<std::optional<IdleContext>, kRetainedContexts> idle_contexts_{};
    std::size_t next_slot_ = 0;
};

class LlamaSession {
public:
    LlamaSession(llama_context_ptr context, LlamaModel& model, const llama_vocab& vocab,
                 WorkloadInfo workload, std::optional<Token> decoder_start,
                 std::uint32_t max_tokens, std::uint32_t sequences) noexcept;
    ~LlamaSession();

    LlamaSession(const LlamaSession&) = delete;
    LlamaSession& operator=(const LlamaSession&) = delete;
    LlamaSession(LlamaSession&&) = delete;
    LlamaSession& operator=(LlamaSession&&) = delete;

    [[nodiscard]] const WorkloadInfo& workload() const noexcept { return workload_; }

    // EmbeddingSession
    [[nodiscard]] std::size_t embedding_dimensions() const noexcept {
        return workload_.output_dimensions;
    }
    // Returned span borrows the context until the next decode or encode.
    [[nodiscard]] std::expected<detail::NativeEmbeddingView, WorkloadError>
    embed(std::span<const Token> tokens);

    // Batched workloads: every input is one sequence of the same decode, and
    // the returned rows are contiguous (row s starts at s * dimensions).
    [[nodiscard]] std::expected<detail::NativeEmbeddingView, WorkloadError>
    embed_many(std::span<const std::span<const Token>> inputs);
    [[nodiscard]] std::expected<std::span<const float>, WorkloadError>
    score_many(std::span<const std::span<const Token>> inputs);

    // RerankSession
    [[nodiscard]] RerankPromptFormat rerank_prompt_format() const noexcept {
        return rerank_prompt_ != nullptr ? rerank_prompt_->view() : RerankPromptFormat{};
    }
    // Returned span borrows session scratch until the next tokenize().
    [[nodiscard]] std::expected<std::span<const Token>, WorkloadError>
    tokenize(std::string_view text);
    [[nodiscard]] std::expected<float, WorkloadError> score(std::span<const Token> tokens);

    // EncoderSession
    [[nodiscard]] std::expected<Token, WorkloadError> encode(std::span<const Token> tokens);

private:
    enum class Pass { Decode, Encode };

    [[nodiscard]] std::expected<void, WorkloadError> process_tokens(std::span<const Token> tokens,
                                                                    Pass pass);
    [[nodiscard]] std::expected<void, WorkloadError>
    process_sequences(std::span<const std::span<const Token>> inputs);
    [[nodiscard]] std::expected<void, WorkloadError> reserve_batch(std::int32_t count);
    [[nodiscard]] std::int32_t run_pass(Pass pass) noexcept;

    /// The backend's message from the last failed pass, if it threw one.
    /// Keeps OOM distinguishable from a Metal error instead of one code.
    [[nodiscard]] const std::string& last_backend_error() const noexcept {
        return last_backend_error_;
    }

    static constexpr SequenceId sequence_{0};

    llama_context_ptr context_;
    LlamaModel& model_;
    const llama_vocab& vocab_;
    WorkloadInfo workload_;
    const RerankPromptSource* rerank_prompt_;
    std::optional<Token> decoder_start_;
    std::uint32_t batch_limit_;
    EmbeddingBatchPtr batch_;
    std::int32_t batch_capacity_ = 0;
    std::vector<llama_token> native_tokens_;
    std::vector<Token> tokens_;
    std::vector<float> batch_output_;
    std::uint32_t max_tokens_;
    std::uint32_t sequences_;
    std::string last_backend_error_;
};

/// The immutable index of realized plans. Copy-on-write: publishers build a
/// new one and install it with a compare-and-swap, so readers always see a
/// complete index and never block.
struct PlanIndex {
    std::unordered_map<std::string, std::vector<std::shared_ptr<const StagePlan>>> by_key;
};

/// A read lease over the plan store. The configs hold `string_view`s into
/// plans the store owns, so the lease pins the index (and with it every plan
/// it references) for exactly as long as the configs are alive — with no lock
/// held. This is also what makes a borrowed-from-temporary safe: the pinned
/// index outlives the backend that produced it.
struct PlanLease {
    std::vector<StageLoadConfig> configs;
    std::shared_ptr<const PlanIndex> pinned;
};

/// Lock-free plan storage: readers snapshot an atomic index pointer, writers
/// publish copy-on-write via CAS. `std::atomic<std::shared_ptr<T>>` uses a
/// short internal spin on libc++ rather than being truly lock-free, but reads
/// dominate (one per plan call) and publishes are once per model version, so
/// the snapshot stays a few uncontended instructions.
class PlanStore {
public:
    PlanStore() : index_(std::make_shared<const PlanIndex>()) {}

    /// A consistent snapshot. Holding the returned pointer keeps every plan in
    /// it alive; dropping it may free plans nobody else references.
    [[nodiscard]] std::shared_ptr<const PlanIndex> snapshot() const noexcept {
        return std::atomic_load_explicit(&index_, std::memory_order_acquire);
    }

    /// Publishes `plans` under `key` unless a racing publisher got there
    /// first; either way returns the index that won, so callers observe the
    /// other thread's work instead of planning twice.
    [[nodiscard]] std::shared_ptr<const PlanIndex>
    publish(const std::string& key, std::vector<std::shared_ptr<const StagePlan>> plans);

private:
    friend std::shared_ptr<const PlanIndex> PlanStore_test_snapshot(const PlanStore&);

    // std::atomic<shared_ptr> is not in every libc++; the free functions are
    // the portable form of the same RCU publication.
    mutable std::shared_ptr<const PlanIndex> index_;
};

class LlamaBackend {
public:
    using ModelResource = LlamaModel;
    using SessionResource = LlamaSession;

    // llama_backend_init() once per process; llama_backend_free() is never
    // called, since no instance can know it is the last.
    LlamaBackend();

    /// Plan `ranges` of `path`, caching realized plans in *this* runtime.
    /// The returned lease pins the plans the configs borrow from; hold it as
    /// long as you hold the configs.
    [[nodiscard]] std::expected<PlanLease, PlanFailure>
    plan_model_stages(const std::string& path, std::span<const LayerRange> ranges);

    /// Plan `ranges` and return the plans themselves. Both share this
    /// backend's store, so calling either first warms the other.
    [[nodiscard]] std::expected<std::vector<StagePlan>, PlanFailure>
    plan_model_plans(const std::string& path, std::span<const LayerRange> ranges);

    [[nodiscard]] PlanStore& plan_store() noexcept { return plan_store_; }

    [[nodiscard]] std::expected<std::unique_ptr<LlamaModel>, RuntimeError>
    load_model(const ModelConfig& config);
    [[nodiscard]] std::expected<std::unique_ptr<LlamaSession>, RuntimeError>
    create_session(LlamaModel& model, const SessionConfig& config);

private:
    PlanStore plan_store_;
};

using LlamaRuntime = NativeRuntime<LlamaBackend>;

} // namespace skippy::llama
