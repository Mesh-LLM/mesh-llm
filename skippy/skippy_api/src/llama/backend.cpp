#include "llama/backend.h"
#include "llama/model_inventory.h"
#include "workload_profile.h"

#include "llama/stage_planner.h"

#include "batch.h"
#include "pooling.h"
#include "selector.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <ranges>
#include <string>
#include <utility>

namespace skippy::llama {
namespace {

// Upper bound on sequences one session may batch. Past this the K/V and
// scheduling bookkeeping costs more than a batched decode saves.
constexpr std::uint32_t kMaxSequences = 64;

std::string token_piece(const llama_vocab& vocab, llama_token token) {
    std::string piece(32, '\0');
    const std::int32_t written = llama_token_to_piece(
        &vocab, token, piece.data(), static_cast<std::int32_t>(piece.size()), 0, true);
    if (written >= 0) {
        piece.resize(static_cast<std::size_t>(written));
        return piece;
    }
    piece.resize(static_cast<std::size_t>(-written));
    const std::int32_t rewritten = llama_token_to_piece(
        &vocab, token, piece.data(), static_cast<std::int32_t>(piece.size()), 0, true);
    piece.resize(static_cast<std::size_t>(std::max(rewritten, 0)));
    return piece;
}

RerankPromptSource rerank_prompt_source(const llama_model& model, const llama_vocab& vocab) {
    RerankPromptSource source;
    if (const char* prompt_template = llama_model_chat_template(&model, "rerank")) {
        source.prompt_template = prompt_template;
    }
    if (llama_vocab_get_add_eos(&vocab))
        source.eos_piece = token_piece(vocab, llama_vocab_eos(&vocab));
    if (llama_vocab_get_add_sep(&vocab))
        source.sep_piece = token_piece(vocab, llama_vocab_sep(&vocab));
    return source;
}

// Metadata is text; a malformed value must not silently become pooling type
// zero. The buffer meta_val_str fills is NUL-terminated, so strtol can consume
// the whole value and report where it stopped.
std::optional<enum llama_pooling_type> parse_pooling(const char* value) noexcept {
    if (value == nullptr || *value == '\0') return std::nullopt;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') return std::nullopt;
    // A corrupted GGUF can declare any integer; only the values llama knows
    // are meaningful, so anything else is a rejection rather than a silent
    // context with an undefined pooling mode.
    switch (parsed) {
    case LLAMA_POOLING_TYPE_UNSPECIFIED:
    case LLAMA_POOLING_TYPE_NONE:
    case LLAMA_POOLING_TYPE_MEAN:
    case LLAMA_POOLING_TYPE_CLS:
    case LLAMA_POOLING_TYPE_LAST:
    case LLAMA_POOLING_TYPE_RANK: return static_cast<enum llama_pooling_type>(parsed);
    default: return std::nullopt;
    }
}

// The pooling a GGUF declares for itself. Read from metadata rather than from a
// context, so a classifier model costs one context, not two.
constexpr DeclaredPooling declared_pooling_of(enum llama_pooling_type pooling) noexcept {
    switch (pooling) {
    case LLAMA_POOLING_TYPE_NONE: return DeclaredPooling::None;
    case LLAMA_POOLING_TYPE_MEAN: return DeclaredPooling::Mean;
    case LLAMA_POOLING_TYPE_CLS: return DeclaredPooling::Cls;
    case LLAMA_POOLING_TYPE_LAST: return DeclaredPooling::Last;
    case LLAMA_POOLING_TYPE_RANK: return DeclaredPooling::Rank;
    default: return DeclaredPooling::Unspecified;
    }
}

enum llama_pooling_type model_pooling_type(const llama_model& model) {
    std::array<char, 64> architecture{};
    if (llama_model_meta_val_str(&model, "general.architecture", architecture.data(),
                                 architecture.size()) < 0) {
        return LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
    const std::string key = std::string(architecture.data()) + ".pooling_type";
    std::array<char, 32> value{};
    if (llama_model_meta_val_str(&model, key.c_str(), value.data(), value.size()) < 0) {
        return LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
    return parse_pooling(value.data()).value_or(LLAMA_POOLING_TYPE_UNSPECIFIED);
}

llama_context_params session_params(std::uint32_t max_tokens, enum llama_pooling_type pooling,
                                    std::uint32_t sequences, bool embeddings) {
    llama_context_params params = llama_context_default_params();
    // Whole input in one physical batch, as non-causal models require. The
    // context holds one max_tokens budget per sequence, so a batch call has
    // room for `sequences` full-length inputs.
    const std::uint32_t total = max_tokens * sequences;
    params.n_ctx = total;
    params.n_batch = total;
    params.n_ubatch = total;
    params.n_seq_max = sequences;
    // Patch 0011: only workloads that read embeddings pay to carry them. A
    // bare causal decoder runs leaner; embedders, rerankers and encoder-only
    // models turn them on.
    params.embeddings = embeddings;
    params.pooling_type = pooling;
    return params;
}

PooledGeometry pooled_geometry(const llama_context& context) noexcept {
    // The pool labels from the context, never from a caller: llama pads n_ctx
    // to 256 while n_batch/n_ubatch keep the requested size, so the smallest
    // of the three per sequence is the batch a recycled context can serve.
    const std::uint32_t sequences = std::max(1U, llama_n_seq_max(&context));
    const std::uint32_t tokens =
        std::min({llama_n_ctx(&context), llama_n_batch(&context), llama_n_ubatch(&context)}) /
        sequences;
    return {.tokens = tokens, .sequences = sequences};
}

std::optional<Token> decoder_start_token(const llama_model& model, const llama_vocab& vocab) {
    llama_token start = llama_model_decoder_start_token(&model);
    if (start == LLAMA_TOKEN_NULL) start = llama_vocab_bos(&vocab);
    if (start == LLAMA_TOKEN_NULL) return std::nullopt;
    return Token{start};
}

} // namespace

LlamaModel::LlamaModel(llama_model_ptr model, Traits traits) noexcept
    : model_(std::move(model)), has_encoder_(llama_model_has_encoder(model_.get())),
      has_decoder_(llama_model_has_decoder(model_.get())),
      embedding_dimensions_(
          static_cast<std::size_t>(std::max(llama_model_n_embd_out(model_.get()), 0))),
      traits_(std::move(traits)) {}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): T2 SessionGeometry, TYPE-RULES §0
LlamaSession::LlamaSession(llama_context_ptr context, LlamaModel& model, const llama_vocab& vocab,
                           WorkloadInfo workload, std::optional<Token> decoder_start,
                           std::uint32_t max_tokens, std::uint32_t sequences) noexcept
    // NOLINTEND(bugprone-easily-swappable-parameters)
    : context_(std::move(context)), model_(model), vocab_(vocab), workload_(workload),
      rerank_prompt_(model.traits().rerank_prompt ? &*model.traits().rerank_prompt : nullptr),
      decoder_start_(decoder_start), batch_limit_(llama_n_ubatch(context_.get())),
      max_tokens_(max_tokens), sequences_(sequences) {
    // Pre-size the tokenize scratch to the session's whole budget: the first
    // tokenize on a fresh session then runs in one native call, and
    // create_session bounds max_tokens by the context's batch, so this is
    // proportional to memory the context already owns. The -count growth
    // path remains for over-long text. The constructor is noexcept, and the
    // pre-size is an optimisation, so a failed allocation falls back to the
    // growth path instead of escaping.
    try {
        native_tokens_.resize(max_tokens_);
    } catch (...) { // NOLINT(bugprone-empty-catch): pre-size only; growth path recovers
    }
}

LlamaSession::~LlamaSession() {
    if (!context_) return;
    // The pool takes the context back quiescent and labelled with the capacity
    // llama actually built: a pass may still be in flight (llama_decode
    // returning does not mean the backend finished), and a warmed larger
    // context relabelled with this session's smaller geometry hides capacity
    // the pool could otherwise serve.
    llama_synchronize(context_.get());
    model_.recycle(std::move(context_));
}

llama_context_ptr LlamaModel::take_idle_context(std::uint32_t max_tokens,
                                                std::uint32_t sequences) noexcept {
    // A larger idle context serves a smaller request: n_ctx and n_seq_max are
    // capacities, and every call clears the K/V it will touch. Take the
    // smallest slot that fits so a 768-token request does not build a new
    // context when a warmed 1024-token one is sitting idle.
    // Best fit, not first fit: taking the smallest context that serves the
    // request keeps the large geometries available for the requests that need
    // them, so an alternating small/large workload stops churning the pool.
    const IdleContext* best = nullptr;
    for (const auto& slot : idle_contexts_) {
        if (!slot) continue;
        const IdleContext& idle = *slot;
        if (idle.geometry.tokens < max_tokens || idle.geometry.sequences < sequences) continue;
        if (best == nullptr || idle.geometry.tokens < best->geometry.tokens) best = &idle;
    }
    if (best == nullptr) return {};
    // The slot still owns the context; move it out and free the slot. Finding
    // the owner again by address keeps the optional access visibly checked.
    for (auto& slot : idle_contexts_) {
        if (!slot || &*slot != best) continue;
        llama_context_ptr context = std::move(slot->context);
        slot.reset();
        return context;
    }
    return {};
}

void LlamaModel::recycle(llama_context_ptr context) noexcept {
    if (!context) return;
    IdleContext released{
        .geometry = pooled_geometry(*context),
        .context = std::move(context),
    };
    auto* const free = std::ranges::find_if(idle_contexts_, [](const auto& slot) { return !slot; });
    if (free != idle_contexts_.end()) {
        free->emplace(std::move(released));
        return;
    }
    // Every slot is taken: evict round-robin rather than by age.
    idle_contexts_[next_slot_].emplace(std::move(released));
    next_slot_ = (next_slot_ + 1) % idle_contexts_.size();
}

std::expected<void, WorkloadError> LlamaSession::reserve_batch(std::int32_t count) {
    if (batch_ && count <= batch_capacity_) return {};
    return allocate_embedding_batch(count, static_cast<std::int32_t>(sequences_))
        .transform([&](EmbeddingBatchPtr batch) {
            batch_ = std::move(batch);
            batch_capacity_ = count;
        });
}

std::expected<void, WorkloadError>
LlamaSession::process_sequences(std::span<const std::span<const Token>> inputs) {
    if (inputs.empty()) return std::unexpected(WorkloadError::EmptyInput);
    if (inputs.size() > sequences_) return std::unexpected(WorkloadError::TooManySequences);

    std::size_t total = 0;
    for (const auto& input : inputs) {
        if (input.empty()) return std::unexpected(WorkloadError::EmptyInput);
        // The per-sequence budget too: the total check below cannot see a
        // 1500-token row in a {512, 4} session, which the single-sequence
        // path would reject for the same input.
        if (input.size() > max_tokens_) return std::unexpected(WorkloadError::TooManyTokens);
        total += input.size();
        if (total > static_cast<std::size_t>(max_tokens_) * sequences_) {
            return std::unexpected(WorkloadError::TooManyTokens);
        }
    }
    // The pool labels by min(n_ctx, n_batch, n_ubatch), but n_ctx is padded
    // while the batch limits are not, so a pooled context can still serve a
    // request whose total fits its label yet exceeds its physical ubatch -
    // llama_decode aborts there instead of failing. The ubatch is the hard
    // floor.
    if (total > batch_limit_) return std::unexpected(WorkloadError::TooManyTokens);

    if (auto reserved = reserve_batch(static_cast<std::int32_t>(total)); !reserved) {
        return reserved;
    }
    detail::fill_embedding_batches(*batch_, inputs);

    // llama_decode returns before the backend finishes; the embeddings getters
    // synchronise, llama_memory_clear does not. Clearing in-flight memory is a
    // race on every GPU backend, so quiesce first - a no-op when idle.
    if (llama_memory_t memory = llama_get_memory(context_.get())) {
        llama_synchronize(context_.get());
        llama_memory_clear(memory, true);
    }
    if (run_pass(Pass::Decode) != 0) {
        return std::unexpected(WorkloadError::DecodeFailed);
    }
    return {};
}

std::expected<detail::NativeEmbeddingView, WorkloadError>
LlamaSession::embed_many(std::span<const std::span<const Token>> inputs) {
    if (workload_.kind != WorkloadKind::Embedding || !workload_.embedding_pooling) {
        return std::unexpected(WorkloadError::UnsupportedWorkload);
    }
    if (auto processed = process_sequences(inputs); !processed) {
        return std::unexpected(processed.error());
    }

    const std::size_t width = workload_.output_dimensions;
    if (width == 0) return std::unexpected(WorkloadError::NoEmbeddingDimensions);

    batch_output_.resize(inputs.size() * width);
    std::int32_t batch_at = 0;
    for (const std::size_t row : std::views::iota(std::size_t{0}, inputs.size())) {
        const float* data = nullptr;
        if (*workload_.embedding_pooling == Pooling::None) {
            data = llama_get_embeddings_ith(
                context_.get(), batch_at + static_cast<std::int32_t>(inputs[row].size()) - 1);
        } else {
            data = llama_get_embeddings_seq(context_.get(), static_cast<llama_seq_id>(row));
        }
        if (data == nullptr) return std::unexpected(WorkloadError::NoOutput);
        std::copy_n(data, width, batch_output_.data() + (row * width));
        batch_at += static_cast<std::int32_t>(inputs[row].size());
    }
    return detail::NativeEmbeddingView{batch_output_};
}

std::expected<std::span<const float>, WorkloadError>
LlamaSession::score_many(std::span<const std::span<const Token>> inputs) {
    if (workload_.kind != WorkloadKind::Rerank || workload_.classifier_outputs != 1) {
        return std::unexpected(workload_.kind != WorkloadKind::Rerank
                                   ? WorkloadError::UnsupportedWorkload
                                   : WorkloadError::UnsupportedClassifierCount);
    }
    if (auto processed = process_sequences(inputs); !processed) {
        return std::unexpected(processed.error());
    }

    batch_output_.resize(inputs.size());
    for (const std::size_t row : std::views::iota(std::size_t{0}, inputs.size())) {
        const float* ranks =
            llama_get_embeddings_seq(context_.get(), static_cast<llama_seq_id>(row));
        if (ranks == nullptr) return std::unexpected(WorkloadError::NoOutput);
        batch_output_[row] = ranks[0];
    }
    return std::span<const float>{batch_output_};
}

std::int32_t LlamaSession::run_pass(Pass pass) noexcept {
    try {
        return pass == Pass::Decode ? llama_decode(context_.get(), *batch_)
                                    : llama_encode(context_.get(), *batch_);
    } catch (const std::exception& failure) {
        // The backend's own message survives into the failure rather than
        // collapsing OOM, Metal errors and everything else into one code.
        // Recording it must not throw out of noexcept either: if the process
        // is out of memory the message is dropped and the failure code kept.
        try {
            last_backend_error_ = failure.what();
        } catch (...) {
            // Out of memory recording the message: drop any stale text and
            // keep the failure code; clear() is noexcept.
            last_backend_error_.clear();
        }
        return -1;
    }
}

std::expected<void, WorkloadError> LlamaSession::process_tokens(std::span<const Token> tokens,
                                                                Pass pass) {
    if (tokens.empty()) return std::unexpected(WorkloadError::EmptyInput);
    // The *configured* geometry bounds a session, not the capacity of the
    // context it happens to be served by: a warmed larger context is reused for
    // smaller requests, and the contract must not silently widen with it.
    if (tokens.size() > detail::max_batch_tokens || tokens.size() > max_tokens_ ||
        tokens.size() > batch_limit_) {
        return std::unexpected(WorkloadError::TooManyTokens);
    }
    const auto count = static_cast<std::int32_t>(tokens.size());
    if (auto reserved = reserve_batch(count); !reserved) return reserved;
    detail::fill_embedding_batch(*batch_, tokens, sequence_);

    // Same contract as process_sequences: quiesce before clearing memory.
    if (llama_memory_t memory = llama_get_memory(context_.get())) {
        llama_synchronize(context_.get());
        llama_memory_clear(memory, true);
    }
    if (run_pass(pass) != 0) {
        return std::unexpected(pass == Pass::Decode ? WorkloadError::DecodeFailed
                                                    : WorkloadError::EncodeFailed);
    }
    return {};
}

std::expected<detail::NativeEmbeddingView, WorkloadError>
LlamaSession::embed(std::span<const Token> tokens) {
    if (workload_.kind != WorkloadKind::Embedding || !workload_.embedding_pooling) {
        return std::unexpected(WorkloadError::UnsupportedWorkload);
    }
    if (auto processed = process_tokens(tokens, Pass::Decode); !processed) {
        return std::unexpected(processed.error());
    }

    const std::size_t width = workload_.output_dimensions;
    const auto view = [width](const float* data) -> detail::NativeEmbeddingResult {
        if (data == nullptr) return std::nullopt;
        return detail::NativeEmbeddingView{data, width};
    };
    const auto last = TokenIndex{static_cast<std::int32_t>(tokens.size()) - 1};

    const detail::NativeEmbeddingResult selected =
        with_pooling(*workload_.embedding_pooling, [&](auto tag) {
            constexpr Pooling mode = decltype(tag)::value;
            // One selector shape, two address modes: the None pool reads the
            // last token of the sequence, every other pool reads the sequence
            // output. Only the lookup differs, so only the lookup branches.
            if constexpr (mode == Pooling::None) {
                return detail::EmbeddingSelector<mode>::select(last, [&](TokenIndex index) {
                    return view(
                        llama_get_embeddings_ith(context_.get(), static_cast<std::int32_t>(index)));
                });
            } else {
                return detail::EmbeddingSelector<mode>::select(sequence_, [&](SequenceId sequence) {
                    return view(llama_get_embeddings_seq(context_.get(),
                                                         static_cast<llama_seq_id>(sequence)));
                });
            }
        });

    if (!selected) return std::unexpected(WorkloadError::NoOutput);
    return *selected;
}

std::expected<std::span<const Token>, WorkloadError> LlamaSession::tokenize(std::string_view text) {
    if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return std::unexpected(WorkloadError::TokenizationFailed);
    }
    const auto length = static_cast<std::int32_t>(text.size());

    // The buffer stays at its capacity between calls. Growing it back from
    // `count` on every call would value-initialise the tail, which is most of
    // what a short tokenize costs.
    if (native_tokens_.empty()) native_tokens_.resize(1);
    // llama_tokenize takes an explicit length, so the view need not be
    // null-terminated; the check cannot see that through the call.
    // NOLINTBEGIN(bugprone-suspicious-stringview-data-usage)
    std::int32_t count =
        llama_tokenize(&vocab_, text.data(), length, native_tokens_.data(),
                       static_cast<std::int32_t>(native_tokens_.size()), true, true);
    // llama_tokenize returns INT32_MIN (not a negative length) when the text
    // itself cannot fit any token buffer, and -count when the buffer is merely
    // too small. Only the first is unfixable by growing.
    if (count == std::numeric_limits<std::int32_t>::min()) {
        return std::unexpected(WorkloadError::TooManyTokens);
    }
    if (count < 0) {
        native_tokens_.resize(static_cast<std::size_t>(-count));
        count = llama_tokenize(&vocab_, text.data(), length, native_tokens_.data(),
                               static_cast<std::int32_t>(native_tokens_.size()), true, true);
        if (count < 0) return std::unexpected(WorkloadError::TokenizationFailed);
    }
    // NOLINTEND(bugprone-suspicious-stringview-data-usage)
    return detail::tokens_from_native(
        std::span<std::int32_t>{native_tokens_.data(), static_cast<std::size_t>(count)}, tokens_);
}

std::expected<float, WorkloadError> LlamaSession::score(std::span<const Token> tokens) {
    if (workload_.kind != WorkloadKind::Rerank)
        return std::unexpected(WorkloadError::UnsupportedWorkload);
    if (workload_.classifier_outputs != 1)
        return std::unexpected(WorkloadError::UnsupportedClassifierCount);
    if (auto processed = process_tokens(tokens, Pass::Decode); !processed) {
        return std::unexpected(processed.error());
    }
    // Rank pooling: the sequence output is float[n_cls_out].
    const float* ranks =
        llama_get_embeddings_seq(context_.get(), static_cast<llama_seq_id>(sequence_));
    if (ranks == nullptr) return std::unexpected(WorkloadError::NoOutput);
    return ranks[0];
}

std::expected<Token, WorkloadError> LlamaSession::encode(std::span<const Token> tokens) {
    if (workload_.kind != WorkloadKind::EncoderDecoder)
        return std::unexpected(WorkloadError::UnsupportedWorkload);
    // A local keeps the presence check and the read visibly connected.
    const std::optional<Token> decoder_start = decoder_start_;
    if (!decoder_start) return std::unexpected(WorkloadError::NoDecoderStartToken);
    if (auto processed = process_tokens(tokens, Pass::Encode); !processed) {
        return std::unexpected(processed.error());
    }
    return *decoder_start;
}

LlamaBackend::LlamaBackend() {
    static std::once_flag initialised;
    std::call_once(initialised, [] { llama_backend_init(); });
}

std::expected<PlanLease, PlanFailure>
LlamaBackend::plan_model_stages(const std::string& path, std::span<const LayerRange> ranges) {
    return skippy::llama::plan_model_stages(*this, path, ranges);
}

std::expected<std::vector<StagePlan>, PlanFailure>
LlamaBackend::plan_model_plans(const std::string& path, std::span<const LayerRange> ranges) {
    return skippy::llama::plan_model_plans(*this, path, ranges);
}

std::expected<std::unique_ptr<LlamaModel>, LoadFailure>
LlamaBackend::load_model(const ModelConfig& config) {
    const llama_model_params params = llama_model_default_params();
    llama_model_ptr model{llama_model_load_from_file(config.path.c_str(), params)};
    if (!model)
        return std::unexpected(LoadFailure{
            .code = RuntimeError::LoadFailed,
            .message = "llama: could not load " + config.path,
        });

    const llama_vocab& vocab = *llama_model_get_vocab(model.get());
    const enum llama_pooling_type declared_pooling = model_pooling_type(*model);
    // n_cls_out defaults to 1; the head is the cls.output.weight tensor (AGENTS.md Traps).
    const auto inventory = probe::cached_inventory(config.path);
    if (!inventory) {
        // A card that cannot be read is a load failure, not a model without a
        // head: guessing "no classifier" would classify silently from a file
        // the parser rejected.
        return std::unexpected(
            LoadFailure{.code = RuntimeError::LoadFailed, .message = inventory.error()});
    }
    const std::uint32_t classifier_outputs =
        probe::has_classifier_head(*inventory) ? llama_model_n_cls_out(model.get()) : 0;
    // Patch 0011's table, evaluated once: the facts are load-time properties.
    const WorkloadProfile profile = classify(ModelFacts{
        .has_encoder = llama_model_has_encoder(model.get()),
        .has_decoder = llama_model_has_decoder(model.get()),
        .classifier_outputs = classifier_outputs,
        .declared_pooling = declared_pooling_of(declared_pooling),
    });
    const bool classifier_workload = profile.classifier_workload;
    const bool rerank_workload = classifier_workload || declared_pooling == LLAMA_POOLING_TYPE_RANK;

    try {
        LlamaModel::Traits traits{
            .declared_pooling = declared_pooling,
            .classifier_outputs = classifier_outputs,
            .profile = profile,
            .rerank_prompt =
                rerank_workload
                    ? std::optional<RerankPromptSource>{rerank_prompt_source(*model, vocab)}
                    : std::nullopt,
            .decoder_start = llama_model_has_encoder(model.get())
                                 ? decoder_start_token(*model, vocab)
                                 : std::nullopt,
        };
        auto loaded = std::make_unique<LlamaModel>(std::move(model), std::move(traits));

        // Sessions are created on demand, but a server should not pay the
        // context build on its first request: build the default geometry now
        // and keep it in the retained pool.
        const SessionConfig warm{};
        if (llama_context_ptr context{llama_init_from_model(
                &loaded->native(),
                session_params(warm.max_tokens,
                               classifier_workload ? LLAMA_POOLING_TYPE_RANK
                                                   : LLAMA_POOLING_TYPE_UNSPECIFIED,
                               warm.sequences, profile.embedding_workload))}) {
            loaded->recycle(std::move(context));
        }
        return loaded;
    } catch (const std::bad_alloc&) {
        return std::unexpected(LoadFailure{.code = RuntimeError::AllocationFailed, .message = {}});
    }
}

std::expected<std::unique_ptr<LlamaSession>, RuntimeError>
LlamaBackend::create_session(LlamaModel& model, const SessionConfig& config) {
    if (!config.dedicated_context) return std::unexpected(RuntimeError::UnsupportedConfiguration);
    if (config.max_tokens == 0 || config.max_tokens > detail::max_batch_tokens) {
        return std::unexpected(RuntimeError::UnsupportedConfiguration);
    }
    if (config.sequences == 0 || config.sequences > kMaxSequences) {
        return std::unexpected(RuntimeError::UnsupportedConfiguration);
    }
    if (config.max_tokens > detail::max_batch_tokens / config.sequences) {
        return std::unexpected(RuntimeError::UnsupportedConfiguration);
    }

    // A model that declares its pooling is authoritative: a mean-pooled
    // embedding model can still carry classifier labels and stay an embedder.
    // One that declares nothing but ships classifier labels is the patch's
    // rerank case, and only rank pooling makes its output readable. Both facts
    // were settled at load; see LlamaModel::Traits.
    const LlamaModel::Traits& traits = model.traits();
    const bool classifier_workload = traits.profile.classifier_workload;

    // A released session of the same geometry already paid for the expensive
    // part; take its context back before building a new one.
    llama_context_ptr context = model.take_idle_context(config.max_tokens, config.sequences);
    if (!context) {
        context.reset(llama_init_from_model(
            &model.native(), session_params(config.max_tokens,
                                            classifier_workload ? LLAMA_POOLING_TYPE_RANK
                                                                : LLAMA_POOLING_TYPE_UNSPECIFIED,
                                            config.sequences, traits.profile.embedding_workload)));
        if (!context) return std::unexpected(RuntimeError::SessionCreationFailed);
    }

    const bool rerank_workload =
        classifier_workload || traits.declared_pooling == LLAMA_POOLING_TYPE_RANK;
    WorkloadProbe probe{
        .pooling = static_cast<NativePooling>(llama_pooling_type(context.get())),
        .has_encoder = model.has_encoder(),
        .has_decoder = model.has_decoder(),
        .embedding_dimensions = model.embedding_dimensions(),
        .classifier_outputs = rerank_workload ? traits.classifier_outputs : 0,
    };
    const WorkloadInfo workload = make_workload_info(probe);

    const std::optional<Token> decoder_start =
        workload.kind == WorkloadKind::EncoderDecoder ? traits.decoder_start : std::nullopt;

    return std::make_unique<LlamaSession>(std::move(context), model, model.vocab(), workload,
                                          decoder_start, config.max_tokens, config.sequences);
}

static_assert(skippy::RuntimeBackend<LlamaBackend>);
static_assert(skippy::EmbeddingSession<LlamaSession>);
static_assert(skippy::RerankSession<LlamaSession>);
static_assert(skippy::EncoderSession<LlamaSession>);
static_assert(detail::TokenBatch<llama_batch>);

static_assert(static_cast<int>(NativePooling::Unspecified) == LLAMA_POOLING_TYPE_UNSPECIFIED);
static_assert(static_cast<int>(NativePooling::None) == LLAMA_POOLING_TYPE_NONE);
static_assert(static_cast<int>(NativePooling::Mean) == LLAMA_POOLING_TYPE_MEAN);
static_assert(static_cast<int>(NativePooling::Cls) == LLAMA_POOLING_TYPE_CLS);
static_assert(static_cast<int>(NativePooling::Last) == LLAMA_POOLING_TYPE_LAST);
static_assert(static_cast<int>(NativePooling::Rank) == LLAMA_POOLING_TYPE_RANK);

static_assert(sizeof(Token) == sizeof(llama_token));
static_assert(sizeof(SequenceId) == sizeof(llama_seq_id));

} // namespace skippy::llama
