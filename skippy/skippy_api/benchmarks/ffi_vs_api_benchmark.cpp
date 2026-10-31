// Old C ABI (skippy.h) versus the new skippy_api facade, in one process.
//
// Both sides run against the same patched llama.cpp build, the same GGUF and
// the same context geometry (n_ctx = n_batch = n_ubatch = 512, one sequence),
// so what is left in the difference is the interface itself:
//
//   FFI  skippy_session_embed()   patch 0010's decode path
//   API  skippy::embed()          LlamaSession::embed + normalize_embedding
//
// Set SKIPPY_BENCH_MODEL to an embedding GGUF. A reranker adds the Rerank
// pair; without one those two benchmarks skip themselves.
//
// When the model does not implement a workload, the pair skips rather than
// silently measuring something else.

#include <benchmark/benchmark.h>

#include "batch.h"
#include "embed.h"
#include "llama/backend.h"
#include "llama/batch.h"
#include "llama/model_inventory.h"
#include "llama/stage_chain.h"
#include "normalize.h"
#include "rerank.h"
#include "runtime.h"
#include "stage_load.h"
#include "stage_sink.h"
#include "types.h"

#include "llama.h"
#include "skippy.h"
#include "skippy/runtime.h"
#include "skippy/tokenization.h"
#include "skippy/workloads.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

// One physical batch holds the whole input, so this sizes the context too.
// Override with SKIPPY_BENCH_MAX_TOKENS to see how session cost scales.
constexpr std::int32_t kMaxTokens = 512;

std::uint32_t context_tokens() {
    const char* value = std::getenv("SKIPPY_BENCH_MAX_TOKENS");
    if (value == nullptr || *value == '\0') return static_cast<std::uint32_t>(kMaxTokens);
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : static_cast<std::uint32_t>(kMaxTokens);
}

// Long enough to tokenise past kMaxTokens; ASCII so substring cuts stay UTF-8.
const std::string kCorpus = "An embedding model turns text into a fixed width vector. "
                            "The vector is normalised, so cosine similarity is a dot product. "
                            "Rerankers score a query and a document together, in one pass, and "
                            "return a single number per pair. Encoder decoder models encode the "
                            "prompt once and decode from the encoder state. ";

std::string corpus_text() {
    std::string text;
    text.reserve(kCorpus.size() * 16);
    for (int i = 0; i < 16; ++i) {
        text += kCorpus;
    }
    return text;
}

const char* status_name(skippy_status status) {
    switch (status) {
    case SKIPPY_STATUS_OK: return "ok";
    case SKIPPY_STATUS_ERROR: return "error";
    case SKIPPY_STATUS_INVALID_ARGUMENT: return "invalid_argument";
    case SKIPPY_STATUS_UNSUPPORTED: return "unsupported";
    case SKIPPY_STATUS_BUFFER_TOO_SMALL: return "buffer_too_small";
    case SKIPPY_STATUS_IO_ERROR: return "io_error";
    case SKIPPY_STATUS_MODEL_ERROR: return "model_error";
    case SKIPPY_STATUS_RUNTIME_ERROR: return "runtime_error";
    }
    return "unknown";
}

std::string describe(skippy_status status, skippy_error* error) {
    std::string message = status_name(status);
    if (error != nullptr) {
        if (error->message != nullptr) {
            message += ": ";
            message += error->message;
        }
        skippy_error_free(error);
    }
    return message;
}

// ---------------------------------------------------------------------------
// Harness: one loaded model per side, created before timing starts.
// ---------------------------------------------------------------------------

struct Harness {
    bool ready = false;
    std::string error;

    std::size_t dimensions = 0;
    bool rerank_workload = false;
    // A facade that cannot drive this workload is a result, not a setup error:
    // the old side still runs and the pair reports the gap.
    bool api_rerank_supported = true;
    std::string api_rerank_gap;

    std::string model_path;

    // FFI side (patch 0010 C ABI).
    skippy_model* ffi_model = nullptr;
    skippy_session* ffi_session = nullptr;
    // Session-create needs a free lane: the harness session holds this model's
    // only one.
    skippy_model* ffi_session_model = nullptr;
    std::vector<llama_token> ffi_token_buffer;

    // New facade.
    std::optional<skippy::llama::LlamaRuntime> api_runtime;
    std::optional<skippy::Model> api_model;
    std::optional<skippy::Session> api_session;
    // Batch-capable session; its geometry is fixed at creation.
    std::optional<skippy::Session> api_batch_session;
    std::uint32_t api_batch_sequences = 0;

    std::vector<skippy::Token> corpus_tokens;
    std::string last_error;

    Harness() { load(); }

    ~Harness() {
        if (ffi_session != nullptr) {
            skippy_error* error = nullptr;
            skippy_session_free(ffi_session, &error);
            skippy_error_free(error);
        }
        for (skippy_model** model : {&ffi_session_model, &ffi_model}) {
            if (*model != nullptr) {
                skippy_error* error = nullptr;
                skippy_model_free(*model, &error);
                skippy_error_free(error);
            }
        }
    }

    Harness(const Harness&) = delete;
    Harness& operator=(const Harness&) = delete;

    [[nodiscard]] static Harness& instance() {
        static Harness harness;
        return harness;
    }

    // The first `count` tokens of the corpus, which is long enough for
    // kMaxTokens.
    [[nodiscard]] std::span<const skippy::Token> tokens(std::size_t count) const {
        return {corpus_tokens.data(), std::min(count, corpus_tokens.size())};
    }

    // Approximate: cuts the corpus at `characters` bytes.
    [[nodiscard]] static std::string text(std::size_t characters) {
        static const std::string corpus = corpus_text();
        return corpus.substr(0, std::min(characters, corpus.size()));
    }

    bool embed_ffi(std::span<const skippy::Token> tokens, std::vector<float>& output) {
        skippy_error* error = nullptr;
        std::size_t written = 0;
        const auto status =
            skippy_session_embed(ffi_session, reinterpret_cast<const llama_token*>(tokens.data()),
                                 tokens.size(), output.data(), output.size(), &written, &error);
        if (status != SKIPPY_STATUS_OK) {
            last_error = describe(status, error);
            return false;
        }
        return written == output.size();
    }

    bool embed_api(std::span<const skippy::Token> tokens, std::vector<float>& output) {
        const auto result = skippy::embed(
            *api_runtime, *api_session, tokens,
            skippy::EmbeddingOutput<skippy::F32Format>{.values = output, .metadata = {}});
        if (!result) {
            last_error = "embed failed with WorkloadError " +
                         std::to_string(static_cast<int>(result.error().code));
            return false;
        }
        return true;
    }

    bool ensure_batch_session(std::uint32_t sequences) {
        if (api_batch_session && api_batch_sequences >= sequences) return true;
        if (api_batch_session) {
            (void)api_runtime->release_session(*api_batch_session);
            api_batch_session.reset();
        }
        auto session = api_runtime->create_session(
            *api_model, {.max_tokens = context_tokens(), .sequences = sequences});
        if (!session) {
            last_error = "batch create_session failed with RuntimeError " +
                         std::to_string(static_cast<int>(session.error()));
            return false;
        }
        api_batch_session.emplace(std::move(*session));
        api_batch_sequences = sequences;
        return true;
    }

    bool embed_batch_api(std::span<const std::span<const skippy::Token>> inputs,
                         std::span<float> output) {
        const auto result = skippy::embed_batch(
            *api_runtime, *api_batch_session, inputs,
            skippy::EmbeddingOutput<skippy::F32Format>{.values = output, .metadata = {}});
        if (!result) {
            last_error = "embed_batch failed with WorkloadError " +
                         std::to_string(static_cast<int>(result.error().code));
            return false;
        }
        return true;
    }

    bool rerank_batch_api(std::string_view query, std::span<const std::string_view> documents,
                          std::span<float> scores) {
        const auto result =
            skippy::rerank_batch(*api_runtime, *api_batch_session, query, documents, scores);
        if (!result) {
            last_error = "rerank_batch failed with WorkloadError " +
                         std::to_string(static_cast<int>(result.error().code));
            return false;
        }
        return true;
    }

    // A distinct document per index, cut from the corpus.
    [[nodiscard]] static std::string document(std::size_t index, std::size_t characters) {
        static const std::string corpus = corpus_text();
        const std::size_t offset = (index * characters) % (corpus.size() - characters);
        return corpus.substr(offset, characters);
    }

    bool rerank_ffi(std::string_view query, std::string_view document, float& score) {
        skippy_error* error = nullptr;
        std::size_t token_count = 0;
        const std::string query_text(query);
        const std::string document_text(document);
        const auto status = skippy_session_rerank(
            ffi_session, query_text.c_str(), document_text.c_str(), &score, &token_count, &error);
        if (status != SKIPPY_STATUS_OK) {
            last_error = describe(status, error);
            return false;
        }
        return true;
    }

    bool rerank_api(std::string_view query, std::string_view document, float& score) {
        const auto result = skippy::rerank(*api_runtime, *api_session, query, document);
        if (!result) {
            last_error = "rerank failed with WorkloadError " +
                         std::to_string(static_cast<int>(result.error().code));
            return false;
        }
        score = result->score;
        return true;
    }

    void load() {
        const char* path = std::getenv("SKIPPY_BENCH_MODEL");
        if (path != nullptr) model_path = path;
        if (path == nullptr || *path == '\0') {
            error = "SKIPPY_BENCH_MODEL is not set (path to an embedding GGUF)";
            return;
        }

        if (!open_ffi(path)) return;
        if (!open_api(path)) return;
        if (!read_workload()) return;
        if (!tokenise_corpus()) return;
        if (!check_agreement()) return;

        ready = true;
    }

    // The geometry the facade's SessionConfig{max_tokens} produces.
    [[nodiscard]] static skippy_runtime_config ffi_config() {
        skippy_runtime_config config = skippy_runtime_config_default();
        config.ctx_size = static_cast<std::int32_t>(context_tokens());
        config.lane_count = 1;
        config.n_batch = static_cast<std::int32_t>(context_tokens());
        config.n_ubatch = static_cast<std::int32_t>(context_tokens());
        // config_default() leaves 0, which is FLASH_ATTN_TYPE_DISABLED; the
        // product passes AUTO, and so does the facade's default params.
        config.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        // 0 (the struct default) means "no layers offloaded", which would put
        // this side on the CPU while the facade uses llama's 999 default.
        config.n_gpu_layers = 999;
        return config;
    }

    // A second handle with a free lane, so SessionCreate/FFI measures creation
    // rather than the harness session holding the only lane.
    bool ensure_session_model() {
        if (ffi_session_model != nullptr) return true;
        const skippy_runtime_config config = ffi_config();
        skippy_error* error = nullptr;
        if (const auto status =
                skippy_model_open(model_path.c_str(), &config, &ffi_session_model, &error);
            status != SKIPPY_STATUS_OK) {
            this->error = "skippy_model_open (session bench): " + describe(status, error);
            return false;
        }
        return true;
    }

    bool open_ffi(const char* path) {
        const skippy_runtime_config config = ffi_config();

        skippy_error* error = nullptr;
        if (const auto status = skippy_model_open(path, &config, &ffi_model, &error);
            status != SKIPPY_STATUS_OK) {
            this->error = "skippy_model_open: " + describe(status, error);
            return false;
        }
        if (const auto status = skippy_session_create(ffi_model, &ffi_session, &error);
            status != SKIPPY_STATUS_OK) {
            this->error = "skippy_session_create: " + describe(status, error);
            return false;
        }
        return true;
    }

    bool open_api(const char* path) {
        api_runtime.emplace(std::in_place);
        auto model = api_runtime->load_model({.path = path});
        if (!model) {
            error = "load_model failed with RuntimeError " +
                    std::to_string(static_cast<int>(model.error().code));
            return false;
        }
        api_model.emplace(std::move(*model));

        auto session = api_runtime->create_session(*api_model, {.max_tokens = context_tokens()});
        if (!session) {
            error = "create_session failed with RuntimeError " +
                    std::to_string(static_cast<int>(session.error()));
            return false;
        }
        api_session.emplace(std::move(*session));
        return true;
    }

    bool read_workload() {
        skippy_workload_info_v1 info{};
        info.abi_version = SKIPPY_WORKLOAD_INFO_V1_ABI_VERSION;
        info.struct_size = sizeof(info);
        skippy_error* native_error = nullptr;
        if (const auto status = skippy_model_workload_info_v1(ffi_model, &info, &native_error);
            status != SKIPPY_STATUS_OK) {
            this->error = "skippy_model_workload_info_v1: " + describe(status, native_error);
            return false;
        }
        if (info.kind != SKIPPY_WORKLOAD_EMBEDDING && info.kind != SKIPPY_WORKLOAD_RERANK) {
            this->error = "model is neither an embedding nor a rerank model";
            return false;
        }
        rerank_workload = info.kind == SKIPPY_WORKLOAD_RERANK;
        dimensions = info.output_dimensions;

        std::size_t api_dimensions = 0;
        int api_kind = -1;
        int api_pooling = -1;
        std::size_t api_classifiers = 0;
        const auto probed = api_runtime->with_session(
            *api_session, [&](const skippy::llama::LlamaSession& session) {
                const auto& workload = session.workload();
                api_dimensions = workload.output_dimensions;
                api_kind = static_cast<int>(workload.kind);
                api_pooling = static_cast<int>(workload.pooling);
                api_classifiers = workload.classifier_outputs;
                return 0;
            });
        if (!probed) {
            this->error = "api workload probe failed";
            return false;
        }

        std::fprintf(stderr,
                     "ffi_vs_api_benchmark: ffi kind=%d pooling=%d cls=%u dims=%u | "
                     "api kind=%d pooling=%d cls=%zu dims=%zu\n",
                     static_cast<int>(info.kind), static_cast<int>(info.pooling),
                     info.classifier_outputs, info.output_dimensions, api_kind, api_pooling,
                     api_classifiers, api_dimensions);
        if (api_dimensions != dimensions) {
            error = "workload widths disagree: ffi " + std::to_string(dimensions) + ", api " +
                    std::to_string(api_dimensions);
            return false;
        }
        if (!rerank_workload && dimensions == 0) {
            error = "embedding model advertises no width";
            return false;
        }
        return true;
    }

    bool tokenise_corpus() {
        const std::string corpus = corpus_text();
        ffi_token_buffer.resize(2048);
        skippy_error* error = nullptr;
        std::size_t count = 0;
        auto status = skippy_tokenize(ffi_model, corpus.c_str(), true, ffi_token_buffer.data(),
                                      ffi_token_buffer.size(), &count, &error);
        if (status == SKIPPY_STATUS_BUFFER_TOO_SMALL) {
            ffi_token_buffer.resize(count);
            error = nullptr;
            status = skippy_tokenize(ffi_model, corpus.c_str(), true, ffi_token_buffer.data(),
                                     ffi_token_buffer.size(), &count, &error);
        }
        if (status != SKIPPY_STATUS_OK) {
            this->error = "skippy_tokenize: " + describe(status, error);
            return false;
        }
        ffi_token_buffer.resize(count);

        corpus_tokens.resize(count);
        static_assert(sizeof(skippy::Token) == sizeof(llama_token));
        std::memcpy(corpus_tokens.data(), ffi_token_buffer.data(), count * sizeof(llama_token));
        if (count < static_cast<std::size_t>(kMaxTokens)) {
            this->error = "corpus tokenised to only " + std::to_string(count) + " tokens";
            return false;
        }
        return true;
    }

    // Both sides must agree before any timing means anything.
    bool check_agreement() {
        if (rerank_workload) {
            const std::string_view query = "distributed inference across several machines";
            const std::string_view document = "One prompt is answered by many devices.";
            float ffi_score = 0.0F;
            float api_score = 0.0F;
            if (!rerank_ffi(query, document, ffi_score)) {
                this->error = "ffi rerank: " + last_error;
                return false;
            }
            if (!rerank_api(query, document, api_score)) {
                api_rerank_supported = false;
                api_rerank_gap = last_error;
                std::fprintf(stderr,
                             "ffi_vs_api_benchmark: facade cannot rerank this model (%s); "
                             "Rerank/API will skip\n",
                             api_rerank_gap.c_str());
                return true;
            }
            if (std::fabs(ffi_score - api_score) > 1e-3F) {
                this->error = "FFI and API rerank scores disagree by " +
                              std::to_string(std::fabs(ffi_score - api_score));
                return false;
            }
            return true;
        }

        std::vector<float> ffi_output(dimensions);
        std::vector<float> api_output(dimensions);
        const auto sample = tokens(64);
        if (!embed_ffi(sample, ffi_output) || !embed_api(sample, api_output)) {
            error = last_error;
            return false;
        }

        float difference = 0.0F;
        for (std::size_t i = 0; i < dimensions; ++i) {
            difference = std::max(difference, std::fabs(ffi_output[i] - api_output[i]));
        }
        if (difference > 1e-4F) {
            error = "FFI and API embeddings disagree by " + std::to_string(difference);
            return false;
        }
        return true;
    }
};

Harness* ready_harness(benchmark::State& state) {
    Harness& harness = Harness::instance();
    if (!harness.ready) {
        state.SkipWithError(harness.error.c_str());
        return nullptr;
    }
    return &harness;
}

// The old normalize: one accumulator.
#ifdef _MSC_VER
#define SKIPPY_NOINLINE __declspec(noinline)
#else
#define SKIPPY_NOINLINE __attribute__((noinline))
#endif

SKIPPY_NOINLINE void normalize_legacy(const float* source, float* output, std::size_t dimensions) {
    double squared_norm = 0.0;
    for (std::size_t i = 0; i < dimensions; ++i) {
        squared_norm += static_cast<double>(source[i]) * source[i];
    }
    const float scale =
        squared_norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(squared_norm)) : 1.0F;
    for (std::size_t i = 0; i < dimensions; ++i) {
        output[i] = source[i] * scale;
    }
}

// ---------------------------------------------------------------------------
// Embed: the whole call, decode included.
// ---------------------------------------------------------------------------

template <bool UseFfi> void Embed(benchmark::State& state) {
    Harness* harness = ready_harness(state);
    if (harness == nullptr) return;

    const auto count = static_cast<std::size_t>(state.range(0));
    const auto tokens = harness->tokens(count);
    std::vector<float> output(harness->dimensions);

    for (auto _ : state) {
        const bool ok =
            UseFfi ? harness->embed_ffi(tokens, output) : harness->embed_api(tokens, output);
        if (!ok) {
            state.SkipWithError(harness->last_error.c_str());
            return;
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(tokens.size()));
    state.SetLabel(std::to_string(harness->dimensions) + " dims");
}

// ---------------------------------------------------------------------------
// Tokenise: text -> tokens.
// ---------------------------------------------------------------------------

template <bool UseFfi> void Tokenize(benchmark::State& state) {
    Harness* harness = ready_harness(state);
    if (harness == nullptr) return;

    const std::string text = Harness::text(static_cast<std::size_t>(state.range(0)));
    std::size_t token_count = 0;

    for (auto _ : state) {
        if constexpr (UseFfi) {
            skippy_error* error = nullptr;
            auto status = skippy_tokenize(harness->ffi_model, text.c_str(), true,
                                          harness->ffi_token_buffer.data(),
                                          harness->ffi_token_buffer.size(), &token_count, &error);
            if (status == SKIPPY_STATUS_BUFFER_TOO_SMALL) {
                harness->ffi_token_buffer.resize(token_count);
                error = nullptr;
                status = skippy_tokenize(harness->ffi_model, text.c_str(), true,
                                         harness->ffi_token_buffer.data(),
                                         harness->ffi_token_buffer.size(), &token_count, &error);
            }
            if (status != SKIPPY_STATUS_OK) {
                const std::string message = describe(status, error);
                state.SkipWithError(message.c_str());
                return;
            }
        } else {
            std::optional<std::size_t> count;
            const auto tokenized = harness->api_runtime->with_session(
                *harness->api_session, [&](skippy::llama::LlamaSession& session) {
                    const auto result = session.tokenize(text);
                    if (!result) return false;
                    count = result->size();
                    return true;
                });
            if (!tokenized || !*tokenized || !count) {
                state.SkipWithError("session tokenize failed");
                return;
            }
            token_count = *count;
        }
        benchmark::DoNotOptimize(token_count);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel(std::to_string(token_count) + " tokens");
}

// ---------------------------------------------------------------------------
// Batch build: the per-call allocation the facade removes.
//   FFI: llama_batch_init + fill + free, as patch 0010 does.
//   API: fill into the session's batch, already at capacity.
// ---------------------------------------------------------------------------

template <bool UseFfi> void BatchBuild(benchmark::State& state) {
    // No model needed: the ids only have to be the right shape and count.
    const auto count = static_cast<std::int32_t>(state.range(0));
    std::vector<skippy::Token> tokens(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) {
        tokens[static_cast<std::size_t>(i)] = skippy::Token{i + 1000};
    }
    const std::span<const skippy::Token> view = tokens;

    std::optional<skippy::llama::EmbeddingBatchPtr> reusable;
    if constexpr (!UseFfi) {
        auto batch = skippy::llama::allocate_embedding_batch(kMaxTokens);
        if (!batch.has_value()) {
            state.SkipWithError("allocate_embedding_batch failed");
            return;
        }
        reusable.emplace(std::move(*batch));
    }

    for (auto _ : state) {
        if constexpr (UseFfi) {
            llama_batch batch = llama_batch_init(count, 0, 1);
            for (std::int32_t i = 0; i < count; ++i) {
                batch.token[i] = static_cast<llama_token>(tokens[static_cast<std::size_t>(i)]);
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            benchmark::DoNotOptimize(batch.n_tokens);
            llama_batch_free(batch);
        } else {
            skippy::detail::fill_embedding_batch(**reusable, view, skippy::SequenceId{0});
            benchmark::DoNotOptimize((*reusable)->n_tokens);
            benchmark::ClobberMemory();
        }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
}

// ---------------------------------------------------------------------------
// Normalise: 384 and 768 are the widths most embedding models ship.
// ---------------------------------------------------------------------------

template <bool UseFfi> void Normalize(benchmark::State& state) {
    const auto dimensions = static_cast<std::size_t>(state.range(0));
    std::vector<float> source(dimensions);
    std::vector<float> output(dimensions);
    for (std::size_t i = 0; i < dimensions; ++i) {
        source[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0F;
    }

    for (auto _ : state) {
        if constexpr (UseFfi) {
            normalize_legacy(source.data(), output.data(), dimensions);
        } else {
            skippy::detail::normalize_embedding(source.data(), output.data(), dimensions);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(dimensions));
}

// ---------------------------------------------------------------------------
// Session create/release: the model stays loaded.
//   FFI: claims a lane on the model's shared context (cheap).
//   API: builds a dedicated llama_context per session (n_ctx = 512).
// ---------------------------------------------------------------------------

template <bool UseFfi> void SessionCreate(benchmark::State& state) {
    Harness* harness = ready_harness(state);
    if (harness == nullptr) return;

    if constexpr (UseFfi) {
        if (!harness->ensure_session_model()) {
            state.SkipWithError(harness->error.c_str());
            return;
        }
    }

    for (auto _ : state) {
        if constexpr (UseFfi) {
            skippy_session* session = nullptr;
            skippy_error* error = nullptr;
            if (const auto status =
                    skippy_session_create(harness->ffi_session_model, &session, &error);
                status != SKIPPY_STATUS_OK) {
                const std::string message = describe(status, error);
                state.SkipWithError(message.c_str());
                return;
            }
            skippy_session_free(session, &error);
            skippy_error_free(error);
        } else {
            auto session = harness->api_runtime->create_session(*harness->api_model,
                                                                {.max_tokens = context_tokens()});
            if (!session) {
                state.SkipWithError("create_session failed");
                return;
            }
            (void)harness->api_runtime->release_session(*session);
        }
    }

    state.SetItemsProcessed(state.iterations());
}

// ---------------------------------------------------------------------------
// Batched embed: the old ABI decodes one text per call; the facade's
// embed_batch decodes B sequences in one native pass. The setup embeds one row
// twice (single and batched) and refuses to time disagreeing implementations.
// ---------------------------------------------------------------------------

template <bool UseFfi> void BatchEmbed(benchmark::State& state) {
    Harness* harness = ready_harness(state);
    if (harness == nullptr) return;

    const auto sequences = static_cast<std::uint32_t>(state.range(0));
    const auto count = static_cast<std::size_t>(state.range(1));
    const auto tokens = harness->tokens(count);
    std::vector<float> output(static_cast<std::size_t>(sequences) * harness->dimensions);
    // The old ABI writes one embedding per call, so its side reuses one row.
    std::vector<float> single(harness->dimensions);
    std::vector<std::span<const skippy::Token>> inputs(sequences, tokens);

    if constexpr (!UseFfi) {
        if (!harness->ensure_batch_session(sequences)) {
            state.SkipWithError(harness->last_error.c_str());
            return;
        }
        if (!harness->embed_api(tokens, single) || !harness->embed_batch_api(inputs, output)) {
            state.SkipWithError(harness->last_error.c_str());
            return;
        }
        const float difference = std::abs(output.front() - single.front());
        if (difference > 1e-4F) {
            const std::string message =
                "batched row differs from single embed by " + std::to_string(difference);
            state.SkipWithError(message.c_str());
            return;
        }
    }

    for (auto _ : state) {
        if constexpr (UseFfi) {
            for (std::uint32_t sequence = 0; sequence < sequences; ++sequence) {
                if (!harness->embed_ffi(tokens, single)) {
                    state.SkipWithError(harness->last_error.c_str());
                    return;
                }
            }
        } else if (!harness->embed_batch_api(inputs, output)) {
            state.SkipWithError(harness->last_error.c_str());
            return;
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(sequences));
}

// ---------------------------------------------------------------------------
// Batched rerank: one query against B documents. The old ABI scores one
// document per call; the facade composes and decodes the whole group at once.
// ---------------------------------------------------------------------------

template <bool UseFfi> void RerankBatch(benchmark::State& state) {
    Harness* harness = ready_harness(state);
    if (harness == nullptr) return;
    if (!harness->rerank_workload) {
        state.SkipWithError("model is not a reranker; point SKIPPY_BENCH_MODEL at one");
        return;
    }

    const auto documents_count = static_cast<std::size_t>(state.range(0));
    const std::string_view query = "What is the capital of France?";
    std::vector<std::string> owned;
    owned.reserve(documents_count);
    for (std::size_t index = 0; index < documents_count; ++index) {
        owned.push_back(Harness::document(index, 200));
    }
    std::vector<std::string_view> documents;
    documents.reserve(owned.size());
    for (const auto& document : owned) {
        documents.emplace_back(document);
    }
    std::vector<float> scores(documents_count);

    if constexpr (!UseFfi) {
        if (!harness->ensure_batch_session(static_cast<std::uint32_t>(documents_count))) {
            state.SkipWithError(harness->last_error.c_str());
            return;
        }
        // Agreement gate: the first document must score the same either way.
        float single = 0.0F;
        if (!harness->rerank_api(query, documents.front(), single) ||
            !harness->rerank_batch_api(query, documents, scores)) {
            state.SkipWithError(harness->last_error.c_str());
            return;
        }
        if (std::abs(scores.front() - single) > 1e-3F) {
            const std::string message = "batched score " + std::to_string(scores.front()) +
                                        " differs from single " + std::to_string(single);
            state.SkipWithError(message.c_str());
            return;
        }
    }

    for (auto _ : state) {
        if constexpr (UseFfi) {
            for (const auto document : documents) {
                float score = 0.0F;
                if (!harness->rerank_ffi(query, document, score)) {
                    state.SkipWithError(harness->last_error.c_str());
                    return;
                }
                scores.front() = score;
            }
        } else if (!harness->rerank_batch_api(query, documents, scores)) {
            state.SkipWithError(harness->last_error.c_str());
            return;
        }
        benchmark::DoNotOptimize(scores.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(documents_count));
}

// ---------------------------------------------------------------------------
// Model load: the old open builds the model *and* its shared context; the
// facade's load_model builds only the model, so the context cost shows up in
// SessionCreate instead. This is the "moved, not added" measurement.
// ---------------------------------------------------------------------------

template <bool UseFfi> void ModelLoad(benchmark::State& state) {
    Harness* harness = ready_harness(state);
    if (harness == nullptr) return;

    for (auto _ : state) {
        if constexpr (UseFfi) {
            const skippy_runtime_config config = Harness::ffi_config();
            skippy_model* model = nullptr;
            skippy_error* error = nullptr;
            if (skippy_model_open(harness->model_path.c_str(), &config, &model, &error) !=
                SKIPPY_STATUS_OK) {
                const std::string message = describe(SKIPPY_STATUS_MODEL_ERROR, error);
                state.SkipWithError(message.c_str());
                return;
            }
            skippy_model_free(model, &error);
            skippy_error_free(error);
        } else {
            auto model = harness->api_runtime->load_model({.path = harness->model_path});
            if (!model) {
                state.SkipWithError("load_model failed");
                return;
            }
            // optional<RuntimeError>: nullopt is the success case.
            if (harness->api_runtime->unload_model(*model).has_value()) {
                state.SkipWithError("unload_model failed");
                return;
            }
        }
    }

    state.SetItemsProcessed(state.iterations());
}

// ---------------------------------------------------------------------------
// Session create when the retained-context pool cannot help: three geometries
// rotate through a two-slot pool, so every call builds a context.
// ---------------------------------------------------------------------------

template <bool UseFfi> void SessionCreateCold(benchmark::State& state) {
    Harness* harness = ready_harness(state);
    if (harness == nullptr) return;

    if constexpr (UseFfi) {
        if (!harness->ensure_session_model()) {
            state.SkipWithError(harness->error.c_str());
            return;
        }
    }

    static constexpr std::array<std::uint32_t, 3> cold_sizes{512, 480, 448};
    // Survives repetitions: each repetition re-enters this function, and the
    // rotation only defeats the pool if it keeps moving.
    static std::size_t turn = 0;

    for (auto _ : state) {
        if constexpr (UseFfi) {
            skippy_session* session = nullptr;
            skippy_error* error = nullptr;
            if (skippy_session_create(harness->ffi_session_model, &session, &error) !=
                SKIPPY_STATUS_OK) {
                const std::string message = describe(SKIPPY_STATUS_RUNTIME_ERROR, error);
                state.SkipWithError(message.c_str());
                return;
            }
            skippy_session_free(session, &error);
            skippy_error_free(error);
        } else {
            auto session = harness->api_runtime->create_session(
                *harness->api_model, {.max_tokens = cold_sizes[turn % cold_sizes.size()]});
            if (!session) {
                state.SkipWithError("create_session failed");
                return;
            }
            if (harness->api_runtime->release_session(*session).has_value()) {
                state.SkipWithError("release_session failed");
                return;
            }
        }
        ++turn;
    }

    state.SetItemsProcessed(state.iterations());
}

// ---------------------------------------------------------------------------
// Rerank: query + document in, one score out. Needs a reranker GGUF.
// ---------------------------------------------------------------------------

template <bool UseFfi> void Rerank(benchmark::State& state) {
    Harness* harness = ready_harness(state);
    if (harness == nullptr) return;
    if (!harness->rerank_workload) {
        state.SkipWithError("model is not a reranker; point SKIPPY_BENCH_MODEL at one");
        return;
    }
    if constexpr (!UseFfi) {
        if (!harness->api_rerank_supported) {
            state.SkipWithError(harness->api_rerank_gap.c_str());
            return;
        }
    }

    const std::string query = "distributed inference across several machines";
    const std::string document = "A mesh of GPUs shares the model, so one prompt "
                                 "is answered by many devices.";

    for (auto _ : state) {
        float score = 0.0F;
        const bool ok = UseFfi ? harness->rerank_ffi(query, document, score)
                               : harness->rerank_api(query, document, score);
        if (!ok) {
            state.SkipWithError(harness->last_error.c_str());
            return;
        }
        benchmark::DoNotOptimize(&score);
    }

    state.SetItemsProcessed(state.iterations());
}

// ---------------------------------------------------------------------------
// stage_chain: hand-rolled C-ABI chaining versus the StageChain facade.
//
// Both sides run the same two-stage prefill over the same model. The FFI side
// is what a C consumer must write: one model and session per range, and every
// prefill hands the previous stage's frame in by hand. The API side is
// StageChain::prefill with a sink. The measured delta is the driving, not the
// planning: both sides' runtime configs come from the same plan lease, built
// once before timing.
// ---------------------------------------------------------------------------

/// The raw C-ABI chain: owns one stage per range and the buffers each stage's
/// export lands in, reused across prefill calls like the facade's
/// PreparedExport.
class ManualChain {
public:
    ManualChain() = default;
    ~ManualChain() { release(); }
    ManualChain(const ManualChain&) = delete;
    ManualChain& operator=(const ManualChain&) = delete;

    [[nodiscard]] bool open(const std::string& path, std::string& error) {
        auto inventory = skippy::probe::read_inventory(path);
        if (!inventory) {
            error = "gguf: " + inventory.error();
            return false;
        }
        const auto layers = skippy::probe::layer_count_from(*inventory);
        const std::int32_t half = layers / 2;
        if (half < 1) {
            error = "model has fewer than two layers to split";
            return false;
        }
        ranges_ = {skippy::LayerRange{0, half}, skippy::LayerRange{half, layers}};

        skippy::llama::LlamaBackend backend;
        auto lease = backend.plan_model_stages(path, ranges_);
        if (!lease) {
            error = "plan: " + lease.error().message;
            return false;
        }
        if (lease->configs.size() != ranges_.size()) {
            error = "planner returned " + std::to_string(lease->configs.size()) + " configs for " +
                    std::to_string(ranges_.size()) + " ranges";
            return false;
        }
        lease_ = std::move(*lease);
        stages_.resize(ranges_.size());
        for (std::size_t index = 0; index < ranges_.size(); ++index) {
            if (!open_stage(path, lease_.configs[index], stages_[index], error)) return false;
        }
        return true;
    }

    /// One prefill through every stage, carrying the frames by hand. The
    /// buffers were sized at warmup, so a BUFFER_TOO_SMALL here is a setup
    /// miss, not a state to retry in the timed loop.
    [[nodiscard]] bool prefill(std::span<const llama_token> tokens, std::string& error) {
        const skippy_activation_desc* input_desc = nullptr;
        const std::uint8_t* input = nullptr;
        for (Stage& stage : stages_) {
            std::size_t needed = 0;
            skippy_error* failure = nullptr;
            const auto status = skippy_prefill_chunk_frame(
                stage.session, tokens.data(), tokens.size(), input_desc, input, &stage.out_desc,
                stage.out.data(), stage.out.size(), &needed, &failure);
            if (status != SKIPPY_STATUS_OK) {
                error = describe(status, failure);
                return false;
            }
            input_desc = &stage.out_desc;
            input = stage.out.data();
        }
        return true;
    }

    /// The terminal stage's frame, for the agreement check against the facade.
    [[nodiscard]] std::span<const std::uint8_t> terminal() const {
        return stages_.empty() ? std::span<const std::uint8_t>{} : stages_.back().out;
    }

    /// The first stage's model, for callers that need a tokenizer.
    [[nodiscard]] skippy_model* first_model() const { return stages_.front().model; }

    /// The ranges this chain covers.
    [[nodiscard]] std::span<const skippy::LayerRange> ranges() const { return ranges_; }

    /// Clears every stage's session so the timed loop re-prefills from
    /// position zero instead of overflowing the context window.
    [[nodiscard]] bool reset(std::string& error) {
        for (Stage& stage : stages_) {
            skippy_error* failure = nullptr;
            if (skippy_session_reset(stage.session, &failure) != SKIPPY_STATUS_OK) {
                error = describe(SKIPPY_STATUS_ERROR, failure);
                return false;
            }
        }
        return true;
    }

    /// Sizes every stage's export buffer with one warmup prefill, so the
    /// timed loop never reallocates.
    [[nodiscard]] bool warm(std::span<const llama_token> tokens, std::string& error) {
        const skippy_activation_desc* input_desc = nullptr;
        const std::uint8_t* input = nullptr;
        for (Stage& stage : stages_) {
            std::size_t needed = stage.out.size();
            for (int attempt = 0; attempt < 3; ++attempt) {
                if (stage.out.size() < needed) stage.out.resize(needed);
                needed = 0;
                skippy_error* failure = nullptr;
                const auto status = skippy_prefill_chunk_frame(
                    stage.session, tokens.data(), tokens.size(), input_desc, input, &stage.out_desc,
                    stage.out.data(), stage.out.size(), &needed, &failure);
                if (status == SKIPPY_STATUS_OK) {
                    stage.out.resize(needed);
                    break;
                }
                if (status != SKIPPY_STATUS_BUFFER_TOO_SMALL || needed == 0) {
                    error = describe(status, failure);
                    return false;
                }
                if (failure != nullptr) skippy_error_free(failure);
            }
            input_desc = &stage.out_desc;
            input = stage.out.data();
        }
        return true;
    }

private:
    struct Stage {
        skippy_model* model = nullptr;
        skippy_session* session = nullptr;
        skippy_activation_desc out_desc{};
        std::vector<std::uint8_t> out;
    };

    void release() {
        for (Stage& stage : stages_) {
            if (stage.session != nullptr) skippy_session_free(stage.session, nullptr);
            if (stage.model != nullptr) skippy_model_free(stage.model, nullptr);
        }
        stages_.clear();
    }

    [[nodiscard]] static bool open_stage(const std::string& path,
                                         const skippy::StageLoadConfig& config, Stage& stage,
                                         std::string& error) {
        stage.out.resize(1U << 20);
        // The config's views run into the plan's flat identity arena, which
        // does not terminate individual strings: copy to owning storage
        // before handing the loader `const char*`s.
        const auto owned = [](std::span<const std::string_view> values) {
            std::vector<std::string> copy;
            copy.reserve(values.size());
            for (const std::string_view value : values)
                copy.emplace_back(value);
            return copy;
        };
        const std::vector<std::string> residents = owned(config.resident_identities);
        const std::vector<std::string> imports = owned(config.activation_import_identities);
        const std::vector<std::string> import_bindings = owned(config.activation_import_bindings);
        const std::vector<std::string> exports = owned(config.activation_export_identities);
        const std::vector<std::string> export_bindings = owned(config.activation_export_bindings);
        std::vector<const char*> resident_names;
        resident_names.reserve(residents.size());
        for (const std::string& name : residents)
            resident_names.push_back(name.c_str());
        std::vector<const char*> import_names;
        for (const std::string& name : imports)
            import_names.push_back(name.c_str());
        std::vector<const char*> import_binding_names;
        for (const std::string& name : import_bindings)
            import_binding_names.push_back(name.c_str());
        std::vector<const char*> export_names;
        for (const std::string& name : exports)
            export_names.push_back(name.c_str());
        std::vector<const char*> export_binding_names;
        for (const std::string& name : export_bindings)
            export_binding_names.push_back(name.c_str());
        const std::string contract{config.execution_contract};

        auto runtime = skippy_runtime_config_default();
        runtime.stage_index = config.stage_index;
        runtime.layer_start = config.layers.start;
        runtime.layer_end = config.layers.end;
        runtime.ctx_size = 512;
        runtime.n_batch = 512;
        runtime.n_ubatch = 512;
        runtime.n_gpu_layers = 999;
        runtime.load_mode = SKIPPY_LOAD_MODE_LAYER_PACKAGE;
        runtime.execution_contract = contract.c_str();
        runtime.resident_tensor_names = resident_names.data();
        runtime.resident_tensor_name_count = resident_names.size();
        runtime.activation_import_identities = import_names.empty() ? nullptr : import_names.data();
        runtime.activation_import_identity_count = import_names.size();
        runtime.activation_import_bindings =
            import_binding_names.empty() ? nullptr : import_binding_names.data();
        runtime.activation_export_identities = export_names.empty() ? nullptr : export_names.data();
        runtime.activation_export_identity_count = export_names.size();
        runtime.activation_export_bindings =
            export_binding_names.empty() ? nullptr : export_binding_names.data();

        skippy_error* failure = nullptr;
        if (skippy_model_open(path.c_str(), &runtime, &stage.model, &failure) != SKIPPY_STATUS_OK) {
            error = describe(SKIPPY_STATUS_ERROR, failure);
            return false;
        }
        if (skippy_session_create(stage.model, &stage.session, &failure) != SKIPPY_STATUS_OK) {
            error = describe(SKIPPY_STATUS_ERROR, failure);
            return false;
        }
        return true;
    }

    std::vector<skippy::LayerRange> ranges_;
    skippy::llama::PlanLease lease_;
    std::vector<Stage> stages_;
};

/// The whole stage_chain A/B: setup, agreement check, timed loop.
void stage_chain_body(benchmark::State& state, bool use_ffi) {
    const char* variable = std::getenv("SKIPPY_BENCH_MODEL");
    if (variable == nullptr || *variable == '\0') {
        state.SkipWithError("SKIPPY_BENCH_MODEL is not set");
        return;
    }
    const std::string path{variable};

    // Both sides share the tokens: tokenize once through the manual chain's
    // first stage, exactly as a C consumer would.
    ManualChain manual;
    std::string error;
    if (!manual.open(path, error)) {
        state.SkipWithError(error.c_str());
        return;
    }
    std::vector<llama_token> tokens(512);
    {
        const std::string text = Harness::text(512);
        std::size_t count = 0;
        skippy_error* failure = nullptr;
        auto status = skippy_tokenize(manual.first_model(), text.c_str(), true, tokens.data(),
                                      tokens.size(), &count, &failure);
        if (status == SKIPPY_STATUS_BUFFER_TOO_SMALL) {
            if (failure != nullptr) skippy_error_free(failure);
            tokens.resize(count);
            failure = nullptr;
            status = skippy_tokenize(manual.first_model(), text.c_str(), true, tokens.data(),
                                     tokens.size(), &count, &failure);
        }
        if (status != SKIPPY_STATUS_OK) {
            state.SkipWithError(describe(status, failure).c_str());
            return;
        }
        tokens.resize(count);
    }

    auto chain = skippy::llama::StageChain::open(path, manual.ranges());
    if (!chain) {
        state.SkipWithError(("chain open: " + chain.error().message).c_str());
        return;
    }

    // Warm both, then refuse to time a pair that disagrees: the frame is the
    // correctness contract, so a silent mismatch would make the ratio lie.
    if (!manual.warm(tokens, error)) {
        state.SkipWithError(error.c_str());
        return;
    }
    skippy::CopyFrameSink sink;
    if (auto failure = chain->prefill(tokens, sink); !failure.has_value()) {
        state.SkipWithError(failure.error().message.c_str());
        return;
    }
    const std::span<const std::uint8_t> manual_frame = manual.terminal();
    if (sink.frame.has_value() &&
        (sink.frame->size() != manual_frame.size() ||
         std::memcmp(sink.frame->data(), manual_frame.data(), manual_frame.size()) != 0)) {
        state.SkipWithError("chain sides disagree on the terminal frame");
        return;
    }

    if (use_ffi) {
        for (auto _ : state) {
            if (!manual.reset(error)) {
                state.SkipWithError(error.c_str());
                return;
            }
            if (!manual.prefill(tokens, error)) {
                state.SkipWithError(error.c_str());
                return;
            }
        }
    } else {
        skippy::DiscardSink discard;
        for (auto _ : state) {
            if (auto failure = chain->reset(); failure.has_value()) {
                state.SkipWithError(failure->message.c_str());
                return;
            }
            if (auto failure = chain->prefill(tokens, discard); !failure.has_value()) {
                state.SkipWithError(failure.error().message.c_str());
                return;
            }
        }
    }
    state.SetItemsProcessed(state.iterations());
}

void stage_chain_ffi(benchmark::State& state) { stage_chain_body(state, true); }
void stage_chain_api(benchmark::State& state) { stage_chain_body(state, false); }

void token_args(benchmark::internal::Benchmark* benchmark) {
    benchmark->Arg(1)->Arg(2)->Arg(8)->Arg(32)->Arg(128)->Arg(512)->UseRealTime();
}

void character_args(benchmark::internal::Benchmark* benchmark) {
    benchmark->Arg(64)->Arg(512)->Arg(4096)->UseRealTime();
}

void dimension_args(benchmark::internal::Benchmark* benchmark) { benchmark->Arg(384)->Arg(768); }

} // namespace

BENCHMARK_TEMPLATE(Embed, true)->Name("Embed/FFI")->Apply(token_args);
BENCHMARK_TEMPLATE(Embed, false)->Name("Embed/API")->Apply(token_args);

BENCHMARK_TEMPLATE(Tokenize, true)->Name("Tokenize/FFI")->Apply(character_args);
BENCHMARK_TEMPLATE(Tokenize, false)->Name("Tokenize/API")->Apply(character_args);

BENCHMARK_TEMPLATE(BatchBuild, true)->Name("BatchBuild/FFI")->Apply(token_args)->UseRealTime();
BENCHMARK_TEMPLATE(BatchBuild, false)->Name("BatchBuild/API")->Apply(token_args)->UseRealTime();

BENCHMARK_TEMPLATE(Normalize, true)->Name("Normalize/FFI")->Apply(dimension_args);
BENCHMARK_TEMPLATE(Normalize, false)->Name("Normalize/API")->Apply(dimension_args);

BENCHMARK_TEMPLATE(SessionCreate, true)
    ->Name("SessionCreate/FFI")
    ->Iterations(1)
    ->Repetitions(20)
    ->UseRealTime();
BENCHMARK_TEMPLATE(SessionCreate, false)
    ->Name("SessionCreate/API")
    ->Iterations(1)
    ->Repetitions(20)
    ->UseRealTime();

BENCHMARK_TEMPLATE(SessionCreateCold, true)
    ->Name("SessionCreateCold/FFI")
    ->Iterations(1)
    ->Repetitions(20)
    ->UseRealTime();
BENCHMARK_TEMPLATE(SessionCreateCold, false)
    ->Name("SessionCreateCold/API")
    ->Iterations(1)
    ->Repetitions(20)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BatchEmbed, true)
    ->Name("BatchEmbed/FFI")
    ->ArgsProduct({{8, 32}, {8, 128}})
    ->UseRealTime();
BENCHMARK_TEMPLATE(BatchEmbed, false)
    ->Name("BatchEmbed/API")
    ->ArgsProduct({{8, 32}, {8, 128}})
    ->UseRealTime();

BENCHMARK_TEMPLATE(RerankBatch, true)->Name("RerankBatch/FFI")->Arg(8)->UseRealTime();
BENCHMARK_TEMPLATE(RerankBatch, false)->Name("RerankBatch/API")->Arg(8)->UseRealTime();

BENCHMARK_TEMPLATE(ModelLoad, true)
    ->Name("ModelLoad/FFI")
    ->Iterations(1)
    ->Repetitions(5)
    ->UseRealTime();
BENCHMARK_TEMPLATE(ModelLoad, false)
    ->Name("ModelLoad/API")
    ->Iterations(1)
    ->Repetitions(5)
    ->UseRealTime();

BENCHMARK_TEMPLATE(Rerank, true)->Name("Rerank/FFI")->UseRealTime();
BENCHMARK_TEMPLATE(Rerank, false)->Name("Rerank/API")->UseRealTime();

BENCHMARK(stage_chain_ffi)->Name("stage_chain/FFI")->UseRealTime();
BENCHMARK(stage_chain_api)->Name("stage_chain/API")->UseRealTime();

// Our own main so a missing model is one line on stderr, and the llama-free
// benchmarks still run: BatchBuild and Normalize need no GGUF.
int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    if (const Harness& harness = Harness::instance(); !harness.ready) {
        std::fprintf(stderr, "ffi_vs_api_benchmark: model-backed benchmarks will skip: %s\n",
                     harness.error.c_str());
        std::fprintf(stderr, "  example: SKIPPY_BENCH_MODEL=~/models/embed.gguf %s\n", argv[0]);
    }
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
