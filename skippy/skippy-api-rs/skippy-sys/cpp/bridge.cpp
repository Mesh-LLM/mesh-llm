#include "skippy-sys/cpp/bridge.h"

#include "llama/stage_chain.h"

#include "skippy-sys/src/lib.rs.h"

#include "embed.h"
#include "encode.h"
#include "family.h"
#include "llama/backend.h"
#include "llama/model_inventory.h"
#include "llama/stage_planner.h"
#include "rerank.h"
#include "types.h"
#include "workload.h"
#include "workload_error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <expected>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace skippy::ffi {
namespace {

constexpr RuntimeStatus to_status(RuntimeError error) noexcept {
    switch (error) {
    case RuntimeError::InvalidModel: return RuntimeStatus::InvalidModel;
    case RuntimeError::InvalidSession: return RuntimeStatus::InvalidSession;
    case RuntimeError::ModelInUse: return RuntimeStatus::ModelInUse;
    case RuntimeError::AllocationFailed: return RuntimeStatus::AllocationFailed;
    case RuntimeError::LoadFailed: return RuntimeStatus::LoadFailed;
    case RuntimeError::SessionCreationFailed: return RuntimeStatus::SessionCreationFailed;
    case RuntimeError::UnsupportedConfiguration: return RuntimeStatus::UnsupportedConfiguration;
    case RuntimeError::CapacityExceeded: return RuntimeStatus::CapacityExceeded;
    case RuntimeError::InvalidBackendResult: return RuntimeStatus::InvalidBackendResult;
    }
    return RuntimeStatus::InternalError;
}

constexpr WorkloadStatus to_status(const WorkloadFailure& failure) noexcept {
    const WorkloadError error = failure.code;
    switch (error) {
    case WorkloadError::InvalidSession: return WorkloadStatus::InvalidSession;
    case WorkloadError::UnsupportedWorkload: return WorkloadStatus::UnsupportedWorkload;
    case WorkloadError::UnsupportedClassifierCount:
        return WorkloadStatus::UnsupportedClassifierCount;
    case WorkloadError::EmptyInput: return WorkloadStatus::EmptyInput;
    case WorkloadError::TooManyTokens: return WorkloadStatus::TooManyTokens;
    case WorkloadError::TooManySequences: return WorkloadStatus::TooManySequences;
    case WorkloadError::TokenizationFailed: return WorkloadStatus::TokenizationFailed;
    case WorkloadError::NoEmbeddingDimensions: return WorkloadStatus::NoEmbeddingDimensions;
    case WorkloadError::OutputTooSmall: return WorkloadStatus::OutputTooSmall;
    case WorkloadError::AllocationFailed: return WorkloadStatus::AllocationFailed;
    case WorkloadError::DecodeFailed: return WorkloadStatus::DecodeFailed;
    case WorkloadError::EncodeFailed: return WorkloadStatus::EncodeFailed;
    case WorkloadError::NoDecoderStartToken: return WorkloadStatus::NoDecoderStartToken;
    case WorkloadError::NoOutput: return WorkloadStatus::NoOutput;
    case WorkloadError::DimensionMismatch: return WorkloadStatus::DimensionMismatch;
    }
    return WorkloadStatus::InternalError;
}

/// The backend's own failure text, crossing the boundary lossily: native
/// failure text is arbitrary bytes, and an empty message is an empty string.
inline rust::String error_text(const WorkloadFailure& failure) noexcept {
    return rust::String::lossy(failure.message);
}

template <typename Status> struct boundary_status;
template <> struct boundary_status<RuntimeStatus> {
    static constexpr auto allocation = RuntimeStatus::AllocationFailed;
    static constexpr auto internal = RuntimeStatus::InternalError;
};
template <> struct boundary_status<WorkloadStatus> {
    static constexpr auto allocation = WorkloadStatus::AllocationFailed;
    static constexpr auto internal = WorkloadStatus::InternalError;
};
// A family probe has no failure status to spell: any failure, allocation
// included, reads as the unknown family.
template <> struct boundary_status<ModelFamily> {
    static constexpr auto allocation = ModelFamily::Unknown;
    static constexpr auto internal = ModelFamily::Unknown;
};

template <typename Result> struct status_of {
    using type = decltype(Result::status);
};
template <typename Result>
    requires std::is_enum_v<Result>
struct status_of<Result> {
    using type = Result;
};

template <typename Result>
constexpr Result failed_with(typename status_of<Result>::type status) noexcept {
    if constexpr (std::is_enum_v<Result>) {
        return status;
    } else {
        Result result{};
        result.status = status;
        return result;
    }
}

// Last line of defence: nothing below may unwind into Rust.
template <typename Result, std::invocable Body>
    requires std::same_as<std::invoke_result_t<Body>, Result>
Result guarded(Body&& body) noexcept {
    using Boundary = boundary_status<typename status_of<Result>::type>;
    try {
        return std::forward<Body>(body)();
    } catch (const std::bad_alloc&) {
        return failed_with<Result>(Boundary::allocation);
    } catch (...) {
        return failed_with<Result>(Boundary::internal);
    }
}

// The staged and plan surface reports failure in-band - ok=false plus a
// message - so guarded<Result> cannot shape it. lossy never throws: it
// crosses into Rust's from_utf8_lossy, so a message with arbitrary bytes
// from a hostile file survives instead of vanishing into a dropped catch.
template <typename Result> Result in_band_failed(const std::string& message) noexcept {
    Result result{};
    result.ok = false;
    result.error = rust::String::lossy(message);
    return result;
}
// The (where, what) spelling keeps the concatenation inside its own catch:
// a caller in a catch arm of a noexcept function must not allocate inline.
template <typename Result> Result in_band_failed(const char* where, const char* what) noexcept {
    try {
        return in_band_failed<Result>(std::string(where) + ": " + what);
    } catch (...) {
        Result bare{};
        bare.ok = false;
        return bare;
    }
}

// The out-of-memory arm must not allocate: in_band_failed's lossy conversion
// builds a rust::String, and under bad_alloc that second allocation is a
// terminate waiting inside noexcept - the run_pass trap. The flag alone is
// the contract here, so the message stays empty.
template <typename Result> Result in_band_oom() noexcept {
    Result result{};
    result.ok = false;
    return result;
}

// The in-band mirror of guarded<>: the result reports failure through
// ok=false plus a message, so the catch arms write that message instead of
// a status. Every catch ladder on the staged and chain surface collapses
// into one call; the prefix names the entry point that failed.
template <typename Result, std::invocable Body>
    requires std::same_as<std::invoke_result_t<Body>, Result>
Result guarded_message(const char* where, Body&& body) noexcept {
    try {
        return std::forward<Body>(body)();
    } catch (const std::bad_alloc&) {
        return in_band_oom<Result>();
    } catch (const std::exception& failure) {
        try {
            return in_band_failed<Result>(where, failure.what());
        } catch (...) {
            return in_band_oom<Result>();
        }
    } catch (...) {
        try {
            return in_band_failed<Result>(where, "unknown failure");
        } catch (...) {
            return in_band_oom<Result>();
        }
    }
}

constexpr NativePooling to_native(skippy::NativePooling pooling) noexcept {
    return static_cast<NativePooling>(std::to_underlying(pooling));
}

constexpr NativePooling to_native(std::optional<Pooling> pooling) noexcept {
    return pooling ? static_cast<NativePooling>(std::to_underlying(*pooling))
                   : NativePooling::Unspecified;
}

constexpr WorkloadInfo to_info(const skippy::WorkloadInfo& info) noexcept {
    return WorkloadInfo{
        .kind = static_cast<WorkloadKind>(std::to_underlying(info.kind)),
        .pooling = to_native(info.pooling),
        .embedding_pooling = to_native(info.embedding_pooling),
        .output_dimensions = info.output_dimensions,
        .classifier_outputs = info.classifier_outputs,
        .has_encoder = info.has_encoder,
        .has_decoder = info.has_decoder,
    };
}

std::string_view as_view(rust::Str text) noexcept { return {text.data(), text.size()}; }

// A Rust &[i32] is layout-compatible with Token; `scratch` is written only on
// standard libraries without start_lifetime_as.
std::span<const Token> as_tokens(rust::Slice<const std::int32_t> tokens,
                                 std::vector<Token>& scratch) {
    return detail::tokens_from_native(std::span<const std::int32_t>{tokens.data(), tokens.size()},
                                      scratch);
}

// Moves a freshly minted handle onto the heap for Rust. If that allocation
// fails the handle has not been moved from yet, so the resource is given back.
ModelResult to_result(llama::LlamaRuntime& runtime,
                      std::expected<Model, skippy::LoadFailure> loaded) noexcept {
    if (!loaded) {
        // The loader's reason travels in the failure value; it crosses here,
        // not through state on the runtime.
        return {.status = to_status(loaded.error().code),
                .model = nullptr,
                .error = rust::String::lossy(loaded.error().message)};
    }
    std::unique_ptr<Model> model{new (std::nothrow) Model(std::move(*loaded))};
    if (!model) {
        (void)runtime.unload_model(*loaded);
        return {.status = RuntimeStatus::AllocationFailed, .model = nullptr, .error = {}};
    }
    return {.status = RuntimeStatus::Ok, .model = std::move(model), .error = {}};
}

SessionResult to_result(llama::LlamaRuntime& runtime,
                        std::expected<Session, RuntimeError> created) noexcept {
    if (!created) return {.status = to_status(created.error()), .session = nullptr};
    std::unique_ptr<Session> session{new (std::nothrow) Session(std::move(*created))};
    if (!session) {
        (void)runtime.release_session(*created);
        return {.status = RuntimeStatus::AllocationFailed, .session = nullptr};
    }
    return {.status = RuntimeStatus::Ok, .session = std::move(session)};
}

template <typename Status>
/// The runtime reports "did it fail, and why" as an unexpected error value.
Status to_plain_status(std::expected<void, RuntimeError> result) noexcept {
    return result.has_value() ? Status::Ok : to_status(result.error());
}

} // namespace

struct Bridge::Impl {
    llama::LlamaRuntime runtime{std::in_place};
    std::vector<Token> token_scratch;
};

Bridge::Bridge() : impl_(std::make_unique<Impl>()) {}
Bridge::~Bridge() = default;

std::unique_ptr<Bridge> new_bridge() noexcept {
    try {
        return std::make_unique<Bridge>();
    } catch (...) {
        return nullptr;
    }
}

namespace {

rust::String hex_of(std::string_view value) {
    static constexpr std::string_view kHex = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() * 2);
    for (const char byte : value) {
        const auto raw = std::bit_cast<unsigned char>(byte);
        out.push_back(kHex[(raw >> 4U) & 0xFU]);
        out.push_back(kHex[raw & 0xFU]);
    }
    return rust::String::lossy(std::move(out));
}

/// Splits [0, layers) into `stage_count` contiguous ranges. Both inputs are
/// bounded before any reserve: a caller-supplied stage count past the
/// model's layer count is a typed rejection, not a giant allocation or a
/// pile of zero-width stages (never allocate blind — AGENTS.md).
} // namespace

PlanStagesResult Bridge::plan_stages(rust::Str path, std::uint32_t stage_count) const noexcept {
    return guarded_message<PlanStagesResult>("plan_stages", [&]() -> PlanStagesResult {
        PlanStagesResult result{};
        const auto fail = [](const std::string& message) {
            return in_band_failed<PlanStagesResult>(message);
        };
        if (stage_count == 0) return fail("a plan needs at least one stage");

        const std::string model_path(path);
        auto inventory = skippy::probe::read_inventory(model_path);
        if (!inventory) return fail(inventory.error());
        const auto layers = skippy::probe::layer_count_from(*inventory);
        const auto ranges = split_layers(layers, stage_count);
        if (!ranges) return fail(ranges.error());

        auto lease =
            skippy::llama::plan_model_stages(impl_->runtime.backend(), model_path, *ranges);
        if (!lease) return fail(lease.error().message);
        // The lease pins the plans these configs borrow from; it lives as long as
        // the loop below and the strings are copied into StagePlanInfo anyway.

        for (const skippy::StageLoadConfig& config : lease->configs) {
            StagePlanInfo info{};
            info.stage_index = config.stage_index;
            info.profile_id_hex = hex_of(config.profile_id);
            info.graph_identity_hex = hex_of(config.graph_identity);
            info.profile_identity_hex = hex_of(config.profile_identity);
            info.slice_identity_hex = hex_of(config.slice_identity);
            info.source_snapshot_hex = hex_of(config.source_snapshot);
            info.graph_configuration_hex = hex_of(config.graph_configuration);
            info.backend_hex = hex_of(config.backend);
            info.frontier_values = config.frontier_values;
            info.state_effects = config.state_effects;
            info.written_state_effects = config.written_state_effects;
            info.recurrent_state = config.recurrent_state;
            info.layer_start = config.layers.start;
            info.layer_end = config.layers.end;
            info.execution_contract_hex = hex_of(config.execution_contract);
            info.resident_identity_hex.reserve(config.resident_identities.size());
            for (const std::string_view identity : config.resident_identities) {
                info.resident_identity_hex.push_back(hex_of(identity));
            }
            info.import_binding_hex.reserve(config.activation_import_bindings.size());
            for (const std::string_view value : config.activation_import_identities) {
                info.import_identity_hex.push_back(hex_of(value));
            }
            for (const std::string_view value : config.activation_import_bindings) {
                info.import_binding_hex.push_back(hex_of(value));
            }
            info.export_binding_hex.reserve(config.activation_export_bindings.size());
            for (const std::string_view value : config.activation_export_identities) {
                info.export_identity_hex.push_back(hex_of(value));
            }
            for (const std::string_view value : config.activation_export_bindings) {
                info.export_binding_hex.push_back(hex_of(value));
            }
            info.request_identity_hex.reserve(config.request_input_identities.size());
            for (const std::string_view value : config.request_input_identities) {
                info.request_identity_hex.push_back(hex_of(value));
            }
            info.request_binding_hex.reserve(config.request_input_bindings.size());
            for (const std::string_view value : config.request_input_bindings) {
                info.request_binding_hex.push_back(hex_of(value));
            }
            result.stages.push_back(std::move(info));
        }
        result.ok = true;
        return result;
    });
}

namespace {

/// The generated enum mirrors the facade's values; this fails to build if not.
constexpr ModelFamily to_ffi(skippy::ModelFamily family) noexcept {
    static_assert(static_cast<unsigned>(skippy::ModelFamily::Bert) ==
                  static_cast<unsigned>(ModelFamily::Bert));
    return static_cast<ModelFamily>(static_cast<unsigned>(family));
}

} // namespace

StagedModelResult Bridge::load_staged_model(rust::Str path, std::uint32_t stages,
                                            std::uint32_t index) noexcept {
    return guarded_message<StagedModelResult>("load_staged_model", [&]() -> StagedModelResult {
        StagedModelResult result{};
        result.ok = false;

        const std::string model_path(path);
        auto inventory = skippy::probe::cached_inventory(model_path);
        if (!inventory) {
            result.error = rust::String::lossy(inventory.error());
            return result;
        }
        const auto layers = skippy::probe::layer_count_from(*inventory);
        const auto ranges = split_layers(layers, stages);
        if (!ranges) {
            result.error = rust::String::lossy(ranges.error());
            return result;
        }
        if (index >= ranges->size()) {
            result.error = rust::String::lossy("stage index is out of range");
            return result;
        }

        auto lease =
            skippy::llama::plan_model_stages(impl_->runtime.backend(), model_path, *ranges);
        if (!lease) {
            result.error = rust::String::lossy(lease.error().message);
            return result;
        }
        const auto spec = skippy::llama::own(lease->configs[index]);
        auto model = skippy::llama::StagedModel::open(model_path, spec);
        if (!model) {
            result.error = rust::String::lossy(model.error().message);
            return result;
        }
        result.ok = true;
        result.model = std::make_unique<StagedModel>(std::move(*model));
        return result;
    });
}

StagedSessionResult Bridge::create_staged_session(const StagedModel& model) noexcept {
    return guarded_message<StagedSessionResult>(
        "create_staged_session", [&]() -> StagedSessionResult {
            StagedSessionResult result{};
            result.ok = false;
            auto session = model.get().create_session();
            if (!session) {
                result.error = rust::String::lossy(session.error().message);
                return result;
            }
            result.ok = true;
            result.session = std::make_unique<StagedSession>(std::move(*session));
            return result;
        });
}

StagedEmbedResult Bridge::staged_embed(StagedSession& session,
                                       rust::Slice<const std::int32_t> tokens,
                                       rust::Slice<float> output) noexcept {
    return guarded_message<StagedEmbedResult>("staged_embed", [&]() -> StagedEmbedResult {
        StagedEmbedResult result{};
        result.ok = false;

        std::vector<llama_token> native(tokens.begin(), tokens.end());
        auto produced = session.get().embed(native, output);
        if (!produced) {
            result.error = rust::String::lossy(produced.error().message);
            return result;
        }
        result.ok = true;
        result.dimensions = *produced;
        return result;
    });
}

ModelFamily Bridge::model_family(rust::Str path) const noexcept {
    // Any failure - including bad_alloc copying the path - reads as unknown.
    return guarded<ModelFamily>([&]() -> ModelFamily {
        const std::string model_path(path);
        auto architecture = skippy::probe::model_architecture(model_path);
        if (!architecture) return ModelFamily::Unknown;
        return to_ffi(skippy::family_of(*architecture));
    });
}

ChainValidation Bridge::validate_chain(rust::Str path, std::uint32_t stage_count) const noexcept {
    return guarded_message<ChainValidation>("validate_chain", [&]() -> ChainValidation {
        ChainValidation result{};
        result.ok = false;

        const std::string model_path(path);
        auto inventory = skippy::probe::cached_inventory(model_path);
        if (!inventory) {
            result.error = rust::String::lossy(inventory.error());
            return result;
        }
        const auto layers = skippy::probe::layer_count_from(*inventory);
        const auto ranges = split_layers(layers, stage_count);
        if (!ranges) {
            result.error = rust::String::lossy(ranges.error());
            return result;
        }

        auto plans = skippy::llama::plan_model_plans(impl_->runtime.backend(), model_path, *ranges);
        if (!plans) {
            result.error = rust::String::lossy(plans.error().message);
            return result;
        }
        if (const auto failure = skippy::validate_chain(*plans); !failure) {
            result.error = rust::String::lossy("plan chain rejected: " +
                                               std::string(skippy::describe(failure.error())));
            return result;
        }
        result.ok = true;
        return result;
    });
}

ModelResult Bridge::load_model(rust::Str path) noexcept {
    return guarded<ModelResult>([&] {
        return to_result(impl_->runtime, impl_->runtime.load_model(
                                             ModelConfig{.path = std::string(as_view(path))}));
    });
}

SuppressTokensResult Bridge::suppress_tokens(const Model& model) const noexcept {
    return guarded_message<SuppressTokensResult>("suppress_tokens", [&]() -> SuppressTokensResult {
        // Copy while the model is borrowed. Allocation, including the result
        // vector's reserve, stays inside the same exception barrier.
        auto suppressed = impl_->runtime.with_model(model, [](const llama::LlamaModel& native) {
            SuppressTokensResult result{};
            const auto declared = native.suppress_tokens();
            if (reserve_suppress_tokens(result.tokens, declared.size()) !=
                TokenCapacityStatus::Ready) {
                return in_band_oom<SuppressTokensResult>();
            }
            std::ranges::for_each(
                declared, [&](Token token) { result.tokens.push_back(std::to_underlying(token)); });
            result.ok = true;
            return result;
        });
        if (!suppressed) {
            return in_band_failed<SuppressTokensResult>("suppress_tokens", "invalid model");
        }
        return std::move(*suppressed);
    });
}

RuntimeStatus Bridge::unload_model(Model& model) noexcept {
    return guarded<RuntimeStatus>(
        [&] { return to_plain_status<RuntimeStatus>(impl_->runtime.unload_model(model)); });
}

SessionResult Bridge::create_session(const Model& model, SessionConfig config) noexcept {
    return guarded<SessionResult>([&] {
        return to_result(impl_->runtime,
                         impl_->runtime.create_session(
                             model, skippy::SessionConfig{.dedicated_context = true,
                                                          .max_tokens = config.max_tokens,
                                                          .sequences = config.sequences}));
    });
}

RuntimeStatus Bridge::release_session(Session& session) noexcept {
    return guarded<RuntimeStatus>(
        [&] { return to_plain_status<RuntimeStatus>(impl_->runtime.release_session(session)); });
}

WorkloadResult Bridge::workload(const Session& session) const noexcept {
    return guarded<WorkloadResult>([&] {
        const auto info = impl_->runtime.with_session(
            session, [](const llama::LlamaSession& native) { return to_info(native.workload()); });
        if (!info) return WorkloadResult{.status = to_status(info.error()), .info = {}};
        return WorkloadResult{.status = RuntimeStatus::Ok, .info = *info};
    });
}

WorkloadStatus Bridge::tokenize(const Session& session, rust::Str text,
                                rust::Vec<std::int32_t>& tokens) noexcept {
    return guarded<WorkloadStatus>([&] {
        const auto tokenized = impl_->runtime.with_session(
            session, [&](llama::LlamaSession& native) { return native.tokenize(as_view(text)); });
        if (!tokenized) return WorkloadStatus::InvalidSession;
        if (!*tokenized) return to_status(tokenized->error());

        tokens.clear();
        tokens.reserve((*tokenized)->size());
        std::ranges::for_each(**tokenized,
                              [&](Token token) { tokens.push_back(std::to_underlying(token)); });
        return WorkloadStatus::Ok;
    });
}

EmbedResult Bridge::embed(const Session& session, rust::Slice<const std::int32_t> tokens,
                          rust::Slice<float> output) noexcept {
    return guarded<EmbedResult>([&] {
        const auto result = skippy::embed(
            impl_->runtime, session, as_tokens(tokens, impl_->token_scratch),
            EmbeddingOutput<F32Format>{.values = {output.data(), output.size()}, .metadata = {}});
        if (result)
            return EmbedResult{
                .status = WorkloadStatus::Ok, .dimensions = result->dimensions, .error = {}};

        EmbedResult failed{.status = to_status(result.error()),
                           .dimensions = 0,
                           .error = error_text(result.error())};
        if (result.error().code == WorkloadError::OutputTooSmall) {
            failed.dimensions = impl_->runtime
                                    .with_session(session,
                                                  [](const llama::LlamaSession& native) {
                                                      return native.embedding_dimensions();
                                                  })
                                    .value_or(0);
        }
        return failed;
    });
}

EmbedResult Bridge::embed_batch(const Session& session, rust::Slice<const std::int32_t> tokens,
                                rust::Slice<const std::uint32_t> lengths,
                                rust::Slice<float> output) noexcept {
    return guarded<EmbedResult>([&] {
        const auto flat = as_tokens(tokens, impl_->token_scratch);

        // Split the flattened buffer into the per-group views embed_batch takes.
        std::vector<std::span<const Token>> groups;
        groups.reserve(lengths.size());
        std::size_t at = 0;
        for (const std::uint32_t length : lengths) {
            if (at + length > flat.size())
                return EmbedResult{.status = WorkloadStatus::EmptyInput, .dimensions = 0};
            groups.emplace_back(flat.subspan(at, length));
            at += length;
        }
        if (at != flat.size())
            return EmbedResult{.status = WorkloadStatus::EmptyInput, .dimensions = 0};

        const auto result = skippy::embed_batch(
            impl_->runtime, session, groups,
            EmbeddingOutput<F32Format>{.values = {output.data(), output.size()}, .metadata = {}});
        if (result)
            return EmbedResult{.status = WorkloadStatus::Ok, .dimensions = result->dimensions};

        EmbedResult failed{.status = to_status(result.error()), .dimensions = 0};
        if (result.error().code == WorkloadError::OutputTooSmall) {
            failed.dimensions = impl_->runtime
                                    .with_session(session,
                                                  [](const llama::LlamaSession& native) {
                                                      return native.embedding_dimensions();
                                                  })
                                    .value_or(0);
        }
        return failed;
    });
}

RerankResult Bridge::rerank(const Session& session, rust::Str query, rust::Str document) noexcept {
    return guarded<RerankResult>([&] {
        const auto result =
            skippy::rerank(impl_->runtime, session, as_view(query), as_view(document));
        if (!result)
            return RerankResult{.status = to_status(result.error()),
                                .score = 0.0f,
                                .token_count = 0,
                                .error = error_text(result.error())};
        return RerankResult{.status = WorkloadStatus::Ok,
                            .score = result->score,
                            .token_count = result->token_count};
    });
}

RerankBatchResult Bridge::rerank_batch(const Session& session, rust::Str query,
                                       rust::Slice<const rust::Str> documents,
                                       rust::Slice<float> scores) noexcept {
    return guarded<RerankBatchResult>([&] {
        std::vector<std::string_view> views;
        views.reserve(documents.size());
        for (const rust::Str document : documents) {
            views.emplace_back(document.data(), document.size());
        }
        const auto result = skippy::rerank_batch(
            impl_->runtime, session, std::string_view{query.data(), query.size()}, views,
            std::span<float>{scores.data(), scores.size()});
        return result ? RerankBatchResult{.status = WorkloadStatus::Ok, .error = {}}
                      : RerankBatchResult{.status = to_status(result.error()),
                                          .error = error_text(result.error())};
    });
}

EncodeResult Bridge::encode(const Session& session,
                            rust::Slice<const std::int32_t> tokens) noexcept {
    return guarded<EncodeResult>([&] {
        const auto result =
            skippy::encode_prompt(impl_->runtime, session, as_tokens(tokens, impl_->token_scratch));
        if (!result)
            return EncodeResult{.status = to_status(result.error()),
                                .decoder_start = -1,
                                .error = error_text(result.error())};
        return EncodeResult{.status = WorkloadStatus::Ok,
                            .decoder_start = std::to_underlying(*result),
                            .error = {}};
    });
}

// Compile-time tests: the boundary enums stay value-compatible with the core.
static_assert(std::to_underlying(WorkloadKind::CausalGeneration) ==
              std::to_underlying(skippy::WorkloadKind::CausalGeneration));
static_assert(std::to_underlying(WorkloadKind::Embedding) ==
              std::to_underlying(skippy::WorkloadKind::Embedding));
static_assert(std::to_underlying(WorkloadKind::Rerank) ==
              std::to_underlying(skippy::WorkloadKind::Rerank));
static_assert(std::to_underlying(WorkloadKind::EncoderDecoder) ==
              std::to_underlying(skippy::WorkloadKind::EncoderDecoder));

static_assert(std::to_underlying(NativePooling::Unspecified) ==
              std::to_underlying(skippy::NativePooling::Unspecified));
static_assert(std::to_underlying(NativePooling::None) ==
              std::to_underlying(skippy::NativePooling::None));
static_assert(std::to_underlying(NativePooling::Mean) ==
              std::to_underlying(skippy::NativePooling::Mean));
static_assert(std::to_underlying(NativePooling::Cls) ==
              std::to_underlying(skippy::NativePooling::Cls));
static_assert(std::to_underlying(NativePooling::Last) ==
              std::to_underlying(skippy::NativePooling::Last));
static_assert(std::to_underlying(NativePooling::Rank) ==
              std::to_underlying(skippy::NativePooling::Rank));
static_assert(std::to_underlying(NativePooling::None) == std::to_underlying(Pooling::None));
static_assert(std::to_underlying(NativePooling::Last) == std::to_underlying(Pooling::Last));

static_assert(to_status(RuntimeError::ModelInUse) == RuntimeStatus::ModelInUse);
static_assert(to_status(WorkloadFailure::code_only(WorkloadError::OutputTooSmall)) ==
              WorkloadStatus::OutputTooSmall);
static_assert(to_native(std::optional<Pooling>{}) == NativePooling::Unspecified);
static_assert(to_native(std::optional<Pooling>{Pooling::Mean}) == NativePooling::Mean);
// The struct-path failed_with pin used EmbedResult; its error: String is not
// constexpr-constructible, so the pin lives with the enum path below and the
// struct path is pinned by the runtime suite (guarded<T> arms).
static_assert(failed_with<RuntimeStatus>(RuntimeStatus::InternalError) ==
              RuntimeStatus::InternalError);

static_assert(sizeof(Token) == sizeof(std::int32_t));

namespace {

/// Collects the terminal frame so Rust can own it across the FFI boundary.
struct CopyingSink {
    std::vector<std::uint8_t> bytes;
    std::optional<std::string> consume(skippy::FrameBytes frame) {
        bytes.assign(reinterpret_cast<const std::uint8_t*>(frame.data()),
                     reinterpret_cast<const std::uint8_t*>(frame.data()) + frame.size());
        return std::nullopt;
    }
};

} // namespace

ChainResult Bridge::open_chain(rust::Str path, rust::Slice<const std::int32_t> ranges) noexcept {
    return guarded_message<ChainResult>("open_chain", [&]() -> ChainResult {
        ChainResult result{};
        result.ok = false;
        result.chain = std::make_unique<StageChainHandle>();

        if (ranges.size() % 2 != 0) {
            result.error = rust::String::lossy("ranges must be start/end pairs");
            return result;
        }
        std::vector<skippy::LayerRange> layers;
        layers.reserve(ranges.size() / 2);
        for (std::size_t index = 0; index + 1 < ranges.size(); index += 2) {
            layers.push_back(skippy::LayerRange{ranges[index], ranges[index + 1]});
        }
        auto chain = skippy::llama::StageChain::open(std::string(path), layers);
        if (!chain) {
            result.error = rust::String::lossy(chain.error().message);
            return result;
        }
        result.chain = std::make_unique<StageChainHandle>(std::move(*chain));
        result.ok = true;
        return result;
    });
}

ChainRunResult Bridge::run_chain(StageChainHandle& chain,
                                 rust::Slice<const std::int32_t> tokens) noexcept {
    return guarded_message<ChainRunResult>("run_chain", [&]() -> ChainRunResult {
        ChainRunResult result{};
        result.ok = false;
        result.has_frame = false;

        std::vector<llama_token> native(tokens.begin(), tokens.end());
        CopyingSink sink;
        if (auto failure = chain.get().prefill(native, sink); !failure) {
            result.error = rust::String::lossy(failure.error().message);
            return result;
        }
        result.has_frame = !sink.bytes.empty();
        result.frame.reserve(sink.bytes.size());
        for (const std::uint8_t byte : sink.bytes) {
            result.frame.push_back(byte);
        }
        result.ok = true;
        return result;
    });
}

} // namespace skippy::ffi
