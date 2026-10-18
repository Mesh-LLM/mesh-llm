#include "llama/stage_planner.h"
#include "llama/backend.h"

#include "llama/model_inventory.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <ranges>
#include <unordered_map>
#include <utility>

namespace skippy::llama {
namespace {

PlanError code_for(enum skippy_status status) noexcept {
    switch (status) {
    case SKIPPY_STATUS_OK:
    case SKIPPY_STATUS_IO_ERROR:
    case SKIPPY_STATUS_MODEL_ERROR:
    case SKIPPY_STATUS_RUNTIME_ERROR:
    case SKIPPY_STATUS_ERROR: return PlanError::BackendFailed;
    case SKIPPY_STATUS_INVALID_ARGUMENT:
    case SKIPPY_STATUS_UNSUPPORTED:
    case SKIPPY_STATUS_BUFFER_TOO_SMALL: return PlanError::Rejected;
    }
    return PlanError::BackendFailed;
}

/// Owns the native error until it is consumed, so no path leaks it.
struct ErrorSlot {
    skippy_error* raw = nullptr;

    ErrorSlot() = default;
    ErrorSlot(const ErrorSlot&) = delete;
    ErrorSlot& operator=(const ErrorSlot&) = delete;

    ~ErrorSlot() {
        if (raw != nullptr) skippy_error_free(raw);
    }
};

/// Turns a native status into our value type, consuming the message.
std::expected<void, PlanFailure> check(enum skippy_status status, ErrorSlot& error) {
    if (status == SKIPPY_STATUS_OK) return {};
    PlanFailure failure{.code = code_for(status), .message = {}};
    if (error.raw != nullptr) {
        if (error.raw->message != nullptr) failure.message = error.raw->message;
        skippy_error_free(error.raw);
        error.raw = nullptr;
    }
    return std::unexpected(std::move(failure));
}

/// A borrowed view of a plan string. The native plan storage is immutable for
/// the plan's lifetime, so the view is valid until the plan is freed — callers
/// copy into owned storage exactly once, which is what the getter loop on the
/// other side of the benchmark does too.
std::expected<std::string_view, PlanFailure> read_view(const skippy_stage_plan& plan,
                                                       skippy_stage_plan_string_ref_v1 reference) {
    ErrorSlot error;
    const char* data = nullptr;
    std::size_t length = 0;
    const auto status = skippy_stage_plan_string_v1(&plan, reference, &data, &length, &error.raw);
    if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
    return std::string_view{data, length};
}

std::expected<std::string, PlanFailure> read_string(const skippy_stage_plan& plan,
                                                    skippy_stage_plan_string_ref_v1 reference) {
    auto view = read_view(plan, reference);
    if (!view.has_value()) return std::unexpected(view.error());
    return std::string{view->data(), view->size()};
}

StateKind state_kind_of(std::int32_t value) noexcept {
    switch (value) {
    case SKIPPY_STAGE_PLAN_STATE_KV_KEY: return StateKind::KvKey;
    case SKIPPY_STAGE_PLAN_STATE_KV_VALUE: return StateKind::KvValue;
    case SKIPPY_STAGE_PLAN_STATE_RECURRENT_CONV: return StateKind::RecurrentConvolution;
    case SKIPPY_STAGE_PLAN_STATE_RECURRENT_SSM: return StateKind::RecurrentSsm;
    default: return StateKind::KvKey;
    }
}

StateAccess access_of(std::int32_t value) noexcept {
    return value == SKIPPY_STAGE_PLAN_STATE_WRITE ? StateAccess::Write : StateAccess::Read;
}

StateResidency residency_of(std::int32_t value) noexcept {
    return value == SKIPPY_STAGE_PLAN_STATE_PER_STAGE ? StateResidency::PerStage
                                                      : StateResidency::LayerLocal;
}

namespace {
/// Appends bytes to an arena and returns the slice naming them; the twin of
/// the helper in stage_plan.cpp for the direct-to-plan materialise path.
ArenaSlice arena_put(std::string& arena, std::string_view bytes) {
    const ArenaSlice slice{
        .offset = static_cast<std::uint32_t>(arena.size()),
        .length = static_cast<std::uint32_t>(bytes.size()),
    };
    arena.append(bytes);
    return slice;
}
} // namespace

std::optional<PlanFailure> append_values(const skippy_stage_plan& plan, std::size_t profile_index,
                                         skippy_stage_plan_value_kind kind, std::uint64_t count,
                                         std::vector<FrontierValue>& storage, std::string& arena) {
    for (const std::size_t index :
         std::views::iota(std::size_t{0}, static_cast<std::size_t>(count))) {
        skippy_stage_plan_value_desc_v1 desc{};
        desc.abi_version = SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION;
        desc.struct_size = sizeof(desc);
        ErrorSlot error;
        const auto status =
            skippy_stage_plan_value_at_v1(&plan, profile_index, kind, index, &desc, &error.raw);
        if (auto ok = check(status, error); !ok.has_value()) return ok.error();

        auto identity = read_view(plan, desc.identity);
        if (!identity.has_value()) return identity.error();
        auto binding = read_view(plan, desc.binding);
        if (!binding.has_value()) return binding.error();
        // Arena put: append the bytes, remember the slice. One arena growth for
        // the whole plan instead of an allocation per identity.
        const auto identity_slice = arena_put(arena, *identity);
        const auto binding_slice = arena_put(arena, *binding);
        storage.push_back(FrontierValue{
            .identity = ValueIdentitySlice{identity_slice},
            .binding = ValueBindingSlice{binding_slice},
        });
    }
    return std::nullopt;
}

std::optional<PlanFailure> append_states(const skippy_stage_plan& plan, std::size_t profile_index,
                                         std::uint64_t count, std::vector<StateEffect>& storage) {
    for (const std::size_t index :
         std::views::iota(std::size_t{0}, static_cast<std::size_t>(count))) {
        skippy_stage_plan_state_desc_v1 desc{};
        desc.abi_version = SKIPPY_STAGE_PLAN_STATE_DESC_V1_ABI_VERSION;
        desc.struct_size = sizeof(desc);
        ErrorSlot error;
        const auto status =
            skippy_stage_plan_state_at_v1(&plan, profile_index, index, &desc, &error.raw);
        if (auto ok = check(status, error); !ok.has_value()) return ok.error();

        storage.push_back(StateEffect{
            .kind = state_kind_of(desc.kind),
            .access = access_of(desc.access),
            .layer = desc.layer,
            .residency = residency_of(desc.residency),
            .write_ordinal = desc.write_ordinal,
        });
    }
    return std::nullopt;
}

std::expected<skippy_stage_plan_desc_v1, PlanFailure> describe(const skippy_stage_plan& plan) {
    skippy_stage_plan_desc_v1 desc{};
    desc.abi_version = SKIPPY_STAGE_PLAN_DESC_V1_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    ErrorSlot error;
    const auto status = skippy_stage_plan_describe_v1(&plan, &desc, &error.raw);
    if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
    return desc;
}

std::vector<skippy_stage_planner_tensor_v1> native_tensors(const StagePlannerInputs& inputs) {
    std::vector<skippy_stage_planner_tensor_v1> tensors;
    tensors.reserve(inputs.tensors.size());
    for (const PackageTensor& tensor : inputs.tensors) {
        skippy_stage_planner_tensor_v1 entry{};
        entry.abi_version = SKIPPY_STAGE_PLANNER_TENSOR_V1_ABI_VERSION;
        entry.struct_size = sizeof(entry);
        entry.tensor_id = tensor.id.value.c_str();
        entry.native_name = tensor.native_name.c_str();
        entry.ggml_type = tensor.ggml_type;
        entry.dimension_count = tensor.dimension_count;
        std::ranges::copy(tensor.dimensions, std::begin(entry.dimensions));
        entry.split_no = tensor.split_no;
        entry.data_offset = tensor.data_offset;
        entry.stored_length = tensor.stored_length;
        tensors.push_back(entry);
    }
    return tensors;
}

std::vector<skippy_stage_planner_profile_v1> native_profiles(const StagePlannerInputs& inputs) {
    std::vector<skippy_stage_planner_profile_v1> profiles;
    profiles.reserve(inputs.profiles.size());
    for (const ProfileSpec& profile : inputs.profiles) {
        skippy_stage_planner_profile_v1 entry{};
        entry.abi_version = SKIPPY_STAGE_PLANNER_PROFILE_V1_ABI_VERSION;
        entry.struct_size = sizeof(entry);
        entry.profile_id = profile.id.value.c_str();
        entry.n_tokens = profile.tokens;
        entry.n_sequences = profile.sequences;
        entry.n_outputs = profile.outputs;
        entry.n_recurrent_rollback_sequences = profile.recurrent_rollback_sequences;
        profiles.push_back(entry);
    }
    return profiles;
}

} // namespace

void StagePlanDeleter::operator()(skippy_stage_plan* plan) const noexcept {
    if (plan != nullptr) skippy_stage_plan_free(plan);
}

void StagePlannerDeleter::operator()(skippy_stage_planner* planner) const noexcept {
    if (planner != nullptr) skippy_stage_planner_free(planner);
}

std::expected<StagePlanner, PlanFailure> StagePlanner::create(const StagePlannerInputs& inputs) {
    if (auto failure = validate_inputs(inputs); failure.has_value()) {
        return std::unexpected(*failure);
    }

    std::vector<std::string> shard_storage = inputs.shard_paths;
    std::vector<const char*> shards;
    shards.reserve(shard_storage.size());
    for (const std::string& path : shard_storage)
        shards.push_back(path.c_str());

    const std::vector<skippy_stage_planner_tensor_v1> tensors = native_tensors(inputs);
    const std::vector<skippy_stage_planner_profile_v1> profiles = native_profiles(inputs);

    skippy_stage_planner_config_v1 config{};
    config.abi_version = SKIPPY_STAGE_PLANNER_CONFIG_V1_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.package_id = inputs.package.value.c_str();
    config.shard_paths = shards.data();
    config.shard_count = shards.size();
    config.tensors = tensors.empty() ? nullptr : tensors.data();
    config.tensor_count = tensors.size();
    config.profiles = profiles.empty() ? nullptr : profiles.data();
    config.profile_count = profiles.size();
    config.graph_configuration_id = inputs.graph_configuration.value.c_str();
    config.backend_id = inputs.backend.value.c_str();

    ErrorSlot error;
    skippy_stage_planner* planner = nullptr;
    const auto status = skippy_stage_planner_create_v1(&config, &planner, &error.raw);
    if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
    if (planner == nullptr) {
        return std::unexpected(PlanFailure{
            .code = PlanError::BackendFailed,
            .message = "planner was accepted but not returned",
        });
    }
    StagePlanner wrapper;
    wrapper.planner_.reset(planner);
    return wrapper;
}

std::expected<StagePlanHandle, PlanFailure> StagePlanner::realize(LayerRange range) const {
    if (planner_ == nullptr) {
        return std::unexpected(PlanFailure{
            .code = PlanError::Rejected,
            .message = "planner is not loaded",
        });
    }
    ErrorSlot error;
    skippy_stage_plan* plan = nullptr;
    const auto status =
        skippy_stage_planner_realize_v1(planner_.get(), range.start, range.end, &plan, &error.raw);
    if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
    if (plan == nullptr) {
        return std::unexpected(PlanFailure{
            .code = PlanError::BackendFailed,
            .message = "realize was accepted but returned no plan",
        });
    }
    return StagePlanHandle{std::unique_ptr<skippy_stage_plan, StagePlanDeleter>{plan}};
}

std::expected<LayerRange, PlanFailure> StagePlanHandle::range() const {
    if (plan_ == nullptr) {
        return std::unexpected(PlanFailure{
            .code = PlanError::Rejected,
            .message = "plan is not loaded",
        });
    }
    const auto desc = describe(*plan_);
    if (!desc.has_value()) return std::unexpected(desc.error());
    return LayerRange{.start = desc->layer_start, .end = desc->layer_end};
}

/// Per-profile value counts, so spans are taken only after every profile's
/// values have been appended and the flat vectors can no longer reallocate.
/// A span into a vector that later grows is a dangling pointer — the exact
/// bug this structure exists to prevent.
struct ProfileCounts {
    std::uint32_t imports = 0;
    std::uint32_t exports = 0;
    std::uint32_t requests = 0;
    std::uint32_t effects = 0;
};

std::optional<PlanFailure> materialize_profile(const skippy_stage_plan& plan, std::size_t index,
                                               LayerRange range, StagePlan& into,
                                               ProfileCounts& counts) {
    skippy_stage_plan_profile_desc_v1 desc{};
    desc.abi_version = SKIPPY_STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    ErrorSlot error;
    const auto status = skippy_stage_plan_profile_at_v1(&plan, index, &desc, &error.raw);
    if (auto ok = check(status, error); !ok.has_value()) return ok.error();

    PlanProfile profile;
    profile.tokens = desc.n_tokens;
    profile.sequences = desc.n_sequences;
    profile.outputs = desc.n_outputs;
    profile.recurrent_rollback_sequences = desc.n_recurrent_rollback_sequences;
    profile.slice_range = range;

    const auto assign = [&](skippy_stage_plan_string_ref_v1 reference, auto& target) {
        auto text = read_string(plan, reference);
        if (!text.has_value()) return false;
        target.value = std::move(*text);
        return true;
    };
    const bool identities_ok = assign(desc.profile_id, profile.id) &&
                               assign(desc.graph_identity, profile.graph_identity) &&
                               assign(desc.profile_identity, profile.profile_identity) &&
                               assign(desc.slice_identity, profile.slice_identity) &&
                               assign(desc.source_snapshot_identity, profile.source_snapshot) &&
                               assign(desc.graph_configuration_id, profile.graph_configuration) &&
                               assign(desc.backend_id, profile.backend);
    if (!identities_ok) {
        return PlanFailure{
            .code = PlanError::BackendFailed,
            .message = "plan profile identity is unreadable",
        };
    }

    // These are patch 0004's per-family annotations as the facade sees them:
    // which state a stage touches, and what crosses its boundary. Values are
    // appended directly into the plan's flat storage and the profile spans
    // into it, so a profile never allocates its own containers.
    const std::size_t imports = into.activation_imports.size();
    const std::size_t exports = into.activation_exports.size();
    const std::size_t requests = into.request_inputs.size();
    const std::size_t effects = into.state_effects.size();
    (void)imports;
    (void)exports;
    (void)requests;
    (void)effects;

    if (auto failure = append_values(plan, index, SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_IMPORT,
                                     desc.activation_import_count, into.activation_imports,
                                     into.identity_arena);
        failure.has_value()) {
        return failure;
    }
    if (auto failure = append_values(plan, index, SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_EXPORT,
                                     desc.activation_export_count, into.activation_exports,
                                     into.identity_arena);
        failure.has_value()) {
        return failure;
    }
    if (auto failure =
            append_values(plan, index, SKIPPY_STAGE_PLAN_VALUE_REQUEST_INPUT,
                          desc.request_input_count, into.request_inputs, into.identity_arena);
        failure.has_value()) {
        return failure;
    }
    if (auto failure = append_states(plan, index, desc.state_effect_count, into.state_effects);
        failure.has_value()) {
        return failure;
    }

    counts.imports = static_cast<std::uint32_t>(into.activation_imports.size() - imports);
    counts.exports = static_cast<std::uint32_t>(into.activation_exports.size() - exports);
    counts.requests = static_cast<std::uint32_t>(into.request_inputs.size() - requests);
    counts.effects = static_cast<std::uint32_t>(into.state_effects.size() - effects);
    into.profiles.push_back(std::move(profile));
    return std::nullopt;
}

std::expected<StagePlan, PlanFailure> StagePlanHandle::materialize() const {
    if (plan_ == nullptr) {
        return std::unexpected(
            PlanFailure{.code = PlanError::Rejected, .message = "plan is not loaded"});
    }
    const auto desc = describe(*plan_);
    if (!desc.has_value()) return std::unexpected(desc.error());

    StagePlan plan;
    plan.layer_count = desc->layer_count;
    plan.range = LayerRange{.start = desc->layer_start, .end = desc->layer_end};

    auto package = read_string(*plan_, desc->package_id);
    if (!package.has_value()) return std::unexpected(package.error());
    plan.package = PackageId{std::move(*package)};

    auto id = read_string(*plan_, desc->plan_id);
    if (!id.has_value()) return std::unexpected(id.error());
    plan.id = PlanId{std::move(*id)};

    auto contract = read_string(*plan_, desc->execution_contract);
    if (!contract.has_value()) return std::unexpected(contract.error());
    plan.execution_contract = std::move(*contract);

    const auto resident_count = static_cast<std::size_t>(desc->resident_tensor_count);
    plan.resident_identities.reserve(resident_count);
    for (const std::size_t index : std::views::iota(std::size_t{0}, resident_count)) {
        skippy_stage_plan_value_desc_v1 value{};
        value.abi_version = SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION;
        value.struct_size = sizeof(value);
        ErrorSlot error;
        const auto status =
            skippy_stage_plan_resident_tensor_at_v1(plan_.get(), index, &value, &error.raw);
        if (auto ok = check(status, error); !ok.has_value()) return std::unexpected(ok.error());
        auto identity = read_view(*plan_, value.identity);
        if (!identity.has_value()) return std::unexpected(identity.error());
        plan.resident_identities.push_back(TensorSlice{
            arena_put(plan.identity_arena, *identity),
        });
    }

    const auto profile_count = static_cast<std::size_t>(desc->profile_count);
    plan.profiles.reserve(profile_count);
    std::vector<ProfileCounts> counts(profile_count);
    for (const std::size_t index : std::views::iota(std::size_t{0}, profile_count)) {
        if (auto failure = materialize_profile(*plan_, index, plan.range, plan, counts[index]);
            failure.has_value()) {
            return std::unexpected(*failure);
        }
    }

    // Phase two: every flat vector is final, so spans cannot dangle by growth.
    // Offsets accumulate across profiles exactly as the appends did.
    std::size_t imports = 0;
    std::size_t exports = 0;
    std::size_t requests = 0;
    std::size_t effects = 0;
    for (const std::size_t index : std::views::iota(std::size_t{0}, profile_count)) {
        PlanProfile& profile = plan.profiles[index];
        const ProfileCounts& count = counts[index];
        profile.activation_imports =
            std::span{plan.activation_imports}.subspan(imports, count.imports);
        profile.activation_exports =
            std::span{plan.activation_exports}.subspan(exports, count.exports);
        profile.request_inputs = std::span{plan.request_inputs}.subspan(requests, count.requests);
        profile.state_effects = std::span{plan.state_effects}.subspan(effects, count.effects);
        imports += count.imports;
        exports += count.exports;
        requests += count.requests;
        effects += count.effects;
    }

    if (auto failure = validate(plan); failure.has_value()) {
        return std::unexpected(
            PlanFailure{.code = *failure, .message = "materialized plan is invalid"});
    }
    return plan;
}

namespace {

std::string cache_key(const std::string& path, std::span<const LayerRange> ranges) {
    std::string key = probe::file_stamp(path);
    for (const LayerRange range : ranges) {
        key += "|" + std::to_string(range.start) + "-" + std::to_string(range.end);
    }
    return key;
}

std::expected<std::vector<StageLoadConfig>, PlanFailure>
configs_from(const std::vector<std::shared_ptr<const StagePlan>>& plans) {
    std::vector<StageLoadConfig> configs;
    configs.reserve(plans.size());
    for (const auto& plan : plans) {
        auto config = stage_load_config(*plan, plan->profiles.front().id);
        if (!config.has_value()) {
            return std::unexpected(PlanFailure{.code = config.error(), .message = {}});
        }
        configs.push_back(std::move(*config));
    }
    return configs;
}

} // namespace

std::shared_ptr<const PlanIndex>
PlanStore::publish(const std::string& key, std::vector<std::shared_ptr<const StagePlan>> plans) {
    std::shared_ptr<const PlanIndex> current =
        std::atomic_load_explicit(&index_, std::memory_order_acquire);
    for (;;) {
        if (current->by_key.contains(key)) return current; // a racing publisher won
        auto next = std::make_shared<PlanIndex>(*current); // copy-on-write
        next->by_key.emplace(key, plans);
        // CAS on the shared_ptr itself: the winner installs its index, the
        // loser's `current` is reloaded and the loop observes the winner.
        if (std::atomic_compare_exchange_weak_explicit(
                &index_, &current, std::static_pointer_cast<const PlanIndex>(next),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            return std::static_pointer_cast<const PlanIndex>(next);
        }
    }
}

std::expected<std::vector<StagePlan>, PlanFailure>
plan_model_plans(LlamaBackend& backend, const std::string& path,
                 std::span<const LayerRange> ranges) {
    // Populate first (idempotent), then copy the plans out of the snapshot.
    // The copies own their values; nothing borrowed escapes this call.
    auto lease = plan_model_stages(backend, path, ranges);
    if (!lease) return std::unexpected(lease.error());
    const std::string key = cache_key(path, ranges);
    const std::shared_ptr<const PlanIndex> snapshot = backend.plan_store().snapshot();
    const auto found = snapshot->by_key.find(key);
    if (found == snapshot->by_key.end()) {
        return std::unexpected(PlanFailure{
            .code = PlanError::Rejected,
            .message = "plan vanished from the store",
        });
    }
    std::vector<StagePlan> plans;
    plans.reserve(found->second.size());
    for (const auto& plan : found->second) {
        plans.push_back(plan->clone()); // plans own their values; spans must not cross
    }
    return plans;
}

std::expected<PlanLease, PlanFailure> plan_model_stages(LlamaBackend& backend,
                                                        const std::string& path,
                                                        std::span<const LayerRange> ranges) {
    if (ranges.empty()) {
        return std::unexpected(
            PlanFailure{.code = PlanError::EmptyChain, .message = "no ranges requested"});
    }
    auto inventory = probe::cached_inventory(path);
    if (!inventory.has_value()) {
        return std::unexpected(PlanFailure{
            .code = PlanError::Rejected,
            .message = inventory.error(),
        });
    }

    StagePlannerInputs inputs;
    inputs.package = PackageId{probe::cached_sha256(path)};
    inputs.shard_paths = {path};
    inputs.tensors.reserve(inventory->size());
    for (const probe::TensorInfo& info : *inventory) {
        inputs.tensors.push_back(PackageTensor{
            .id = TensorId{info.name},
            .native_name = info.name,
            .ggml_type = static_cast<std::int32_t>(info.type),
            .dimension_count = info.dimension_count,
            .dimensions = info.dimensions,
            .split_no = 0,
            .data_offset = info.offset,
            .stored_length = info.stored_length,
        });
    }
    // A token budget large enough for any prefill the caller can ask for; the
    // planner only uses it to size the guarded profile.
    inputs.profiles = {
        ProfileSpec{
            .id = ProfileId{"decode"},
            .tokens = 512,
            .sequences = 1,
            .outputs = 512,
        },
    };
    inputs.graph_configuration = GraphConfigurationId{"stage"};
    inputs.backend = BackendId{"metal"};

    const std::string key = cache_key(path, ranges);
    PlanStore& store = backend.plan_store();

    // Fast path: snapshot the index without blocking, build the configs, and
    // pin the snapshot in the lease we return.
    if (auto snapshot = store.snapshot(); snapshot->by_key.contains(key)) {
        auto configs = configs_from(snapshot->by_key.at(key));
        if (!configs) return std::unexpected(configs.error());
        return PlanLease{.configs = std::move(*configs), .pinned = std::move(snapshot)};
    }

    auto planner = StagePlanner::create(inputs);
    if (!planner.has_value()) return std::unexpected(planner.error());

    std::vector<StagePlan> realized;
    realized.reserve(ranges.size());
    for (const LayerRange range : ranges) {
        auto plan = planner->realize(range);
        if (!plan.has_value()) return std::unexpected(plan.error());
        auto materialized = plan->materialize();
        if (!materialized.has_value()) return std::unexpected(materialized.error());
        realized.push_back(std::move(*materialized));
    }

    // Publish under shared ownership, then build the lease from whichever
    // index won the race — ours or the other publisher's.
    std::vector<std::shared_ptr<const StagePlan>> stored;
    stored.reserve(realized.size());
    for (StagePlan& plan : realized) {
        stored.push_back(std::make_shared<const StagePlan>(std::move(plan)));
    }
    const std::shared_ptr<const PlanIndex> winner = store.publish(key, std::move(stored));
    auto configs = configs_from(winner->by_key.at(key));
    if (!configs) return std::unexpected(configs.error());
    return PlanLease{.configs = std::move(*configs), .pinned = winner};
}

std::expected<void, PlanFailure> validate_chain(std::span<const StagePlanHandle> plans) {
    std::vector<const skippy_stage_plan*> native;
    native.reserve(plans.size());
    for (const StagePlanHandle& plan : plans) {
        if (!plan.valid()) {
            return std::unexpected(PlanFailure{
                .code = PlanError::Rejected,
                .message = "chain contains an empty plan",
            });
        }
        native.push_back(plan.native());
    }
    ErrorSlot error;
    const auto status =
        skippy_stage_plan_validate_chain_v1(native.data(), native.size(), &error.raw);
    return check(status, error);
}

} // namespace skippy::llama
