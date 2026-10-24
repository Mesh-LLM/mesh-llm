/**
 * @file stage_plan_benchmark.cpp
 * @brief Old C ABI vs facade for stage planning.
 *
 * Both sides are given identical inputs and asked for the same information.
 * The old side is what a C consumer must write: realize, describe, then one
 * call per profile, value, state effect and string. The facade side realizes
 * and materializes once.
 */

#include "llama/model_inventory.h"
#include "llama/stage_planner.h"

#include "skippy.h"

#include <benchmark/benchmark.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace {

using skippy::LayerRange;
using skippy::StagePlannerInputs;
using skippy::llama::StagePlanHandle;
using skippy::llama::StagePlanner;

std::string model_path() {
    const char* value = std::getenv("SKIPPY_BENCH_MODEL");
    return value == nullptr ? std::string{} : std::string{value};
}

/// Owns one planner per side and the string storage both configs borrow.
class StageHarness {
public:
    explicit StageHarness(const std::string& path, LayerRange range) : range_(range) {
        auto inventory = skippy::probe::read_inventory(path);
        if (!inventory) return;

        tensors_ = *inventory;
        for (const auto& info : tensors_) {
            ids_.emplace_back(info.name);
            natives_.emplace_back(info.name);
        }
        package_ = skippy::probe::sha256_of(path);

        StagePlannerInputs inputs;
        inputs.package = skippy::PackageId{package_};
        inputs.shard_paths = {path};
        inputs.tensors.reserve(tensors_.size());
        for (const auto& info : tensors_) {
            inputs.tensors.push_back(skippy::PackageTensor{
                .id = skippy::TensorId{info.name},
                .native_name = info.name,
                .ggml_type = static_cast<std::int32_t>(info.type),
                .dimension_count = info.dimension_count,
                .dimensions = info.dimensions,
                .split_no = 0,
                .data_offset = info.offset,
                .stored_length = info.stored_length,
            });
        }
        inputs.profiles = {skippy::ProfileSpec{
            .id = skippy::ProfileId{"decode"}, .tokens = 512, .sequences = 1, .outputs = 512}};
        inputs.graph_configuration = skippy::GraphConfigurationId{"bench"};
        inputs.backend = skippy::BackendId{"metal"};

        auto created = StagePlanner::create(inputs);
        if (!created) return;
        api_.emplace(std::move(*created));

        // The native side gets the same inputs through the C struct.
        native_tensors_.reserve(inputs.tensors.size());
        for (const auto& tensor : inputs.tensors) {
            skippy_stage_planner_tensor_v1 entry{};
            entry.abi_version = SKIPPY_STAGE_PLANNER_TENSOR_V1_ABI_VERSION;
            entry.struct_size = sizeof(entry);
            entry.ggml_type = tensor.ggml_type;
            entry.dimension_count = tensor.dimension_count;
            std::ranges::copy(tensor.dimensions, std::begin(entry.dimensions));
            entry.split_no = tensor.split_no;
            entry.data_offset = tensor.data_offset;
            entry.stored_length = tensor.stored_length;
            native_tensors_.push_back(entry);
        }
        // Point each entry at its own name storage (the vector may reallocate).
        for (std::size_t index = 0; index < native_tensors_.size(); ++index) {
            native_tensors_[index].tensor_id = ids_[index].c_str();
            native_tensors_[index].native_name = natives_[index].c_str();
        }

        skippy_stage_planner_profile_v1 profile{};
        profile.abi_version = SKIPPY_STAGE_PLANNER_PROFILE_V1_ABI_VERSION;
        profile.struct_size = sizeof(profile);
        profile.profile_id = "decode";
        profile.n_tokens = 512;
        profile.n_sequences = 1;
        profile.n_outputs = 512;
        native_profiles_.push_back(profile);

        const char* shard = path.c_str();
        skippy_stage_planner_config_v1 config{};
        config.abi_version = SKIPPY_STAGE_PLANNER_CONFIG_V1_ABI_VERSION;
        config.struct_size = sizeof(config);
        config.package_id = package_.c_str();
        config.shard_paths = &shard;
        config.shard_count = 1;
        config.tensors = native_tensors_.data();
        config.tensor_count = native_tensors_.size();
        config.profiles = native_profiles_.data();
        config.profile_count = native_profiles_.size();
        config.graph_configuration_id = "bench";
        config.backend_id = "metal";

        skippy_error* error = nullptr;
        ready_ = skippy_stage_planner_create_v1(&config, &native_, &error) == SKIPPY_STATUS_OK;
        if (!ready_ && error != nullptr) skippy_error_free(error);
    }

    ~StageHarness() {
        if (native_ != nullptr) skippy_stage_planner_free(native_);
    }
    StageHarness(const StageHarness&) = delete;
    StageHarness& operator=(const StageHarness&) = delete;

    [[nodiscard]] bool ready() const { return ready_ && api_.has_value(); }
    [[nodiscard]] LayerRange range() const { return range_; }
    [[nodiscard]] skippy_stage_planner* native() const { return native_; }
    [[nodiscard]] const StagePlanner& api() const { return *api_; }

    /// The C consumer's whole materialization loop, including string copies.
    [[nodiscard]] std::size_t materialize_native() const;
    [[nodiscard]] std::size_t materialize_api() const;

public:
    LayerRange range_;
    std::vector<skippy::probe::TensorInfo> tensors_;
    std::vector<std::string> ids_;
    std::vector<std::string> natives_;
    std::vector<skippy_stage_planner_tensor_v1> native_tensors_;
    std::vector<skippy_stage_planner_profile_v1> native_profiles_;
    std::string package_;
    /// Realizes once so materialization can be measured on its own.
    void cache_plan(bool use_api) {
        if (use_api) {
            auto plan = api_->realize(range_);
            if (plan) cached_api_ = std::move(*plan);
            return;
        }
        skippy_error* error = nullptr;
        if (skippy_stage_planner_realize_v1(native_, range_.start, range_.end, &cached_native_,
                                            &error) != SKIPPY_STATUS_OK) {
            if (error != nullptr) skippy_error_free(error);
        }
    }

    /// Realize and free, the only work both sides share.
    std::size_t realize_only(bool use_api) {
        if (use_api) {
            auto plan = api_->realize(range_);
            return plan ? 1 : 0;
        }
        skippy_stage_plan* plan = nullptr;
        skippy_error* error = nullptr;
        if (skippy_stage_planner_realize_v1(native_, range_.start, range_.end, &plan, &error) !=
            SKIPPY_STATUS_OK) {
            if (error != nullptr) skippy_error_free(error);
            return 0;
        }
        skippy_stage_plan_free(plan);
        return 1;
    }

    /// Materializes a plan that was already realized.
    std::size_t materialize_cached(bool use_api) {
        if (use_api) {
            if (!cached_api_.has_value()) return 0;
            auto materialized = cached_api_->materialize();
            if (!materialized) return 0;
            std::size_t total = materialized->execution_contract.size();
            for (const auto& profile : materialized->profiles) {
                total += materialized->activation_imports_of(profile).size() +
                         materialized->activation_exports_of(profile).size() +
                         materialized->state_effects_of(profile).size();
            }
            return total + materialized->resident_identities.size();
        }
        return materialize_native_plan(cached_native_);
    }

    std::size_t materialize_native_plan(skippy_stage_plan* plan) const;
    [[nodiscard]] skippy_stage_plan* cached_native() const noexcept { return cached_native_; }
    [[nodiscard]] const std::optional<StagePlanHandle>& cached_api() const noexcept {
        return cached_api_;
    }

public:
    skippy_stage_planner* native_ = nullptr;
    std::optional<StagePlanner> api_;
    skippy_stage_plan* cached_native_ = nullptr;
    std::optional<StagePlanHandle> cached_api_;
    bool ready_ = false;
};

std::size_t StageHarness::materialize_native_plan(skippy_stage_plan* plan) const {
    if (plan == nullptr) return 0;
    skippy_error* error = nullptr;
    skippy_stage_plan_desc_v1 desc{};
    desc.abi_version = SKIPPY_STAGE_PLAN_DESC_V1_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    if (skippy_stage_plan_describe_v1(plan, &desc, &error) != SKIPPY_STATUS_OK) {
        if (error != nullptr) skippy_error_free(error);
        return 0;
    }
    // Retain what is read: the facade builds owned values, so a getter loop
    // that throws its strings away would compare unlike work.
    std::size_t total = 0;
    std::vector<std::string> retained;
    const auto read = [&](skippy_stage_plan_string_ref_v1 reference) {
        const char* data = nullptr;
        std::size_t length = 0;
        if (skippy_stage_plan_string_v1(plan, reference, &data, &length, &error) !=
            SKIPPY_STATUS_OK) {
            skippy_error_free(error);
            return;
        }
        retained.emplace_back(data, length);
        total += length;
    };
    read(desc.package_id);
    read(desc.plan_id);
    read(desc.execution_contract);
    for (std::size_t resident = 0; resident < desc.resident_tensor_count; ++resident) {
        skippy_stage_plan_value_desc_v1 value{};
        value.abi_version = SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION;
        value.struct_size = sizeof(value);
        if (skippy_stage_plan_resident_tensor_at_v1(plan, resident, &value, &error) !=
            SKIPPY_STATUS_OK) {
            skippy_error_free(error);
            return total;
        }
        read(value.identity);
    }
    for (std::size_t index = 0; index < desc.profile_count; ++index) {
        skippy_stage_plan_profile_desc_v1 profile{};
        profile.abi_version = SKIPPY_STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION;
        profile.struct_size = sizeof(profile);
        if (skippy_stage_plan_profile_at_v1(plan, index, &profile, &error) != SKIPPY_STATUS_OK) {
            skippy_error_free(error);
            return total;
        }
        // The facade materialises the profile's own identities too.
        // All seven: the facade reads the same identity set per profile, so a
        // loop that skipped any would compare unlike work.
        read(profile.profile_id);
        read(profile.graph_identity);
        read(profile.profile_identity);
        read(profile.slice_identity);
        read(profile.source_snapshot_identity);
        read(profile.graph_configuration_id);
        read(profile.backend_id);

        const auto read_values = [&](skippy_stage_plan_value_kind kind, std::uint64_t count) {
            for (std::size_t slot = 0; slot < count; ++slot) {
                skippy_stage_plan_value_desc_v1 value{};
                value.abi_version = SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION;
                value.struct_size = sizeof(value);
                if (skippy_stage_plan_value_at_v1(plan, index, kind, slot, &value, &error) ==
                    SKIPPY_STATUS_OK) {
                    read(value.identity);
                    read(value.binding);
                } else if (error != nullptr) {
                    skippy_error_free(error);
                }
            }
        };
        read_values(SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_IMPORT, profile.activation_import_count);
        read_values(SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_EXPORT, profile.activation_export_count);
        read_values(SKIPPY_STAGE_PLAN_VALUE_REQUEST_INPUT, profile.request_input_count);
        // The facade also reads every persistent-state effect, so the getter
        // loop must too or the row compares unlike work.
        for (std::size_t effect = 0; effect < profile.state_effect_count; ++effect) {
            skippy_stage_plan_state_desc_v1 state{};
            state.abi_version = SKIPPY_STAGE_PLAN_STATE_DESC_V1_ABI_VERSION;
            state.struct_size = sizeof(state);
            if (skippy_stage_plan_state_at_v1(plan, index, effect, &state, &error) ==
                SKIPPY_STATUS_OK) {
            } else if (error != nullptr) {
                skippy_error_free(error);
            }
        }
    }
    benchmark::DoNotOptimize(retained);
    return total;
}

std::size_t StageHarness::materialize_native() const {
    skippy_stage_plan* plan = nullptr;
    skippy_error* error = nullptr;
    if (skippy_stage_planner_realize_v1(native_, range_.start, range_.end, &plan, &error) !=
        SKIPPY_STATUS_OK) {
        if (error != nullptr) skippy_error_free(error);
        return 0;
    }

    skippy_stage_plan_desc_v1 desc{};
    desc.abi_version = SKIPPY_STAGE_PLAN_DESC_V1_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    if (skippy_stage_plan_describe_v1(plan, &desc, &error) != SKIPPY_STATUS_OK) {
        skippy_error_free(error);
        skippy_stage_plan_free(plan);
        return 0;
    }

    // Retain what is read: the facade builds owned values, so a getter loop
    // that throws its strings away would compare unlike work.
    std::size_t total = 0;
    std::vector<std::string> retained;
    const auto read = [&](skippy_stage_plan_string_ref_v1 reference) {
        const char* data = nullptr;
        std::size_t length = 0;
        if (skippy_stage_plan_string_v1(plan, reference, &data, &length, &error) !=
            SKIPPY_STATUS_OK) {
            skippy_error_free(error);
            return;
        }
        retained.emplace_back(data, length);
        total += length;
    };
    read(desc.package_id);
    read(desc.plan_id);
    read(desc.execution_contract);

    for (std::size_t resident = 0; resident < desc.resident_tensor_count; ++resident) {
        skippy_stage_plan_value_desc_v1 value{};
        value.abi_version = SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION;
        value.struct_size = sizeof(value);
        if (skippy_stage_plan_resident_tensor_at_v1(plan, resident, &value, &error) !=
            SKIPPY_STATUS_OK) {
            skippy_error_free(error);
            break;
        }
        read(value.identity);
    }

    for (std::size_t index = 0; index < desc.profile_count; ++index) {
        skippy_stage_plan_profile_desc_v1 profile{};
        profile.abi_version = SKIPPY_STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION;
        profile.struct_size = sizeof(profile);
        if (skippy_stage_plan_profile_at_v1(plan, index, &profile, &error) != SKIPPY_STATUS_OK) {
            skippy_error_free(error);
            break;
        }
        // All seven: the facade reads the same identity set per profile, so a
        // loop that skipped three would compare unlike work.
        read(profile.profile_id);
        read(profile.graph_identity);
        read(profile.profile_identity);
        read(profile.slice_identity);
        read(profile.source_snapshot_identity);
        read(profile.graph_configuration_id);
        read(profile.backend_id);

        for (std::size_t part = 0; part < profile.activation_import_count; ++part) {
            skippy_stage_plan_value_desc_v1 value{};
            value.abi_version = SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION;
            value.struct_size = sizeof(value);
            if (skippy_stage_plan_value_at_v1(plan, index,
                                              SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_IMPORT, part,
                                              &value, &error) != SKIPPY_STATUS_OK) {
                skippy_error_free(error);
                break;
            }
            read(value.identity);
            read(value.binding);
        }
        for (std::size_t part = 0; part < profile.activation_export_count; ++part) {
            skippy_stage_plan_value_desc_v1 value{};
            value.abi_version = SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION;
            value.struct_size = sizeof(value);
            if (skippy_stage_plan_value_at_v1(plan, index,
                                              SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_EXPORT, part,
                                              &value, &error) != SKIPPY_STATUS_OK) {
                skippy_error_free(error);
                break;
            }
            read(value.identity);
            read(value.binding);
        }
        for (std::size_t effect = 0; effect < profile.state_effect_count; ++effect) {
            skippy_stage_plan_state_desc_v1 state{};
            state.abi_version = SKIPPY_STAGE_PLAN_STATE_DESC_V1_ABI_VERSION;
            state.struct_size = sizeof(state);
            if (skippy_stage_plan_state_at_v1(plan, index, effect, &state, &error) !=
                SKIPPY_STATUS_OK) {
                skippy_error_free(error);
                break;
            }
        }
    }

    skippy_stage_plan_free(plan);
    return total;
}

std::size_t StageHarness::materialize_api() const {
    auto plan = api().realize(range_);
    if (!plan) return 0;
    auto materialized = plan->materialize();
    if (!materialized) return 0;
    std::size_t total = materialized->execution_contract.size();
    total += materialized->resident_identities.size();
    for (const auto& profile : materialized->profiles) {
        total += materialized->activation_imports_of(profile).size() +
                 materialized->activation_exports_of(profile).size() +
                 materialized->state_effects_of(profile).size();
    }
    return total;
}

} // namespace

void realize(benchmark::State& state, bool use_api) {
    const std::string path = model_path();
    if (path.empty()) {
        state.SkipWithError("SKIPPY_BENCH_MODEL is not set");
        return;
    }
    StageHarness harness(path, LayerRange{0, static_cast<std::int32_t>(state.range(0))});
    if (!harness.ready()) {
        state.SkipWithError("planner setup failed");
        return;
    }
    for (auto _ : state) {
        std::size_t produced = harness.realize_only(use_api);
        benchmark::DoNotOptimize(&produced);
    }
}

void materialize(benchmark::State& state, bool use_api) {
    const std::string path = model_path();
    if (path.empty()) {
        state.SkipWithError("SKIPPY_BENCH_MODEL is not set");
        return;
    }
    StageHarness harness(path, LayerRange{0, static_cast<std::int32_t>(state.range(0))});
    if (!harness.ready()) {
        state.SkipWithError("planner setup failed");
        return;
    }
    harness.cache_plan(use_api);
    for (auto _ : state) {
        std::size_t produced = harness.materialize_cached(use_api);
        benchmark::DoNotOptimize(&produced);
    }
}

/// Validates a two-stage chain: the native entry point against the facade's
/// pure validator over materialised plans.
void validate(benchmark::State& state, bool use_api) {
    const std::string path = model_path();
    if (path.empty()) {
        state.SkipWithError("SKIPPY_BENCH_MODEL is not set");
        return;
    }
    auto inventory = skippy::probe::cached_inventory(path);
    if (!inventory) {
        state.SkipWithError("inventory failed");
        return;
    }
    const auto layers = skippy::probe::layer_count_from(*inventory));
    const std::int32_t half = layers / 2;

    StageHarness first(path, LayerRange{0, half});
    StageHarness second(path, LayerRange{half, layers});
    if (!first.ready() || !second.ready()) {
        state.SkipWithError("planner setup failed");
        return;
    }
    first.cache_plan(use_api);
    second.cache_plan(use_api);

    std::size_t validated = 0;
    if (use_api) {
        // Materialise once: the native call validates handles, so charging the
        // facade for materialisation inside the loop would compare unlike work.
        std::vector<skippy::StagePlan> plans;
        plans.reserve(2);
        for (const auto* handle : {&first, &second}) {
            if (!handle->cached_api().has_value()) {
                state.SkipWithError("plan was not realized");
                return;
            }
            auto plan = handle->cached_api()->materialize();
            if (!plan.has_value()) {
                state.SkipWithError("materialize failed");
                return;
            }
            plans.push_back(std::move(*plan));
        }
        // Time only successful validation: a failing loop would measure the
        // error path, not the check. validate_chain returns the failure as
        // an engaged optional, so success is the empty one.
        if (skippy::validate_chain(plans).has_value()) {
            state.SkipWithError("chain validation failed");
            return;
        }
        for (auto _ : state) {
            validated = skippy::validate_chain(plans).has_value() ? 0 : 1;
            benchmark::DoNotOptimize(&validated);
        }
        return;
    }

    skippy_stage_plan* plans[] = {first.cached_native(), second.cached_native()};
    {
        skippy_error* error = nullptr;
        const auto status = skippy_stage_plan_validate_chain_v1(plans, 2, &error);
        if (error != nullptr) skippy_error_free(error);
        if (status != SKIPPY_STATUS_OK) {
            state.SkipWithError("chain validation failed");
            return;
        }
    }
    for (auto _ : state) {
        skippy_error* error = nullptr;
        const auto status = skippy_stage_plan_validate_chain_v1(plans, 2, &error);
        if (error != nullptr) skippy_error_free(error);
        validated = status == SKIPPY_STATUS_OK ? 1 : 0;
        benchmark::DoNotOptimize(&validated);
    }
}

void stage_validate_ffi(benchmark::State& state) { validate(state, false); }
void stage_validate_api(benchmark::State& state) { validate(state, true); }

void stage_realize_ffi(benchmark::State& state) { realize(state, false); }
void stage_realize_api(benchmark::State& state) { realize(state, true); }
void stage_materialize_ffi(benchmark::State& state) { materialize(state, false); }
void stage_materialize_api(benchmark::State& state) { materialize(state, true); }

BENCHMARK(stage_realize_ffi)->Name("stage_realize/FFI")->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK(stage_realize_api)->Name("stage_realize/API")->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK(stage_materialize_ffi)
    ->Name("stage_materialize/FFI")
    ->Arg(6)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(stage_materialize_api)
    ->Name("stage_materialize/API")
    ->Arg(6)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(stage_validate_ffi)->Name("stage_validate/FFI")->Unit(benchmark::kMicrosecond);
BENCHMARK(stage_validate_api)->Name("stage_validate/API")->Unit(benchmark::kMicrosecond);
