/**
 * @file gguf_planner_probe.cpp
 * @brief Answers two questions about the native stage planner with one run:
 *        does it accept a plain GGUF as a single-shard package, and what does
 *        planning cost?
 *
 * Usage: skippy_api_planner_probe <model.gguf> [layer_count]
 *
 * The tensor inventory is read straight from the GGUF header; nothing here
 * uses llama to parse it, so a shape bug shows up as a planner rejection
 * rather than a misleading success.
 */

#include "llama/backend.h"
#include "llama/stage_planner.h"
#include "stage_load.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <expected>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "llama/model_inventory.h"

namespace {

using namespace skippy::probe;

template <typename F> double milliseconds(F&& work) {
    const auto start = std::chrono::steady_clock::now();
    work();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

double microseconds(std::size_t iterations, const std::function<void()>& work) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < iterations; ++index)
        work();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(stop - start).count() /
           static_cast<double>(iterations);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: skippy_api_planner_probe <model.gguf> [layer_count]\n";
        return 2;
    }
    const std::string path = argv[1];

    const auto inventory = read_inventory(path);
    if (!inventory) {
        std::cerr << "gguf: " << inventory.error() << "\n";
        return 2;
    }
    const std::int64_t layers = argc > 2 ? std::stoll(argv[2]) : layer_count_from(*inventory);
    std::cout << "gguf: " << inventory->size() << " tensors, " << layers << " layers\n";

    std::vector<skippy::PackageTensor> tensors;
    tensors.reserve(inventory->size());
    for (const TensorInfo& info : *inventory) {
        skippy::PackageTensor tensor;
        tensor.id = skippy::TensorId{info.name};
        tensor.native_name = info.name;
        tensor.ggml_type = static_cast<std::int32_t>(info.type);
        tensor.dimension_count = info.dimension_count;
        tensor.dimensions = info.dimensions;
        tensor.data_offset = info.offset;
        tensor.stored_length = info.stored_length;
        tensors.push_back(std::move(tensor));
    }

    skippy::StagePlannerInputs inputs;
    inputs.package = skippy::PackageId{sha256_of(path)};
    inputs.shard_paths = {path};
    inputs.tensors = std::move(tensors);
    // "decode" touches KV state, "embed" does not; asking for both shows the
    // per-family state annotations that patch 0004 added.
    inputs.profiles = {
        skippy::ProfileSpec{
            .id = skippy::ProfileId{"decode"}, .tokens = 512, .sequences = 1, .outputs = 1},
        skippy::ProfileSpec{
            .id = skippy::ProfileId{"embed"}, .tokens = 512, .sequences = 1, .outputs = 512}};
    inputs.graph_configuration = skippy::GraphConfigurationId{"probe"};
    inputs.backend = skippy::BackendId{"metal"};

    skippy::llama::StagePlanner planner;
    double create_ms = 0.0;
    create_ms = milliseconds([&] {
        auto created = skippy::llama::StagePlanner::create(inputs);
        if (!created) {
            std::cerr << "planner rejected: " << created.error().message << "\n";
            return;
        }
        planner = std::move(*created);
    });
    if (!planner.valid()) {
        std::cerr << "create: " << create_ms << " ms (failed)\n";
        return 1;
    }
    std::cout << "create: " << create_ms << " ms\n";

    double realize_ms = 0.0;
    auto plan = std::expected<skippy::llama::StagePlanHandle, skippy::PlanFailure>{};
    realize_ms = milliseconds([&] {
        const auto start = argc > 3 ? std::stoi(argv[3]) : 0;
        const auto end = argc > 4 ? std::stoi(argv[4]) : static_cast<std::int32_t>(layers);
        plan = planner.realize(skippy::LayerRange{start, end});
    });
    if (!plan) {
        std::cerr << "realize rejected: " << plan.error().message << "\n";
        std::cerr << "realize: " << realize_ms << " ms (failed)\n";
        return 1;
    }
    std::cout << "realize: " << realize_ms << " ms\n";

    auto range = plan->range();
    if (!range) {
        std::cerr << "describe rejected: " << range.error().message << "\n";
        return 1;
    }
    std::cout << "range: [" << range->start << ", " << range->end << ")\n";

    const double materialize_ms = milliseconds([&] {
        if (const auto value = plan->materialize(); !value) {
            std::cerr << "materialize rejected: " << value.error().message << "\n";
        }
    });
    auto materialized = plan->materialize();
    if (!materialized) {
        std::cerr << "materialize rejected: " << materialized.error().message << "\n";
        return 1;
    }
    std::cout << "materialized: " << materialized->profiles.size() << " profiles, "
              << materialized->resident_identities.size() << " resident tensors, "
              << materialized->execution_contract.size() << " contract bytes\n";

    for (const skippy::PlanProfile& profile : materialized->profiles) {
        const auto semantics = skippy::summarize(*materialized, profile.id);
        if (!semantics) continue;
        std::cout << "profile " << profile.id.value << ": "
                  << materialized->state_effects_of(profile).size() << " state effects, "
                  << semantics->written << " written, layers [" << semantics->first_layer << ", "
                  << semantics->last_layer
                  << "], recurrent=" << (semantics->recurrent ? "yes" : "no")
                  << ", frontier=" << semantics->frontier_values << "\n";
    }

    // Gate for patch 0003: can the native loader open this stage using only
    // what the plan says (closure, contract, boundary identities)?
    if (argc > 4) {
        const auto load =
            skippy::stage_load_config(*materialized, materialized->profiles.front().id);
        if (!load) {
            std::cout << "staged open: config rejected (" << static_cast<int>(load.error())
                      << ")\n";
        } else {
            auto config = skippy_runtime_config_default();
            config.stage_index = load->stage_index;
            config.layer_start = load->layers.start;
            config.layer_end = load->layers.end;
            config.ctx_size = 512;
            config.n_batch = 512;
            config.n_ubatch = 512;
            config.n_gpu_layers = 999;
            config.load_mode = SKIPPY_LOAD_MODE_LAYER_PACKAGE;
            config.execution_contract = load->execution_contract.data();

            std::vector<const char*> residents;
            for (const std::string_view name : load->resident_identities) {
                residents.push_back(name.data());
            }
            config.resident_tensor_names = residents.empty() ? nullptr : residents.data();
            config.resident_tensor_name_count = residents.size();

            std::vector<const char*> import_ids;
            std::vector<const char*> import_bindings;
            for (const std::string_view name : load->activation_import_identities) {
                import_ids.push_back(name.data());
            }
            for (const std::string_view name : load->activation_import_bindings) {
                import_bindings.push_back(name.data());
            }
            config.activation_import_identities = import_ids.empty() ? nullptr : import_ids.data();
            config.activation_import_identity_count = import_ids.size();
            config.activation_import_bindings =
                import_bindings.empty() ? nullptr : import_bindings.data();

            std::vector<const char*> export_ids;
            std::vector<const char*> export_bindings;
            for (const std::string_view name : load->activation_export_identities) {
                export_ids.push_back(name.data());
            }
            for (const std::string_view name : load->activation_export_bindings) {
                export_bindings.push_back(name.data());
            }
            config.activation_export_identities = export_ids.empty() ? nullptr : export_ids.data();
            config.activation_export_identity_count = export_ids.size();
            config.activation_export_bindings =
                export_bindings.empty() ? nullptr : export_bindings.data();

            skippy_model* model = nullptr;
            skippy_error* error = nullptr;
            const auto start_ms = std::chrono::steady_clock::now();
            const auto status = skippy_model_open(path.c_str(), &config, &model, &error);
            const double open_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - start_ms)
                                       .count();
            if (status != SKIPPY_STATUS_OK) {
                std::cout << "staged open: rejected after " << open_ms << " ms: "
                          << (error != nullptr && error->message != nullptr ? error->message : "?")
                          << "\n";
                if (error != nullptr) skippy_error_free(error);
            } else {
                std::cout << "staged open: ok in " << open_ms << " ms, layers ["
                          << config.layer_start << ", " << config.layer_end << ")\n";
                if (const auto freed = skippy_model_free(model, nullptr);
                    freed != SKIPPY_STATUS_OK) {
                    std::cout << "staged free: status " << static_cast<int>(freed) << "\n";
                }
            }
        }
    }

    // Repeat planning through the one-call entry point: the first call pays
    // inventory, identity and realize; the second is a cache hit.
    {
        const auto start = std::chrono::steady_clock::now();
        static skippy::llama::LlamaBackend probe_backend;
        auto first_lease = probe_backend.plan_model_stages(
            path, std::array<skippy::LayerRange, 1>{
                      skippy::LayerRange{0, static_cast<std::int32_t>(layers)}});
        const double first_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        const auto cached_start = std::chrono::steady_clock::now();
        auto second_lease = probe_backend.plan_model_stages(
            path, std::array<skippy::LayerRange, 1>{
                      skippy::LayerRange{0, static_cast<std::int32_t>(layers)}});
        const double cached_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - cached_start)
                                     .count();
        if (first_lease.has_value() && second_lease.has_value()) {
            std::cout << "plan_model_stages: first " << first_ms << " ms, cached " << cached_ms
                      << " ms\n";
        }
    }

    const double describe_us = microseconds(1000, [&] {
        const auto value = plan->range();
        if (!value) std::abort();
    });
    std::cout << "materialize: " << materialize_ms << " ms\n";
    std::cout << "describe: " << describe_us << " us/call\n";

    return 0;
}
