/**
 * @file stage_pipeline_probe.cpp
 * @brief Runs a model as N chained stages and checks the chain against one stage.
 *
 * Usage: skippy_api_stage_pipeline_probe <model.gguf> <stages> [layers]
 *
 * Each stage is planned, configured and opened from its own plan, then fed the
 * previous stage's activation payload. The last stage's payload is compared
 * with the payload of a single stage covering the whole model: if the chain
 * composes, the bytes agree, which is the correctness question a split model
 * actually has.
 *
 * The payload check is extended by a K-step greedy continuation: both the
 * chain and an unstaged reference keep decoding their own predictions, and the
 * continuations must agree at every step. One matching token is a single
 * sample; K matching tokens is composition holding under feedback.
 */

#include "llama/model_inventory.h"
#include "llama/stage_planner.h"
#include "stage_load.h"

#include "skippy.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace skippy::probe;
using Clock = std::chrono::steady_clock;

double millis(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct OwnedStrings {
    std::vector<std::string> storage;
    std::vector<const char*> pointers;

    void add(const std::string_view value) { storage.emplace_back(value); }
    void seal() {
        pointers.clear();
        pointers.reserve(storage.size());
        for (const std::string& value : storage)
            pointers.push_back(value.c_str());
    }
};

struct Stage {
    skippy::LayerRange range;
    skippy::llama::StagePlanHandle plan;
    skippy_model* model = nullptr;
    skippy_session* session = nullptr;
    skippy_activation_desc output_desc{};
    std::vector<std::uint8_t> output;

    Stage() = default;
    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;
    Stage(Stage&& other) noexcept
        : range(other.range), plan(std::move(other.plan)),
          model(std::exchange(other.model, nullptr)),
          session(std::exchange(other.session, nullptr)), output_desc(other.output_desc),
          output(std::move(other.output)) {}
    Stage& operator=(Stage&&) = delete;
    ~Stage() {
        if (session != nullptr) skippy_session_free(session, nullptr);
        if (model != nullptr) skippy_model_free(model, nullptr);
    }
};

std::string message_of(enum skippy_status status, skippy_error* error) {
    std::string text = "status " + std::to_string(static_cast<int>(status));
    if (error != nullptr) {
        if (error->message != nullptr) text += ": " + std::string(error->message);
        skippy_error_free(error);
    }
    return text;
}

/// Opens one stage: planner -> plan -> loader config -> model -> session.
bool open_stage(skippy::llama::StagePlanner& planner, const std::string& path,
                skippy::LayerRange range, Stage& stage) {
    stage.range = range;
    auto plan = planner.realize(range);
    if (!plan) {
        std::cerr << "realize [" << range.start << "," << range.end << "): " << plan.error().message
                  << "\n";
        return false;
    }
    stage.plan = std::move(*plan);

    auto materialized = stage.plan.materialize();
    if (!materialized) {
        std::cerr << "materialize: " << materialized.error().message << "\n";
        return false;
    }
    auto load = skippy::stage_load_config(*materialized, materialized->profiles.front().id);
    if (!load) {
        std::cerr << "stage load config rejected\n";
        return false;
    }

    OwnedStrings residents;
    for (const std::string_view name : load->resident_identities)
        residents.add(name);
    residents.seal();
    OwnedStrings imports;
    OwnedStrings import_bindings;
    for (const std::string_view name : load->activation_import_identities)
        imports.add(name);
    for (const std::string_view name : load->activation_import_bindings)
        import_bindings.add(name);
    imports.seal();
    import_bindings.seal();
    OwnedStrings exports;
    OwnedStrings export_bindings;
    for (const std::string_view name : load->activation_export_identities)
        exports.add(name);
    for (const std::string_view name : load->activation_export_bindings)
        export_bindings.add(name);
    exports.seal();
    export_bindings.seal();

    auto config = skippy_runtime_config_default();
    config.stage_index = 0;
    config.layer_start = range.start;
    config.layer_end = range.end;
    config.ctx_size = 512;
    config.n_batch = 512;
    config.n_ubatch = 512;
    config.n_gpu_layers = 999;
    config.load_mode = SKIPPY_LOAD_MODE_LAYER_PACKAGE;
    config.execution_contract = load->execution_contract.data();
    config.resident_tensor_names = residents.pointers.data();
    config.resident_tensor_name_count = residents.pointers.size();
    config.activation_import_identities =
        imports.pointers.empty() ? nullptr : imports.pointers.data();
    config.activation_import_identity_count = imports.pointers.size();
    config.activation_import_bindings =
        import_bindings.pointers.empty() ? nullptr : import_bindings.pointers.data();
    config.activation_export_identities =
        exports.pointers.empty() ? nullptr : exports.pointers.data();
    config.activation_export_identity_count = exports.pointers.size();
    config.activation_export_bindings =
        export_bindings.pointers.empty() ? nullptr : export_bindings.pointers.data();

    skippy_error* error = nullptr;
    if (skippy_model_open(path.c_str(), &config, &stage.model, &error) != SKIPPY_STATUS_OK) {
        std::cerr << "open [" << range.start << "," << range.end
                  << "): " << message_of(SKIPPY_STATUS_ERROR, error) << "\n";
        return false;
    }
    if (skippy_session_create(stage.model, &stage.session, &error) != SKIPPY_STATUS_OK) {
        std::cerr << "session [" << range.start << "," << range.end
                  << "): " << message_of(SKIPPY_STATUS_ERROR, error) << "\n";
        return false;
    }
    return true;
}

/// Prefills one stage, growing the output buffer when the runtime asks.
std::size_t boundary_bytes(const skippy_activation_boundary_desc& boundary) {
    std::size_t total = 0;
    for (std::uint32_t index = 0; index < boundary.part_count; ++index) {
        total = std::max<std::size_t>(total, boundary.parts[index].payload_offset +
                                                 boundary.parts[index].payload_bytes);
    }
    return total;
}

bool run_stage(Stage& stage, const std::vector<llama_token>& tokens,
               const skippy_activation_desc* input_desc, const std::uint8_t* input) {
    skippy_activation_boundary_desc boundary{};
    std::size_t needed = 0;
    if (skippy_model_output_activation_boundary(stage.model, &boundary)) {
        needed = boundary_bytes(boundary);
    }
    std::cout << "  boundary: parts=" << boundary.part_count << " bytes=" << needed << "\n";

    for (int attempt = 0; attempt < 3; ++attempt) {
        stage.output.resize(needed == 0 ? (1U << 20U) : needed);
        skippy_error* error = nullptr;
        const auto status = skippy_prefill_chunk_frame(
            stage.session, tokens.data(), tokens.size(), input_desc, input, &stage.output_desc,
            stage.output.data(), stage.output.size(), &needed, &error);
        if (status == SKIPPY_STATUS_OK) {
            stage.output.resize(needed);
            return true;
        }
        if (status == SKIPPY_STATUS_BUFFER_TOO_SMALL && needed > 0) {
            if (error != nullptr) skippy_error_free(error);
            continue;
        }
        std::cerr << "prefill [" << stage.range.start << "," << stage.range.end
                  << "): " << message_of(status, error) << "\n";
        return false;
    }
    std::cerr << "prefill: buffer sizing did not converge\n";
    return false;
}

std::size_t payload_size(const Stage& stage) { return stage.output.size(); }

std::string compare(const std::vector<std::uint8_t>& left, const std::vector<std::uint8_t>& right) {
    if (left.size() != right.size()) {
        return "size " + std::to_string(left.size()) + " != " + std::to_string(right.size());
    }
    if (left.empty()) return "both payloads are empty";
    auto* lhs = reinterpret_cast<float*>(const_cast<std::uint8_t*>(left.data()));
    auto* rhs = reinterpret_cast<float*>(const_cast<std::uint8_t*>(right.data()));
    const std::size_t count = left.size() / sizeof(float);
    double worst = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        worst = std::max(
            worst, std::abs(static_cast<double>(lhs[index]) - static_cast<double>(rhs[index])));
    }
    return "max abs difference " + std::to_string(worst) + " over " + std::to_string(count) +
           " floats";
}

} // namespace

std::expected<std::vector<llama_token>, std::string> tokenize(skippy_model* model,
                                                              const std::string& text) {
    std::vector<llama_token> tokens(512);
    std::size_t count = 0;
    skippy_error* error = nullptr;
    auto status =
        skippy_tokenize(model, text.c_str(), true, tokens.data(), tokens.size(), &count, &error);
    if (status == SKIPPY_STATUS_BUFFER_TOO_SMALL) {
        if (error != nullptr) skippy_error_free(error);
        error = nullptr;
        tokens.resize(count);
        status = skippy_tokenize(model, text.c_str(), true, tokens.data(), tokens.size(), &count,
                                 &error);
    }
    if (status != SKIPPY_STATUS_OK) return std::unexpected(message_of(status, error));
    tokens.resize(count);
    return tokens;
}

skippy_sampling_config greedy_sampling() {
    skippy_sampling_config sampling{};
    sampling.version = 1;
    sampling.flags =
        SKIPPY_SAMPLING_FLAG_ENABLED; // NOLINT(bugprone-signed-bitwise): the C ABI declares flags as int and the macro is that ABI
    sampling.seed = 0;
    sampling.temperature = 1.0F;
    sampling.top_k = 1;
    sampling.top_p = 1.0F;
    sampling.sampler_count = 1;
    sampling.samplers[0] = SKIPPY_SAMPLER_TOP_K;
    return sampling;
}

/// Decodes one token through every stage, carrying the activation forward.
///
/// A non-first slice needs the upstream activation on every call, prefill or
/// decode, because the runtime keeps no incoming frame between calls.
std::expected<llama_token, std::string> predict_chain(std::vector<Stage>& stages, llama_token next,
                                                      std::vector<std::uint8_t>* terminal_frame) {
    const auto sampling = greedy_sampling();
    const skippy_activation_desc* input_desc = nullptr;
    const void* input = nullptr;
    std::vector<std::uint8_t> carried;
    llama_token predicted = -1;

    for (Stage& stage : stages) {
        // Size the decode frame from the stage's own activation boundary and
        // retry on BUFFER_TOO_SMALL, exactly like the prefill path: a fixed
        // 1 MiB guess fails misleadingly on a larger frame.
        std::size_t needed = 0;
        skippy_activation_boundary_desc boundary{};
        if (skippy_model_output_activation_boundary(stage.model, &boundary)) {
            needed = boundary_bytes(boundary);
        }
        std::vector<std::uint8_t> output(needed == 0 ? (1U << 20U) : needed);
        std::size_t bytes = 0;
        skippy_error* error = nullptr;
        const auto status = skippy_decode_step_frame_sampled(
            stage.session, next, &sampling, input_desc, input, &stage.output_desc, output.data(),
            output.size(), &bytes, &predicted, &error);
        if (status == SKIPPY_STATUS_BUFFER_TOO_SMALL && bytes > 0) {
            if (error != nullptr) skippy_error_free(error);
            output.resize(bytes);
            skippy_error* retry = nullptr;
            const auto again = skippy_decode_step_frame_sampled(
                stage.session, next, &sampling, input_desc, input, &stage.output_desc,
                output.data(), output.size(), &bytes, &predicted, &retry);
            if (again != SKIPPY_STATUS_OK) return std::unexpected(message_of(again, retry));
        } else if (status != SKIPPY_STATUS_OK) {
            return std::unexpected(message_of(status, error));
        }
        output.resize(bytes);
        carried = std::move(output);
        input_desc = &stage.output_desc;
        input = carried.data();
    }
    if (terminal_frame != nullptr) *terminal_frame = std::move(carried);
    return predicted;
}

std::vector<skippy::LayerRange> partition(std::int32_t layers, int count) {
    std::vector<skippy::LayerRange> ranges;
    const std::int32_t base = layers / count;
    const std::int32_t remainder = layers % count;
    std::int32_t start = 0;
    for (int index = 0; index < count; ++index) {
        const std::int32_t width = base + (index < remainder ? 1 : 0);
        ranges.push_back(skippy::LayerRange{.start = start, .end = start + width});
        start += width;
    }
    return ranges;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity,bugprone-exception-escape): a probe main reads as a script; an unexpected throw terminating with the message is the diagnostic
int main(
    int argc,
    char**
        argv) { // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape): a probe main reads as a script; an unexpected throw terminating with the message is the diagnostic
    if (argc < 3) {
        std::cerr << "usage: skippy_api_stage_pipeline_probe <model.gguf> <stages> [layers] [text] "
                     "[steps]\n";
        return 2;
    }
    const std::string path = argv[1];
    const int stage_count = std::stoi(argv[2]);
    if (stage_count < 1) {
        std::cerr << "need at least one stage\n";
        return 2;
    }

    auto inventory = skippy::probe::read_inventory(path);
    if (!inventory) {
        std::cerr << "gguf: " << inventory.error() << "\n";
        return 2;
    }
    const std::int32_t layers =
        argc > 3 ? std::stoi(argv[3]) : skippy::probe::layer_count_from(*inventory);
    const std::string text = argc > 4 ? argv[4] : "the quick brown fox jumps over the lazy dog";

    std::vector<skippy::PackageTensor> tensors;
    tensors.reserve(inventory->size());
    for (const auto& info : *inventory) {
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
    inputs.package = skippy::PackageId{skippy::probe::sha256_of(path)};
    inputs.shard_paths = {path};
    inputs.tensors = std::move(tensors);
    inputs.profiles = {
        skippy::ProfileSpec{
            .id = skippy::ProfileId{"decode"},
            .tokens = 512,
            .sequences = 1,
            .outputs = 512,
        },
    };
    inputs.graph_configuration = skippy::GraphConfigurationId{"pipeline"};
    inputs.backend = skippy::BackendId{"metal"};

    auto planner = skippy::llama::StagePlanner::create(inputs);
    if (!planner) {
        std::cerr << "planner: " << planner.error().message << "\n";
        return 1;
    }

    const std::vector<skippy::LayerRange> ranges = partition(layers, stage_count);
    std::vector<Stage> stages(static_cast<std::size_t>(stage_count));
    for (int index = 0; index < stage_count; ++index) {
        if (!open_stage(*planner, path, ranges[static_cast<std::size_t>(index)],
                        stages[static_cast<std::size_t>(index)])) {
            return 1;
        }
    }

    const auto tokens = tokenize(stages.front().model, text);
    if (!tokens) {
        std::cerr << "tokenize: " << tokens.error() << "\n";
        return 1;
    }
    std::cout << "tokens: " << tokens->size() << "\n";
    if (tokens->size() < 2) {
        std::cerr << "need at least two tokens: one of context and one to predict\n";
        return 2;
    }

    // Chain: stage 0 sees no input, each later stage consumes the previous one.
    // The chain and the reference below both predict the token after the text,
    // so every prefill takes the context without the final token and the final
    // token is decoded exactly once, by predict_chain.
    const std::vector<llama_token> context(tokens->begin(), tokens->end() - 1);
    const llama_token next = tokens->back();

    const skippy_activation_desc* input_desc = nullptr;
    const std::uint8_t* input = nullptr;
    std::vector<std::uint8_t> carried;
    for (int index = 0; index < stage_count; ++index) {
        Stage& stage = stages[static_cast<std::size_t>(index)];
        const auto start = Clock::now();
        if (!run_stage(stage, context, input_desc, input)) return 1;
        std::cout << "stage " << index << " [" << stage.range.start << ", " << stage.range.end
                  << "): " << payload_size(stage) << " bytes in " << millis(start) << " ms\n";
        carried = stage.output;
        input_desc = &stage.output_desc;
        input = carried.data();
    }

    if (stage_count == 1) return 0;

    // Composition check: one stage holding every layer must emit the same
    // terminal payload as the chain.
    Stage monolithic;
    if (!open_stage(*planner, path, skippy::LayerRange{.start = 0, .end = layers}, monolithic))
        return 1;
    if (!run_stage(monolithic, context, nullptr, nullptr)) return 1;
    std::cout << "single stage: " << payload_size(monolithic) << " bytes\n";
    std::cout << "chain vs single: " << compare(stages.back().output, monolithic.output) << "\n";
    // K-step greedy continuation: both paths prefilled `context` above, so
    // step 0 decodes the text's final token on both. Every later step feeds
    // each path its own prediction back, so the paths stay comparable exactly
    // as long as they keep agreeing; the run fails at the first divergence.
    const int steps = argc > 5 ? std::stoi(argv[5]) : 16;
    if (steps < 1) {
        std::cerr << "need at least one continuation step\n";
        return 2;
    }

    Stage full;
    auto full_config = skippy_runtime_config_default();
    full_config.ctx_size = 512;
    full_config.n_batch = 512;
    full_config.n_ubatch = 512;
    full_config.n_gpu_layers = 999;
    skippy_error* error = nullptr;
    if (skippy_model_open(path.c_str(), &full_config, &full.model, &error) != SKIPPY_STATUS_OK) {
        std::cerr << "full open: " << message_of(SKIPPY_STATUS_ERROR, error) << "\n";
        return 1;
    }
    if (skippy_session_create(full.model, &full.session, &error) != SKIPPY_STATUS_OK) {
        std::cerr << "full session: " << message_of(SKIPPY_STATUS_ERROR, error) << "\n";
        return 1;
    }
    if (!run_stage(full, context, nullptr, nullptr)) return 1;
    std::vector<Stage> one;
    one.reserve(1);
    one.emplace_back(std::move(full));

    llama_token chain_feed = next;
    llama_token full_feed = next;
    std::vector<llama_token> chain_continuation;
    std::vector<llama_token> full_continuation;
    bool diverged = false;
    for (int step = 0; step < steps; ++step) {
        std::vector<std::uint8_t> chain_frame;
        std::vector<std::uint8_t> full_frame;
        const auto chain_token = predict_chain(stages, chain_feed, &chain_frame);
        if (!chain_token) {
            std::cerr << "chain predict: " << chain_token.error() << "\n";
            return 1;
        }
        const auto full_token = predict_chain(one, full_feed, &full_frame);
        if (!full_token) {
            std::cerr << "full predict: " << full_token.error() << "\n";
            return 1;
        }
        chain_continuation.push_back(*chain_token);
        full_continuation.push_back(*full_token);
        if (*chain_token != *full_token) {
            std::cout << "step " << step << ": chain=" << *chain_token << " full=" << *full_token
                      << " -> first divergence\n";
            diverged = true;
            break;
        }
        std::cout << "step " << step << ": token " << *chain_token << ", terminal frame "
                  << compare(chain_frame, full_frame) << "\n";
        chain_feed = *chain_token;
        full_feed = *full_token;
    }

    auto render = [](const std::vector<llama_token>& tokens) {
        std::string text;
        for (const llama_token token : tokens)
            text += std::to_string(token) + " ";
        return text;
    };
    std::cout << "chain continuation: " << render(chain_continuation) << "\n";
    std::cout << "full continuation:  " << render(full_continuation) << "\n";
    std::cout << steps << "-step greedy continuation: " << (diverged ? "MISMATCH" : "MATCH")
              << "\n";
    return diverged ? 1 : 0;
}
