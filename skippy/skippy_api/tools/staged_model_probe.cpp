/// Opens a stage through the facade and reports what it can run.
///
/// Proves the staged path end to end at the facade level: plan, own, open,
/// session, workload info. It prints rather than asserts so a failure shows
/// the native message instead of a stack trace.

#include "llama/backend.h"
#include "llama/model_inventory.h"
#include "llama/stage_planner.h"
#include "llama/staged.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
// bounded strtol: a probe count from argv; a non-numeric argument keeps the default
int parse_int(const char* text, int fallback) noexcept {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) return fallback;
    return static_cast<int>(value); // bounded by the range check above
}
} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape): a probe main reads as a script; an unexpected throw terminating with the message is the diagnostic
int main(
    int argc,
    char**
        argv) { // NOLINT(bugprone-exception-escape): a probe that throws has nothing to keep alive; terminating with the message is the diagnostic
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> [stage:0] [layers]\n";
        return 2;
    }
    const std::string path = argv[1];
    const auto stage_index = static_cast<std::uint32_t>(
        argc > 2 ? parse_int(argv[2], 0) : 0); // bounded by parse_int's [0, INT_MAX] check

    auto inventory = skippy::probe::cached_inventory(path);
    if (!inventory) {
        std::cerr << "inventory: " << inventory.error() << "\n";
        return 1;
    }
    const auto layers = skippy::probe::layer_count_from(*inventory);
    const std::int32_t half = layers / 2;
    const std::array ranges{
        skippy::LayerRange{.start = 0, .end = half},
        skippy::LayerRange{.start = half, .end = layers},
    };

    skippy::llama::LlamaBackend backend;
    auto lease = backend.plan_model_stages(path, ranges);
    if (!lease) {
        std::cerr << "plan: " << lease.error().message << "\n";
        return 1;
    }
    if (stage_index >= lease->configs.size()) {
        std::cerr << "stage " << stage_index << " is out of range\n";
        return 2;
    }

    const auto spec = skippy::llama::own(lease->configs[stage_index]);
    auto model = skippy::llama::StagedModel::open(path, spec);
    if (!model) {
        std::cerr << "open: " << model.error().message << " (status " << model.error().status
                  << ")\n";
        return 1;
    }
    std::cout << "stage " << stage_index << " opened, layers [" << model->layers().start << ", "
              << model->layers().end << ")\n";

    auto session = model->create_session();
    if (!session) {
        std::cerr << "session: " << session.error().message << "\n";
        return 1;
    }

    skippy_workload_info_v1 info{};
    info.abi_version = SKIPPY_WORKLOAD_INFO_V1_ABI_VERSION;
    info.struct_size = sizeof(info);
    skippy_error* error = nullptr;
    if (skippy_model_workload_info_v1(model->native(), &info, &error) != SKIPPY_STATUS_OK) {
        std::cerr << "workload info failed\n";
        if (error != nullptr) skippy_error_free(error);
        return 1;
    }
    std::cout << "workload: kind=" << static_cast<unsigned>(info.kind)
              << " pooling=" << static_cast<unsigned>(info.pooling)
              << " classifier=" << info.classifier_outputs << "\n";
    return 0;
}
