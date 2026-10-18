/// Opens a stage through the facade and reports what it can run.
///
/// Proves the staged path end to end at the facade level: plan, own, open,
/// session, workload info. It prints rather than asserts so a failure shows
/// the native message instead of a stack trace.

#include "llama/model_inventory.h"
#include "llama/backend.h"
#include "llama/stage_planner.h"
#include "llama/staged.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> [stage:0] [layers]\n";
        return 2;
    }
    const std::string path = argv[1];
    const auto stage_index = static_cast<std::uint32_t>(argc > 2 ? std::atoi(argv[2]) : 0);

    auto inventory = skippy::probe::cached_inventory(path);
    if (!inventory) {
        std::cerr << "inventory: " << inventory.error() << "\n";
        return 1;
    }
    const auto layers = static_cast<std::int32_t>(skippy::probe::layer_count_from(*inventory));
    const std::int32_t half = layers / 2;
    const std::array ranges{skippy::LayerRange{0, half}, skippy::LayerRange{half, layers}};

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
