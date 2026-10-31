#include "llama/stage_chain.h"

#include "llama/backend.h"
#include "llama/stage_planner.h"

#include <cstddef>
#include <utility>

namespace skippy::llama {
namespace {} // namespace

std::expected<StageChain, StagedFailure> StageChain::open(const std::string& path,
                                                          std::span<const LayerRange> ranges) {
    auto lease = LlamaBackend{}.plan_model_stages(path, ranges);
    if (!lease) {
        return std::unexpected(StagedFailure{.status = 0, .message = lease.error().message});
    }
    // The lease pins the plans the configs borrow from, so it outlives the
    // temporary backend that produced them.

    StageChain chain;
    chain.stages_.reserve(lease->configs.size());
    for (const StageLoadConfig& config : lease->configs) {
        auto model = StagedModel::open(path, own(config));
        if (!model) return std::unexpected(model.error());
        auto session = model->create_session();
        if (!session) return std::unexpected(session.error());
        chain.stages_.push_back(Stage{
            .model = std::move(*model),
            .session = std::move(*session),
            .out = {},
        });
    }
    return chain;
}

std::expected<void, StagedFailure> StageChain::run_stage(std::size_t index,
                                                         std::span<const llama_token> tokens,
                                                         const PreparedExport* input,
                                                         std::size_t hint) {
    Stage& stage = stages_[index];
    skippy_activation_desc input_desc{};
    if (input != nullptr) input_desc = input->desc;

    auto buffer = stage.out.reserve(hint);
    std::size_t needed = 0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        skippy_error* error = nullptr;
        const auto status = skippy_prefill_chunk_frame(
            stage.session.native(), tokens.data(), tokens.size(),
            input == nullptr ? nullptr : &input_desc,
            input == nullptr ? nullptr : input->written().data(), &stage.out.desc, buffer.data(),
            buffer.size(), &needed, &error);
        if (status == SKIPPY_STATUS_OK) {
            stage.out.set_written(needed);
            return {};
        }
        StagedFailure failure{.status = static_cast<int>(status), .message = {}};
        if (error != nullptr) {
            if (error->message != nullptr) failure.message = error->message;
            skippy_error_free(error);
        }
        if (status != SKIPPY_STATUS_BUFFER_TOO_SMALL || needed == 0) {
            return std::unexpected(failure);
        }
        buffer = stage.out.reserve(needed);
    }
    return std::unexpected(
        StagedFailure{.status = 0, .message = "activation frame sizing did not converge"});
}

std::optional<StagedFailure> StageChain::reset() {
    for (Stage& stage : stages_) {
        skippy_error* error = nullptr;
        const auto status = skippy_session_reset(stage.session.native(), &error);
        if (status != SKIPPY_STATUS_OK) {
            StagedFailure failure{.status = static_cast<int>(status), .message = {}};
            if (error != nullptr) {
                if (error->message != nullptr) failure.message = error->message;
                skippy_error_free(error);
            }
            return failure;
        }
    }
    return std::nullopt;
}

} // namespace skippy::llama
