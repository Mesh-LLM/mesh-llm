#include "llama/causal_readout.h"

#include "skippy/execution.h"

#include <vector>

namespace skippy::llama {

bool supports_causal_readout(const StagedModel& model) noexcept {
    return skippy_model_supports_causal_readout(model.native());
}

std::expected<void, ReadoutFailure> read_selected_log_probs(StagedSession& session,
                                                            std::span<const skippy::Token> tokens,
                                                            std::span<const skippy::Token> selected,
                                                            std::span<float> out_scores) {
    if (selected.empty() || out_scores.size() < selected.size() ||
        selected.size() > skippy::detail::max_selected_tokens) {
        return std::unexpected(ReadoutFailure{
            .status = static_cast<int>(SKIPPY_STATUS_INVALID_ARGUMENT),
            .message = "selected readout exceeds its bound or the output is undersized",
        });
    }
    // The native ABI speaks int32 tokens; convert once, here, at the boundary.
    std::vector<llama_token> fed;
    fed.reserve(tokens.size());
    for (const skippy::Token token : tokens)
        fed.push_back(static_cast<llama_token>(token));
    std::vector<llama_token> ids;
    ids.reserve(selected.size());
    for (const skippy::Token token : selected)
        ids.push_back(static_cast<llama_token>(token));

    skippy_token_score_request score{
        .token_ids = ids.data(),
        .token_count = ids.size(),
        .out_log_probabilities = out_scores.data(),
    };
    skippy_iteration_request request{
        .session = session.native(),
        .token_ids = fed.data(),
        .token_count = fed.size(),
        .positions = nullptr,
        .position_count = 0,
        .sampling = nullptr,
        .input_desc = nullptr,
        .input_payload = nullptr,
        .sample_last = false,
        .readout = &score,
    };
    size_t sampled_count = 0;
    skippy_error* error = nullptr;
    const skippy_status status =
        skippy_iteration_batch_sampled(&request, 1, nullptr, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, 0, &sampled_count, &error);
    if (status != SKIPPY_STATUS_OK) {
        ReadoutFailure failure{
            .status = static_cast<int>(status),
            .message = error != nullptr && error->message != nullptr ? error->message : "",
        };
        if (error != nullptr) skippy_error_free(error);
        return std::unexpected(std::move(failure));
    }
    return {};
}

} // namespace skippy::llama
