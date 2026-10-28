#include "fixtures.h"

#include <iostream>

static void ok(skippy_status status, skippy_error * error) {
    const std::string detail = error ? error->message : "native verification failed";
    skippy_error_free(error);
    require(status == SKIPPY_STATUS_OK, detail);
}

static void verify_case(skippy_model * model, bool lazy, bool reject, bool stochastic) {
    skippy_session * session = nullptr;
    skippy_error * error = nullptr;
    auto status = skippy_session_create(model, &session, &error);
    ok(status, error);
    auto sampling = skippy_config(73);
    if (!stochastic) { sampling.temperature = 0.0f; }
    sampling.logit_bias_count = 1;
    sampling.logit_bias[0] = {42, 100000.0f};
    const std::vector<llama_token> prefix(8, 1);
    const char * metadata = lazy
        ? R"({"grammar":"root ::= <[42]> <[43]> <[44]>","grammar_lazy":true,"grammar_triggers":[{"type":0,"token":42}]})"
        : R"({"grammar":"root ::= <[42]> <[43]> <[44]>"})";
    error = nullptr;
    status = skippy_session_configure_chat_sampling(session, &sampling, metadata, prefix.size() + 1, &error);
    ok(status, error);
    size_t bytes = 0;
    error = nullptr;
    status = skippy_prefill_chunk(session, prefix.data(), prefix.size(), nullptr, 0, nullptr, 0, &bytes, &error);
    ok(status, error);
    const std::vector<llama_token> proposal{1, 42, reject ? 44 : 43};
    llama_token predicted[3]{};
    size_t count = 0;
    error = nullptr;
    status = skippy_verify_tokens_frame_sampled(session, proposal.data(), proposal.size(), &sampling,
        nullptr, nullptr, nullptr, nullptr, 0, &bytes, predicted, 3, &count, 0, nullptr, &error);
    ok(status, error);
    require(count == (reject ? 2u : 3u), "verification did not stop at the rejected suffix");
    require(predicted[0] == 42 && predicted[1] == 43, "incorrect verified prefix");
    if (!reject) { require(predicted[2] == 44, "incorrect accepted draft"); }
    auto expected_history = prefix;
    expected_history.insert(expected_history.end(), proposal.begin(), proposal.begin() + count);
    require(session->token_history == expected_history, "rejected token entered committed sampler history");
    require(session->sampling_accepted_token_count == expected_history.size(), "sampler history drift");
    error = nullptr;
    if (reject) {
        status = skippy_trim_session(session, prefix.size() + count, &error);
    } else {
        status = skippy_retire_verify_checkpoint(session, prefix.size(), proposal.size(), &error);
    }
    ok(status, error);
    require(session->verify_checkpoints.empty(), "verify checkpoint leak");
    require(session->token_history == expected_history, "trim changed the accepted prefix");
    if (session->grammar_sampler == nullptr) {
        // A recurrent checkpoint restore can invalidate the cached sampler.
        // Reconfigure as the serving host does, then replay committed history.
        error = nullptr;
        status = skippy_session_configure_chat_sampling(session, &sampling, metadata, prefix.size() + 1, &error);
        ok(status, error);
    }
    llama_token next = LLAMA_TOKEN_NULL;
    error = nullptr;
    status = skippy_decode_step_sampled(session, reject ? 43 : 44, &sampling,
        nullptr, 0, nullptr, 0, &bytes, &next, &error);
    ok(status, error);
    require(session->grammar_sampler != nullptr, "grammar disabled after recovery");
    require(reject ? next == 44 : llama_vocab_is_eog(llama_model_get_vocab(model->model), next), "incorrect continuation after verification");
    error = nullptr;
    status = skippy_session_free(session, &error);
    ok(status, error);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "usage: grammar-verification MODEL");
        auto config = skippy_runtime_config_default();
        config.ctx_size = 512;
        config.n_batch = 64;
        config.n_ubatch = 64;
        config.n_threads = 8;
        config.n_threads_batch = 8;
        config.n_gpu_layers = 999;
        skippy_model * model = nullptr;
        skippy_error * error = nullptr;
        const auto status = skippy_model_open(argv[1], &config, &model, &error);
        ok(status, error);
        int cases = 0;
        for (int repeat = 0; repeat < 4; ++repeat) {
            for (bool lazy : {false, true}) {
                for (bool reject : {false, true}) {
                    for (bool stochastic : {false, true}) {
                        verify_case(model, lazy, reject, stochastic);
                        ++cases;
                    }
                }
            }
        }
        error = nullptr;
        const auto freed = skippy_model_free(model, &error);
        ok(freed, error);
        std::cout << json({{"verification_cases", cases}, {"accepted_and_rejected_proposals", true},
                           {"grammar_recovery", true}, {"checkpoint_cleanup", true},
                           {"scope", "MTP target verification transaction; controlled proposals, not draft-head quality"}}) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "grammar-verification: " << error.what() << '\n';
        return 1;
    }
}
