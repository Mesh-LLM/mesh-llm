// Patch 0014 A/B: the temperature-0 argmax fast path, raw C ABI (the
// native `skippy_select_unique_argmax` the runtime itself runs) against the
// facade mirror in `skippy/argmax_sampling.h`. Pure computation: no model,
// no session, deterministic logits. Gated on identical tokens before and
// after the timed loop.

#include "argmax_sampling.h"

#include "sampling_argmax.h"

#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

constexpr std::int32_t kVocab = 151936; // qwen2.5 vocabulary

const std::vector<float>& logits() {
    static const std::vector<float> values = [] {
        std::vector<float> buffer(static_cast<std::size_t>(kVocab), -1.0f);
        buffer[100] = 4.0f;    // plain unique maximum
        buffer[151000] = 3.0f; // biased to win, below
        buffer[151001] = 4.0f; // ties the plain maximum unless biased
        return buffer;
    }();
    return values;
}

const std::vector<llama_logit_bias>& native_biases() {
    static const std::vector<llama_logit_bias> values{
        {151000, 50.0f}, {151001, -1.0f}, {151000, 0.25f}, {-1, 4.0f}, {kVocab, 9.0f},
    };
    return values;
}

const std::vector<skippy::LogitBias>& facade_biases() {
    static const std::vector<skippy::LogitBias> values{
        skippy::LogitBias{.token = skippy::Token{151000}, .bias = 50.0f},
        skippy::LogitBias{.token = skippy::Token{151001}, .bias = -1.0f},
        skippy::LogitBias{.token = skippy::Token{151000}, .bias = 0.25f},
        skippy::LogitBias{.token = skippy::Token{-1}, .bias = 4.0f},
        skippy::LogitBias{.token = skippy::Token{kVocab}, .bias = 9.0f},
    };
    return values;
}

/// Both sides must answer 151000: plain max 100 scores 4.0, 151001 ties at
/// 4.0 after -1.0 (3.0), and 151000 wins with 53.25 after two biases.
bool answers_agree(llama_token native_token, std::optional<skippy::Token> facade_token) {
    return native_token == 151000 && facade_token == skippy::Token{151000};
}

void argmax_select_ffi(benchmark::State& state) {
    skippy_argmax_plan plan;
    skippy_prepare_argmax_plan(&plan, native_biases(), kVocab);

    skippy_sampling_config config{};
    config.temperature = 0.0f;
    if (!skippy_sampling_selects_argmax(&config) ||
        !answers_agree(skippy_select_unique_argmax(logits().data(), kVocab, plan),
                       std::optional<skippy::Token>{skippy::Token{151000}})) {
        state.SkipWithError("native arm disagrees");
        return;
    }
    llama_token token = 0;
    for (auto _ : state) {
        token = skippy_select_unique_argmax(logits().data(), kVocab, plan);
    }
    if (token != 151000) {
        state.SkipWithError("native arm diverged during the run");
    }
    state.SetItemsProcessed(state.iterations());
}

void argmax_select_api(benchmark::State& state) {
    const skippy::ArgmaxPlan plan{facade_biases(), kVocab};

    skippy::ArgmaxEligibility knobs{};
    if (!skippy::selects_argmax(knobs) || !answers_agree(151000, plan.select_unique(logits()))) {
        state.SkipWithError("facade arm disagrees");
        return;
    }
    std::optional<skippy::Token> token;
    for (auto _ : state) {
        token = plan.select_unique(logits());
    }
    if (token != skippy::Token{151000}) {
        state.SkipWithError("facade arm diverged during the run");
    }
    state.SetItemsProcessed(state.iterations());
}

void argmax_eligible_ffi(benchmark::State& state) {
    skippy_sampling_config config{};
    config.temperature = 0.0f;
    for (auto _ : state) {
        benchmark::DoNotOptimize(skippy_sampling_selects_argmax(&config));
    }
}

void argmax_eligible_api(benchmark::State& state) {
    skippy::ArgmaxEligibility knobs{};
    for (auto _ : state) {
        benchmark::DoNotOptimize(skippy::selects_argmax(knobs));
    }
}

} // namespace

BENCHMARK(argmax_select_ffi)->Name("Argmax/Select/151936/FFI");
BENCHMARK(argmax_select_api)->Name("Argmax/Select/151936/API");
BENCHMARK(argmax_eligible_ffi)->Name("Argmax/Eligible/FFI");
BENCHMARK(argmax_eligible_api)->Name("Argmax/Eligible/API");
