#include "normalize.h"

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

// Keep the baseline out of line, like the library function.
#ifdef _MSC_VER
#define SKIPPY_NOINLINE __declspec(noinline)
#else
#define SKIPPY_NOINLINE __attribute__((noinline))
#endif

SKIPPY_NOINLINE
void normalize_original(const float* source, float* output, std::size_t dimensions) {
    double squared_norm = 0.0;

    for (std::size_t i = 0; i < dimensions; ++i) {
        squared_norm += static_cast<double>(source[i]) * source[i];
    }

    const float scale =
        squared_norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(squared_norm)) : 1.0F;

    for (std::size_t i = 0; i < dimensions; ++i) {
        output[i] = source[i] * scale;
    }
}

template <auto Normalize> static void BM_Normalize(benchmark::State& state) {
    const auto dimensions = static_cast<std::size_t>(state.range(0));

    std::vector<float> source(dimensions);
    std::vector<float> output(dimensions);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    for (float& value : source) {
        value = distribution(generator);
    }

    // Setup and allocation above are outside the timed loop.
    for (auto _ : state) {
        benchmark::DoNotOptimize(source.data());
        benchmark::DoNotOptimize(output.data());

        Normalize(source.data(), output.data(), dimensions);

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(dimensions));
}

BENCHMARK_TEMPLATE(BM_Normalize, normalize_original)
    ->Name("Embedding/Original")
    ->Arg(2)
    ->Arg(4)
    ->Arg(7)
    ->Arg(384)
    ->Arg(387)
    ->Arg(768)
    ->Arg(771)
    ->Arg(1536)
    ->Arg(4096);

BENCHMARK_TEMPLATE(BM_Normalize, skippy::detail::normalize_embedding)
    ->Name("Embedding/Unrolled")
    ->Arg(2)
    ->Arg(4)
    ->Arg(7)
    ->Arg(384)
    ->Arg(387)
    ->Arg(768)
    ->Arg(771)
    ->Arg(1536)
    ->Arg(4096);
