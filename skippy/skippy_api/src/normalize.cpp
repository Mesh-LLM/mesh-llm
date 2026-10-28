#include "normalize.h"

#include <cmath>
#include <cstddef>

namespace skippy::detail {

void normalize_embedding(const float* source, float* output, std::size_t dimensions) noexcept {
    // Four independent accumulators so the compiler has lanes to vectorise with;
    // not every compiler does this from a single accumulator. Native SIMD is
    // deliberately deferred: normalisation is ~0.005% of an embed call, so this
    // is written for clarity, not for the last nanosecond. See
    // benchmarks/embedding_benchmark.cpp for the comparison against the
    // single-accumulator original.
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;

    std::size_t i = 0;
    const std::size_t end = dimensions - (dimensions % 4);

    for (; i < end; i += 4) {
        const double a = source[i];
        const double b = source[i + 1];
        const double c = source[i + 2];
        const double d = source[i + 3];

        sum0 += a * a;
        sum1 += b * b;
        sum2 += c * c;
        sum3 += d * d;
    }

    double squared_norm = (sum0 + sum1) + (sum2 + sum3);

    for (; i < dimensions; ++i) {
        const double value = source[i];
        squared_norm += value * value;
    }

    const float scale =
        squared_norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(squared_norm)) : 1.0F;

    for (i = 0; i < dimensions; ++i) {
        output[i] = source[i] * scale;
    }
}

} // namespace skippy::detail
