#pragma once

#include <cstddef>

namespace skippy::detail {

// L2-normalises `dimensions` floats from `source` into `output`. The two may
// alias exactly (in-place) or not at all; partial overlap is not supported.
// A zero vector is copied unchanged. No dependency on any native library.
void normalize_embedding(const float* source, float* output, std::size_t dimensions) noexcept;

} // namespace skippy::detail
