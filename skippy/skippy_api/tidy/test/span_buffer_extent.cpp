// CHECKS: skippy-span-buffer-extent
#include <cstddef>
#include <span>

float scores[256];

// The honest forms: the buffer's own extent, and provable constant prefixes.
std::span<float> honest_extent() { return std::span<float>(scores); }
std::span<float> whole_by_literal() { return std::span<float>(scores, 256); }
std::span<float> prefix_by_literal() { return std::span<float>(scores, 16); }

// PR #2907 finding #8: a runtime count over a fixed buffer claims whatever
// the count says and steers a write past the buffer while every downstream
// size check passes.
std::span<float> runtime_count(std::size_t count) {
    return std::span<float>(scores, count);
    // CHECK: [[@LINE-1]]:{{[0-9]+}}: warning: std::span over fixed buffer 'scores' with a runtime count
}

std::span<float> brace_runtime_count(std::size_t count) {
    return std::span<float>{scores, count};
    // CHECK: [[@LINE-1]]:{{[0-9]+}}: warning: std::span over fixed buffer 'scores' with a runtime count
}

// A runtime count over a pointer is not the bug shape: there is no
// compile-time extent for the count to lie about.
std::span<float> from_pointer(float* data, std::size_t count) {
    return std::span<float>(data, count);
}
