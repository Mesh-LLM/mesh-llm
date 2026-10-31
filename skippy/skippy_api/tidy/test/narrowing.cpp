// CHECKS: skippy-explicit-narrowing
#include <cstddef>
#include <cstdint>
#include <string>

std::uint32_t arena_offset(const std::string& arena) {
    return static_cast<std::uint32_t>(arena.size());
    // CHECK: [[@LINE-1]]:12: warning: cast to 'std::uint32_t' (aka 'unsigned int') from '{{.*}}' (aka 'unsigned long') can lose value (64 → 32 bits); use narrow<std::uint32_t>()
}

std::int32_t layers(std::int64_t count) {
    return static_cast<std::int32_t>(count); // bounded by layer_count_from: < INT32_MAX
}

std::uint64_t widen(std::uint32_t value) { return static_cast<std::uint64_t>(value); }

std::int32_t literal() { return static_cast<std::int32_t>(512UL); }

std::uint32_t sign_loss(std::int32_t value) {
    return static_cast<std::uint32_t>(value);
    // CHECK: [[@LINE-1]]:12: warning: cast to 'std::uint32_t' (aka 'unsigned int') from 'std::int32_t' (aka 'int') can lose value (32 → 32 bits, signedness changes)
}
