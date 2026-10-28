// Fuzzes slice resolution over a plan's arena and flat value storage with
// fuzzer-chosen lengths, counts and ArenaSlice/ValueRange positions, touching
// every byte and element the accessors hand back. Corrupt positions must read
// short or empty, never out of bounds.
#include "stage_plan.h"

#include <cassert>
#include <cstdint>
#include <span>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 16U) return 0;
    const auto word = [&data](std::size_t at) {
        return static_cast<std::uint32_t>(data[at]) |
               (static_cast<std::uint32_t>(data[at + 1]) << 8U) |
               (static_cast<std::uint32_t>(data[at + 2]) << 16U) |
               (static_cast<std::uint32_t>(data[at + 3]) << 24U);
    };
    const auto arena_length = word(0U) % 257U;
    const auto value_count = word(4U) % 65U;
    const skippy::ArenaSlice slice{.offset = word(8U), .length = word(12U)};

    skippy::StagePlan plan;
    plan.identity_arena.assign(arena_length, '#');
    const skippy::FrontierValue value{};
    plan.activation_imports.assign(value_count, value);
    plan.activation_exports.assign(value_count, value);
    skippy::PlanProfile profile{};
    profile.id = skippy::ProfileId{"p0"};
    profile.slice_range = skippy::LayerRange{.start = 0, .end = 1};
    profile.activation_imports = skippy::ValueRange{.offset = 0U, .count = value_count};
    profile.activation_exports = skippy::ValueRange{.offset = 0U, .count = value_count};
    plan.profiles.push_back(profile);

    volatile std::uint32_t sink = 0;
    for (const char byte : plan.view(slice))
        sink += static_cast<std::uint32_t>(byte);
    for (const auto& imported : plan.activation_imports_of(profile)) {
        sink += imported.identity.slice.length + imported.binding.slice.offset;
    }
    for (const auto& exported : plan.activation_exports_of(profile)) {
        sink += exported.identity.slice.length + exported.binding.slice.offset;
    }
    assert(sink != 0xDEADBEEFU); // reads the touched bytes' sum
    return 0;
}
