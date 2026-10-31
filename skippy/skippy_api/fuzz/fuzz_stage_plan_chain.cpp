// Fuzzes plan assembly and the clone contract: N fuzzer-chosen profiles are
// appended, the plan is cloned, the original destroyed, and the clone is then
// validated, summarized and turned into a loader config with every returned
// byte and span touched. Stored positions must survive the clone, not the
// pointers.
#include "stage_load.h"
#include "stage_plan.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 5U) return 0;
    const auto profile_count = static_cast<std::size_t>(data[0] % 8U) + 1U;
    std::size_t at = 1U;

    skippy::StagePlan plan;
    plan.layer_count = 1;
    plan.range = skippy::LayerRange{.start = 0, .end = 1};
    plan.package = skippy::PackageId{"pkg"};
    plan.id = skippy::PlanId{"plan"};
    for (std::size_t index = 0; index < profile_count && at + 8U <= size; ++index) {
        const auto imports = static_cast<std::size_t>(data[at] % 16U);
        const auto exports = static_cast<std::size_t>(data[at + 1U] % 16U);
        at += 2U;
        skippy::OwnedProfile profile;
        profile.id = skippy::ProfileId{"p" + std::to_string(index)};
        profile.slice_range = skippy::LayerRange{.start = 0, .end = 1};
        for (std::size_t value = 0; value < imports; ++value) {
            profile.activation_imports.push_back(skippy::OwnedValue{
                .identity = skippy::ValueIdentity{"in" + std::to_string(value)},
                .binding = skippy::ValueBinding{"b" + std::to_string(value)},
            });
        }
        for (std::size_t value = 0; value < exports; ++value) {
            profile.activation_exports.push_back(skippy::OwnedValue{
                .identity = skippy::ValueIdentity{"out" + std::to_string(value)},
                .binding = skippy::ValueBinding{"b" + std::to_string(value)},
            });
        }
        if (at < size) {
            profile.tokens = static_cast<std::uint32_t>(data[at]);
            ++at;
        }
        plan.append(std::move(profile));
    }

    skippy::StagePlan clone = plan.clone();
    {
        const skippy::StagePlan destroyed = std::move(plan);
    }

    const skippy::StagePlan stages[] = {std::move(clone)};
    [[maybe_unused]] const auto chain_ok = skippy::validate_chain(stages);
    volatile std::uint32_t sink = 0;
    for (const auto& profile : stages[0].profiles) {
        for (const auto& imported : stages[0].activation_imports_of(profile)) {
            for (const char byte : stages[0].view(imported.identity.slice)) {
                sink += static_cast<std::uint32_t>(byte);
            }
            for (const char byte : stages[0].view(imported.binding.slice)) {
                sink += static_cast<std::uint32_t>(byte);
            }
        }
        [[maybe_unused]] const auto semantics = skippy::summarize(stages[0], profile.id);
        [[maybe_unused]] const auto config = skippy::stage_load_config(stages[0], profile.id);
    }
    assert(sink != 0xDEADBEEFU); // reads the touched bytes' sum
    return 0;
}
