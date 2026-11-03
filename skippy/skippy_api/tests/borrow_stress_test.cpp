//! Stress cases for borrows that outlive growth: every test here exists
//! because a span into a container that later reallocates is a dangling
//! pointer, and release builds hide it. The ASAN suite runs these with
//! heap-use-after-free detection; the mocked suite runs them for logic.

#include "stage_plan.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using skippy::LayerRange;
using skippy::OwnedProfile;
using skippy::PackageId;
using skippy::PlanId;
using skippy::PlanProfile;
using skippy::ProfileId;
using skippy::StagePlan;
using skippy::ValueIdentity;

OwnedProfile profile_with(LayerRange range, std::size_t imports, std::size_t exports,
                          std::string id) {
    std::vector<skippy::OwnedValue> import_values;
    import_values.reserve(imports);
    for (std::size_t index = 0; index < imports; ++index) {
        import_values.push_back(skippy::OwnedValue{
            .identity = ValueIdentity{id + ".in." + std::to_string(index)}, .binding = {}});
    }
    std::vector<skippy::OwnedValue> export_values;
    export_values.reserve(exports);
    for (std::size_t index = 0; index < exports; ++index) {
        export_values.push_back(skippy::OwnedValue{
            .identity = ValueIdentity{id + ".out." + std::to_string(index)}, .binding = {}});
    }
    return OwnedProfile{
        .id = ProfileId{std::move(id)},
        .graph_identity = {"graph"},
        .profile_identity = {"profile"},
        .slice_identity = {"slice"},
        .source_snapshot = {"source"},
        .graph_configuration = {"config"},
        .backend = {"metal"},
        .slice_range = range,
        .activation_imports = std::move(import_values),
        .activation_exports = std::move(export_values),
        .request_inputs = {},
        .state_effects = {},
    };
}

StagePlan assembled(std::size_t profile_count, std::size_t values_per_profile) {
    StagePlan plan;
    plan.package = PackageId{"package"};
    plan.id = PlanId{"plan"};
    plan.layer_count = 12;
    plan.range = LayerRange{0, 12};
    plan.execution_contract = "{}";
    for (std::size_t index = 0; index < profile_count; ++index) {
        plan.append(profile_with(LayerRange{0, 12}, values_per_profile, values_per_profile,
                                 "p" + std::to_string(index)));
    }
    return plan;
}

/// The exact reported shape: four profiles with three imports each, reading
/// the first profile's values only after every append has forced whatever
/// reallocation the growth causes.
TEST(BorrowStress, FourProfilesThreeImportsReadTheFirstLast) {
    StagePlan plan = assembled(4, 3);
    ASSERT_EQ(plan.profiles.size(), 4U);
    const PlanProfile& front = plan.profiles.front();
    const auto imports = plan.activation_imports_of(front);
    ASSERT_EQ(imports.size(), 3U);
    EXPECT_EQ(plan.view(imports.front().identity.slice), "p0.in.0");
    EXPECT_EQ(plan.view(imports.back().identity.slice), "p0.in.2");
    const auto back = plan.activation_imports_of(plan.profiles.back());
    ASSERT_EQ(back.size(), 3U);
    EXPECT_EQ(plan.view(back.front().identity.slice), "p3.in.0");
}

/// Growth that must exceed geometric capacity several times over.
TEST(BorrowStress, ManyProfilesForceRepeatedReallocation) {
    StagePlan plan = assembled(50, 20);
    ASSERT_EQ(plan.profiles.size(), 50U);
    std::size_t total = 0;
    for (const PlanProfile& profile : plan.profiles) {
        total += plan.activation_imports_of(profile).size();
        total += plan.activation_exports_of(profile).size();
    }
    EXPECT_EQ(total, 50U * 40U);
    EXPECT_EQ(plan.view(plan.activation_imports_of(plan.profiles.front()).front().identity.slice),
              "p0.in.0");
    EXPECT_EQ(plan.view(plan.activation_exports_of(plan.profiles[49]).back().identity.slice),
              "p49.out.19");
}

/// clone() replays append across every profile; an earlier profile read via
/// the clone after the replay is the second reported shape.
TEST(BorrowStress, CloneKeepsEarlyProfilesReadable) {
    StagePlan plan = assembled(8, 6);
    StagePlan copy = plan.clone();
    const auto early = copy.activation_imports_of(copy.profiles.front());
    ASSERT_EQ(early.size(), 6U);
    EXPECT_EQ(copy.view(early.front().identity.slice), "p0.in.0");
    EXPECT_EQ(copy.view(copy.activation_exports_of(copy.profiles.back()).back().identity.slice),
              "p7.out.5");
}

/// A chain of plans validated after all growth, reading handoffs end to end.
TEST(BorrowStress, ChainValidationAfterGrowth) {
    StagePlan first = assembled(4, 3);
    first.range = LayerRange{0, 6};
    StagePlan second = assembled(4, 3);
    second.range = LayerRange{6, 12};
    const std::array<StagePlan, 2> chain{std::move(first), std::move(second)};
    const auto failure = skippy::validate_chain(chain);
    // The profiles do not actually hand off frontiers here; the point is
    // that reading every value across a grown chain is memory-safe.
    (void)failure;
}

} // namespace
