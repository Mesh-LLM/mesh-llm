#include "stage_load.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using skippy::FrontierValue;
using skippy::LayerRange;
using skippy::OwnedProfile;
using skippy::OwnedValue;
using skippy::PackageId;
using skippy::PlanError;
using skippy::PlanId;
using skippy::PlanProfile;
using skippy::ProfileId;
using skippy::StagePlan;
using skippy::TensorId;
using skippy::TensorSlice;
using skippy::ValueBinding;
using skippy::ValueIdentity;

OwnedValue value(std::string identity, std::string binding) {
    return OwnedValue{.identity = ValueIdentity{std::move(identity)},
                      .binding = ValueBinding{std::move(binding)}};
}

StagePlan plan(std::string contract = "{\"version\":1}") {
    StagePlan assembled;
    assembled.package = PackageId{"sha256:package"};
    assembled.id = PlanId{"plan"};
    assembled.layer_count = 6;
    assembled.range = LayerRange{0, 6};
    assembled.execution_contract = std::move(contract);
    // Residents are slices into the plan's own arena: append the bytes, then
    // record the slices the loader config will resolve.
    const std::string residents[]{"t.0", "t.1"};
    assembled.resident_identities.reserve(2);
    for (const std::string& resident : residents) {
        const auto offset = static_cast<std::uint32_t>(assembled.identity_arena.size());
        assembled.identity_arena.append(resident);
        assembled.resident_identities.push_back(TensorSlice{
            .slice = {.offset = offset, .length = static_cast<std::uint32_t>(resident.size())}});
    }
    assembled.append(OwnedProfile{
        .id = ProfileId{"decode"},
        .graph_identity = {"graph"},
        .profile_identity = {"profile"},
        .slice_identity = {"slice"},
        .source_snapshot = {"source"},
        .graph_configuration = {"graph-config"},
        .backend = {"metal"},
        .slice_range = {0, 6},
        .activation_imports = {value("in.a", "bind.a")},
        .activation_exports = {value("out.b", "bind.b"), value("out.c", "bind.c")},
        .request_inputs = {},
        .state_effects = {},
    });
    return assembled;
}

} // namespace

TEST(StageLoadConfig, CarriesTheClosureAndContract) {
    const StagePlan value = plan();
    const auto config = skippy::stage_load_config(value, ProfileId{"decode"});
    ASSERT_TRUE(config.has_value());
    EXPECT_TRUE(config->staged());
    const auto& profile = value.profiles.front();
    EXPECT_EQ(config->frontier_values, value.activation_exports_of(profile).size() +
                                           value.activation_imports_of(profile).size());
    EXPECT_EQ(config->state_effects, value.state_effects_of(profile).size());
    EXPECT_FALSE(config->recurrent_state);
    EXPECT_EQ(config->layers, (LayerRange{0, 6}));
    EXPECT_EQ(config->execution_contract, "{\"version\":1}");
    ASSERT_EQ(config->resident_identities.size(), 2U);
    EXPECT_EQ(config->resident_identities[0], "t.0");
    EXPECT_EQ(config->resident_identities[1], "t.1");
}

TEST(StageLoadConfig, PairsFrontierIdentitiesWithTheirBindings) {
    const StagePlan value = plan();
    const auto config = skippy::stage_load_config(value, ProfileId{"decode"});
    ASSERT_TRUE(config.has_value());
    ASSERT_EQ(config->activation_import_identities.size(), 1U);
    EXPECT_EQ(config->activation_import_identities[0], "in.a");
    EXPECT_EQ(config->activation_import_bindings[0], "bind.a");
    ASSERT_EQ(config->activation_export_identities.size(), 2U);
    EXPECT_EQ(config->activation_export_identities[1], "out.c");
    EXPECT_EQ(config->activation_export_bindings[1], "bind.c");
}

TEST(StageLoadConfig, AnUnknownProfileIsRejected) {
    const StagePlan value = plan();
    EXPECT_EQ(skippy::stage_load_config(value, ProfileId{"absent"}).error(),
              PlanError::ProfileIdentityMismatch);
}

TEST(StageLoadConfig, AStageWithoutTensorsCannotBeLoaded) {
    StagePlan value = plan();
    value.resident_identities.clear();
    EXPECT_EQ(skippy::stage_load_config(value, ProfileId{"decode"}).error(),
              PlanError::ProfileCountMismatch);
}

TEST(StageLoadConfig, AStageWithoutAContractCannotBeLoaded) {
    StagePlan value = plan("");
    EXPECT_EQ(skippy::stage_load_config(value, ProfileId{"decode"}).error(),
              PlanError::ActivationFrontierMismatch);
}
