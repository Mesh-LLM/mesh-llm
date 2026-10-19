#include "stage_plan.h"

#include <gtest/gtest.h>
#include <limits>

#include <ranges>
#include <string>
#include <utility>
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
using skippy::ProfileSpec;
using skippy::StagePlan;
using skippy::StagePlannerInputs;
using skippy::StateAccess;
using skippy::StateEffect;
using skippy::StateKind;
using skippy::StateResidency;
using skippy::ValueIdentity;

std::vector<OwnedValue> to_values(const std::vector<ValueIdentity>& identities) {
    std::vector<OwnedValue> values;
    values.reserve(identities.size());
    for (const ValueIdentity& identity : identities) {
        values.push_back(OwnedValue{.identity = identity, .binding = {}});
    }
    return values;
}

OwnedProfile profile(LayerRange range, std::vector<ValueIdentity> imports,
                     std::vector<ValueIdentity> exports, std::string id = "decode") {
    return OwnedProfile{
        .id = ProfileId{std::move(id)},
        .graph_identity = {"graph"},
        .profile_identity = {"profile"},
        .slice_identity = {"slice"},
        .source_snapshot = {"source"},
        .graph_configuration = {"graph-config"},
        .backend = {"metal"},
        .slice_range = range,
        .activation_imports = to_values(imports),
        .activation_exports = to_values(exports),
    };
}

StagePlan plan(LayerRange range, std::int32_t layer_count, std::vector<OwnedProfile> profiles) {
    StagePlan assembled;
    assembled.package = PackageId{"package"};
    assembled.id = PlanId{"plan"};
    assembled.layer_count = layer_count;
    assembled.range = range;
    assembled.execution_contract = "{}";
    for (OwnedProfile& entry : profiles) {
        assembled.append(std::move(entry));
    }
    return assembled;
}

StagePlan single(std::int32_t layer_count = 12) {
    return plan({0, layer_count}, layer_count, {profile({0, layer_count}, {}, {})});
}

} // namespace

TEST(StagePlan, UnitRangeIsHalfOpen) {
    const LayerRange range{4, 8};
    EXPECT_EQ(range.count(), 4);
    EXPECT_TRUE(range.contains(4));
    EXPECT_TRUE(range.contains(7));
    EXPECT_FALSE(range.contains(8));
    EXPECT_FALSE(range.contains(3));
}

TEST(StagePlan, AWholeModelPlanIsValid) { EXPECT_FALSE(skippy::validate(single()).has_value()); }

TEST(StagePlan, RejectsAnEmptyOrInvertedRange) {
    EXPECT_EQ(*skippy::validate(plan({4, 4}, 12, {profile({4, 4}, {}, {})})),
              PlanError::InvalidRange);
    EXPECT_EQ(*skippy::validate(plan({8, 4}, 12, {profile({8, 4}, {}, {})})),
              PlanError::InvalidRange);
    EXPECT_EQ(*skippy::validate(plan({-1, 4}, 12, {profile({-1, 4}, {}, {})})),
              PlanError::InvalidRange);
    EXPECT_EQ(*skippy::validate(plan({0, 13}, 12, {profile({0, 13}, {}, {})})),
              PlanError::InvalidRange);
}

TEST(StagePlan, RejectsAPlanWithoutProfiles) {
    EXPECT_EQ(*skippy::validate(plan({0, 12}, 12, {})), PlanError::ProfileCountMismatch);
}

TEST(StagePlan, RejectsAnEmptyPackageIdentity) {
    StagePlan value = single();
    value.package = PackageId{};
    EXPECT_EQ(*skippy::validate(value), PlanError::PackageMismatch);
}

TEST(StagePlan, RejectsAProfileBuiltForAnotherRange) {
    StagePlan value = plan({0, 6}, 12, {profile({6, 12}, {}, {})});
    EXPECT_EQ(*skippy::validate(value), PlanError::ProfileIdentityMismatch);
}

TEST(StagePlanChain, RejectsAnEmptyChain) {
    EXPECT_EQ(*skippy::validate_chain({}), PlanError::EmptyChain);
}

TEST(StagePlanChain, AcceptsAContiguousPartition) {
    StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    const std::array<StagePlan, 2> chain{std::move(first), std::move(second)};
    EXPECT_FALSE(skippy::validate_chain(chain).has_value());
}

TEST(StagePlanChain, RejectsAGapBetweenStages) {
    StagePlan first = plan({0, 5}, 12, {profile({0, 5}, {}, {})});
    StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {}, {})});
    const std::array<StagePlan, 2> chain{std::move(first), std::move(second)};
    EXPECT_EQ(*skippy::validate_chain(chain), PlanError::IncompletePartition);
}

TEST(StagePlanChain, RejectsAChainThatDoesNotStartAtLayerZero) {
    StagePlan stage = plan({4, 12}, 12, {profile({4, 12}, {}, {})});
    EXPECT_EQ(*skippy::validate_chain(std::array<StagePlan, 1>{std::move(stage)}),
              PlanError::IncompletePartition);
}

TEST(StagePlanChain, RejectsAChainThatDoesNotReachTheLastLayer) {
    StagePlan stage = plan({0, 11}, 12, {profile({0, 11}, {}, {})});
    EXPECT_EQ(*skippy::validate_chain(std::array<StagePlan, 1>{std::move(stage)}),
              PlanError::IncompletePartition);
}

TEST(StagePlanChain, RejectsStagesFromDifferentPackages) {
    StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    second.package = PackageId{"other"};
    const std::array<StagePlan, 2> chain{std::move(first), std::move(second)};
    EXPECT_EQ(*skippy::validate_chain(chain), PlanError::PackageMismatch);
}

TEST(StagePlanChain, RejectsStagesThatDisagreeOnTheModel) {
    StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    StagePlan second = plan({6, 12}, 13, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    const std::array<StagePlan, 2> chain{std::move(first), std::move(second)};
    EXPECT_EQ(*skippy::validate_chain(chain), PlanError::LayerCountMismatch);
}

TEST(StagePlanChain, RejectsStagesWithDifferentProfileCounts) {
    StagePlan first = plan({0, 6}, 12,
                           {profile({0, 6}, {}, {ValueIdentity{"h.5"}}),
                            profile({0, 6}, {}, {ValueIdentity{"h.5"}}, "aux")});
    StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    const std::array<StagePlan, 2> chain{std::move(first), std::move(second)};
    EXPECT_EQ(*skippy::validate_chain(chain), PlanError::ProfileCountMismatch);
}

TEST(StagePlanChain, RejectsStagesWithDifferentGuards) {
    StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}}, {})});
    second.profiles.front().graph_identity = {"other-graph"};
    const std::array<StagePlan, 2> chain{std::move(first), std::move(second)};
    EXPECT_EQ(*skippy::validate_chain(chain), PlanError::ProfileIdentityMismatch);
}

TEST(StagePlanChain, RequiresExportsToBeExactlyTheNextImports) {
    StagePlan first = plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.5"}})});
    StagePlan second = plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.4"}}, {})});
    const std::array<StagePlan, 2> chain{std::move(first), std::move(second)};
    EXPECT_EQ(*skippy::validate_chain(chain), PlanError::ActivationFrontierMismatch);
}

TEST(StagePlanChain, FrontierOrderIsPartOfTheContract) {
    StagePlan first =
        plan({0, 6}, 12, {profile({0, 6}, {}, {ValueIdentity{"h.4"}, ValueIdentity{"h.5"}})});
    StagePlan second =
        plan({6, 12}, 12, {profile({6, 12}, {ValueIdentity{"h.5"}, ValueIdentity{"h.4"}}, {})});
    const std::array<StagePlan, 2> chain{std::move(first), std::move(second)};
    EXPECT_EQ(*skippy::validate_chain(chain), PlanError::ActivationFrontierMismatch);
}

TEST(StagePlanChain, AcceptsAThreeStageChain) {
    StagePlan first = plan({0, 4}, 12, {profile({0, 4}, {}, {ValueIdentity{"h.3"}})});
    StagePlan middle =
        plan({4, 8}, 12, {profile({4, 8}, {ValueIdentity{"h.3"}}, {ValueIdentity{"h.7"}})});
    StagePlan last = plan({8, 12}, 12, {profile({8, 12}, {ValueIdentity{"h.7"}}, {})});
    const std::array<StagePlan, 3> chain{std::move(first), std::move(middle), std::move(last)};
    EXPECT_FALSE(skippy::validate_chain(chain).has_value());
    EXPECT_TRUE(last.terminal());
}

namespace {

StateEffect effect(StateKind kind, StateAccess access, std::int32_t layer,
                   StateResidency residency = StateResidency::LayerLocal) {
    return StateEffect{
        .kind = kind, .access = access, .layer = layer, .residency = residency, .write_ordinal = 0};
}

StagePlan with_states(std::vector<StateEffect> effects, std::size_t imports = 0,
                      std::size_t exports = 0) {
    StagePlan value = plan({0, 6}, 6, {});
    OwnedProfile entry = profile({0, 6}, {}, {});
    entry.state_effects = std::move(effects);
    for (std::size_t index = 0; index < imports; ++index) {
        entry.activation_imports.push_back(OwnedValue{.identity = ValueIdentity{"in"}});
    }
    for (std::size_t index = 0; index < exports; ++index) {
        entry.activation_exports.push_back(OwnedValue{.identity = ValueIdentity{"out"}});
    }
    value.append(std::move(entry));
    return value;
}

} // namespace

TEST(StagePlanSemantics, SummarizesKvEffectsPerLayer) {
    StagePlan value = with_states({effect(StateKind::KvKey, StateAccess::Write, 0),
                                   effect(StateKind::KvValue, StateAccess::Write, 0),
                                   effect(StateKind::KvKey, StateAccess::Write, 5)});
    const auto semantics = skippy::summarize(value, ProfileId{"decode"});
    ASSERT_TRUE(semantics.has_value());
    EXPECT_EQ(semantics->state_kinds.size(), 2U);
    EXPECT_FALSE(semantics->recurrent);
    EXPECT_EQ(semantics->first_layer, 0);
    EXPECT_EQ(semantics->last_layer, 5);
    EXPECT_EQ(semantics->written, 3U);
}

TEST(StagePlanSemantics, RecurrenTStateIsRecognisedByKindNotByFamily) {
    StagePlan value = with_states({effect(StateKind::RecurrentSsm, StateAccess::Write, 2),
                                   effect(StateKind::KvKey, StateAccess::Read, 2)});
    const auto semantics = skippy::summarize(value, ProfileId{"decode"});
    ASSERT_TRUE(semantics.has_value());
    EXPECT_TRUE(semantics->recurrent);
    EXPECT_EQ(semantics->written, 1U);
}

TEST(StagePlanSemantics, CountsFrontierValuesBothWays) {
    StagePlan value = with_states({}, 2, 1);
    const auto semantics = skippy::summarize(value, ProfileId{"decode"});
    ASSERT_TRUE(semantics.has_value());
    EXPECT_EQ(semantics->frontier_values, 3U);
    EXPECT_EQ(semantics->first_layer, -1);
}

TEST(StagePlanSemantics, AnUnknownProfileIsRejected) {
    StagePlan value = with_states({});
    EXPECT_EQ(skippy::summarize(value, ProfileId{"absent"}).error(),
              PlanError::ProfileIdentityMismatch);
}

TEST(StagePlanSemantics, StateEffectsAreFilteredByLayerInPlanOrder) {
    OwnedProfile entry = profile({0, 6}, {}, {});
    entry.state_effects = {effect(StateKind::KvKey, StateAccess::Write, 1),
                           effect(StateKind::KvValue, StateAccess::Write, 1),
                           effect(StateKind::KvKey, StateAccess::Write, 2)};
    StagePlan value = plan({0, 6}, 6, {});
    value.append(std::move(entry));

    const std::vector<StateEffect> layer_one =
        skippy::state_effects_at(value, value.profiles.front(), 1);
    ASSERT_EQ(layer_one.size(), 2U);
    EXPECT_EQ(layer_one[0].kind, StateKind::KvKey);
    EXPECT_EQ(layer_one[1].kind, StateKind::KvValue);
    EXPECT_TRUE(skippy::state_effects_at(value, value.profiles.front(), 4).empty());
}

TEST(StagePlannerInputs, RejectsAPlannerWithoutShards) {
    StagePlannerInputs value{
        .package = PackageId{"package"},
        .shard_paths = {},
        .profiles = {ProfileSpec{.id = ProfileId{"decode"}, .tokens = 512}},
        .graph_configuration = {"graph"},
        .backend = {"metal"},
    };
    const auto valid = skippy::validate_inputs(value);
    ASSERT_TRUE(valid.has_value());
    EXPECT_EQ(valid->code, PlanError::Rejected);
    EXPECT_NE(valid->message.find("shard"), std::string::npos);
}

TEST(StagePlannerInputs, RejectsAPlannerWithoutProfiles) {
    StagePlannerInputs value{
        .package = PackageId{"package"},
        .shard_paths = {"shard.gguf"},
        .profiles = {},
        .graph_configuration = {"graph"},
        .backend = {"metal"},
    };
    const auto valid = skippy::validate_inputs(value);
    ASSERT_TRUE(valid.has_value());
    EXPECT_EQ(valid->code, PlanError::ProfileCountMismatch);
}

TEST(StagePlannerInputs, RejectsAnEmptyPackageIdentity) {
    StagePlannerInputs value{
        .package = PackageId{},
        .shard_paths = {"shard.gguf"},
        .profiles = {ProfileSpec{.id = ProfileId{"decode"}, .tokens = 512}},
        .graph_configuration = {"graph"},
        .backend = {"metal"},
    };
    EXPECT_EQ(skippy::validate_inputs(value)->code, PlanError::PackageMismatch);
}

TEST(StagePlannerInputs, AcceptsAPlausibleRequest) {
    StagePlannerInputs value{
        .package = PackageId{"package"},
        .shard_paths = {"shard.gguf"},
        .profiles = {ProfileSpec{.id = ProfileId{"decode"}, .tokens = 512}},
        .graph_configuration = {"graph"},
        .backend = {"metal"},
    };
    EXPECT_FALSE(skippy::validate_inputs(value).has_value());
}

TEST(StagePlan, TerminalIsAboutTheLastLayerNotTheLastPlan) {
    EXPECT_TRUE(plan({6, 12}, 12, {profile({6, 12}, {}, {})}).terminal());
    EXPECT_FALSE(plan({0, 6}, 12, {profile({0, 6}, {}, {})}).terminal());
}
namespace {

TEST(StagePlan, AppendingALaterProfileDoesNotDangleAnEarlierOne) {
    // The regression for the dangling-span bug: profile 0's values were
    // stored as spans, and profile 1's append reallocated the vector under
    // them. Enough values to force growth past any small-buffer capacity.
    StagePlan value = plan({0, 6}, 6, {});
    std::vector<ValueIdentity> many;
    many.reserve(64);
    for (const std::size_t index : std::views::iota(std::size_t{0}, std::size_t{64})) {
        many.push_back(ValueIdentity{"second." + std::to_string(index)});
    }
    std::vector<ValueIdentity> firsts;
    firsts.push_back(ValueIdentity{"first"});
    value.append(profile({0, 6}, firsts, {}));
    value.append(profile({0, 6}, {}, std::move(many))); // same slice, 64 exports

    const PlanProfile& early = value.profiles[0];
    const PlanProfile& late = value.profiles[1];
    ASSERT_EQ(value.activation_imports_of(early).size(), 1U);
    EXPECT_EQ(value.view(value.activation_imports_of(early).front().identity.slice), "first");
    ASSERT_EQ(value.request_inputs_of(late).size(), 0U);
    ASSERT_EQ(value.activation_exports_of(late).size(), 64U);
    EXPECT_EQ(value.view(value.activation_exports_of(late).back().identity.slice), "second.63");
    EXPECT_FALSE(skippy::validate(value).has_value());
}

TEST(StagePlan, CorruptValuePositionsReadShortInsteadOfThrowing) {
    // Regression for the noexcept bug: these accessors resolved a profile's
    // recorded position with subspan, which throws std::out_of_range when the
    // position is corrupt — out of a noexcept function, that is
    // std::terminate. The contract is view()'s: a corrupt position yields a
    // short or empty span.
    StagePlan value = plan({0, 6}, 6, {});
    std::vector<ValueIdentity> firsts;
    firsts.push_back(ValueIdentity{"first"});
    value.append(profile({0, 6}, firsts, {}));
    PlanProfile& corrupt = value.profiles[0];

    // A count past the end of a one-element store clamps to that element.
    corrupt.activation_imports.offset = 0;
    corrupt.activation_imports.count = std::numeric_limits<std::uint32_t>::max();
    ASSERT_EQ(value.activation_imports_of(corrupt).size(), 1U);
    EXPECT_EQ(value.view(value.activation_imports_of(corrupt).front().identity.slice), "first");

    // An offset past the end of an occupied store clamps to empty.
    corrupt.activation_imports.offset = std::numeric_limits<std::uint32_t>::max();
    corrupt.activation_imports.count = 1;
    EXPECT_TRUE(value.activation_imports_of(corrupt).empty());

    // Empty stores clamp the same way instead of throwing.
    corrupt.activation_exports.offset = 2;
    corrupt.activation_exports.count = 3;
    corrupt.request_inputs.offset = std::numeric_limits<std::uint32_t>::max();
    corrupt.request_inputs.count = std::numeric_limits<std::uint32_t>::max();
    corrupt.state_effects.offset = 1;
    corrupt.state_effects.count = 4;
    EXPECT_TRUE(value.activation_exports_of(corrupt).empty());
    EXPECT_TRUE(value.request_inputs_of(corrupt).empty());
    EXPECT_TRUE(value.state_effects_of(corrupt).empty());
}

} // namespace
