#include "recurrent_rollback.h"

#include "stage_load.h"

#include <gtest/gtest.h>

#include <limits>

namespace {

using skippy::RollbackTokens;
using skippy::StageLoadConfig;

TEST(RollbackTokens, PlannedDepthsClampIntoTheNativeWindow) {
    // The native clamp is planned -> min(max(planned, 2), 64); 0 keeps the
    // default of 2, so both ends and the interior must survive unchanged.
    EXPECT_EQ(RollbackTokens::from_planned(0).value(), 2U);
    EXPECT_EQ(RollbackTokens::from_planned(1).value(), 2U);
    EXPECT_EQ(RollbackTokens::from_planned(2).value(), 2U);
    EXPECT_EQ(RollbackTokens::from_planned(5).value(), 5U);
    EXPECT_EQ(RollbackTokens::from_planned(63).value(), 63U);
    EXPECT_EQ(RollbackTokens::from_planned(64).value(), 64U);
    EXPECT_EQ(RollbackTokens::from_planned(65).value(), 64U);
    EXPECT_EQ(RollbackTokens::from_planned(std::numeric_limits<std::uint32_t>::max()).value(), 64U);
}

TEST(RollbackTokens, DefaultIsTheNativeDefault) {
    EXPECT_EQ(RollbackTokens{}.value(), 2U);
    EXPECT_EQ(RollbackTokens{}, RollbackTokens::from_planned(0));
}

TEST(RollbackTokens, EqualDepthsCompareEqual) {
    EXPECT_EQ(RollbackTokens::from_planned(8), RollbackTokens::from_planned(8));
    EXPECT_NE(RollbackTokens::from_planned(4), RollbackTokens::from_planned(8));
    // Distinct plans outside the window clamp onto the same window edge.
    EXPECT_EQ(RollbackTokens::from_planned(100), RollbackTokens::from_planned(1000));
}

TEST(RollbackTokens, StageSpecDefaultsToTheNativeDefault) {
    // A caller that never sizes the plan keeps the native default of 2, the
    // same outcome as leaving the native field at 0.
    const StageLoadConfig::Spec spec{};
    EXPECT_EQ(spec.recurrent_rollback, RollbackTokens{});
    EXPECT_EQ(spec.recurrent_rollback.value(), 2U);
}

TEST(RollbackTokens, ClampTableHoldsAtCompileTime) {
    // The clamp is constexpr, so the mirror is pinned without running it;
    // a regression in the window edges fails the build, not a machine.
    static_assert(RollbackTokens::from_planned(0).value() == 2);
    static_assert(RollbackTokens::from_planned(1).value() == 2);
    static_assert(RollbackTokens::from_planned(2).value() == 2);
    static_assert(RollbackTokens::from_planned(64).value() == 64);
    static_assert(RollbackTokens::from_planned(65).value() == 64);
    static_assert(RollbackTokens::from_planned(std::numeric_limits<std::uint32_t>::max()).value() ==
                  64);
    static_assert(RollbackTokens{}.value() == RollbackTokens::from_planned(0).value());
    SUCCEED();
}

} // namespace
