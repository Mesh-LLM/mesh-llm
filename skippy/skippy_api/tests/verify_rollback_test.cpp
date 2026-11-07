#include "stage_load.h"
#include "verify_rollback.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using skippy::StageLoadConfig;
using skippy::VerifyRollback;

TEST(VerifyRollback, NoneAndInPlaceAreDistinctStates) {
    // "never verifies drafts" and "N tokens in place" are two policies, not
    // one integer: none() must not equal any in-place depth, and only the
    // native encoding folds it to 0.
    const auto two = VerifyRollback::in_place(2);
    ASSERT_TRUE(two.has_value());
    EXPECT_NE(VerifyRollback::none(), *two);
    EXPECT_EQ(VerifyRollback::none().tokens(), 0U);
    EXPECT_EQ(two->tokens(), 2U);
    EXPECT_TRUE(VerifyRollback::none().is_none());
    EXPECT_FALSE(two->is_none());
}

TEST(VerifyRollback, InPlaceRejectsZero) {
    // 0 is none()'s native encoding, so in_place(0) must be refused rather
    // than silently mean the opposite policy.
    const auto zero = VerifyRollback::in_place(0);
    ASSERT_FALSE(zero.has_value());
    EXPECT_NE(zero.error().find("none()"), std::string::npos);
}

TEST(VerifyRollback, NativeDefaultMatchesTheNativeLoader) {
    // The native default (SKIPPY_VERIFY_ROLLBACK_TOKENS_DEFAULT) is two
    // planned snapshots for a drafting session, and it is an explicit
    // in-place depth, not a special value.
    const auto two = VerifyRollback::in_place(2);
    ASSERT_TRUE(two.has_value());
    EXPECT_EQ(VerifyRollback::native_default().tokens(), 2U);
    EXPECT_EQ(VerifyRollback::native_default(), *two);
}

TEST(VerifyRollback, StageSpecHasNoDefaultPolicy) {
    // The spec must not default: a session cannot pick up snapshots - or a
    // native default - by omission; the caller states the policy.
    const StageLoadConfig::Spec spec{};
    EXPECT_FALSE(spec.verify_rollback.has_value());
}

TEST(VerifyRollback, OmittedPolicyIsRejectedWithAReason) {
    // The resolver open() consults refuses an omitted policy with the rule
    // it broke, and maps both stated policies to their native encodings:
    // none to 0 (no snapshots), an in-place depth to itself.
    const auto refused = skippy::verify_rollback_tokens_of(std::nullopt);
    ASSERT_FALSE(refused.has_value());
    EXPECT_NE(refused.error().find("verify_rollback"), std::string::npos);

    const auto none = skippy::verify_rollback_tokens_of(VerifyRollback::none());
    ASSERT_TRUE(none.has_value());
    EXPECT_EQ(*none, 0U);

    const auto five = VerifyRollback::in_place(5);
    ASSERT_TRUE(five.has_value());
    const auto in_place = skippy::verify_rollback_tokens_of(*five);
    ASSERT_TRUE(in_place.has_value());
    EXPECT_EQ(*in_place, 5U);
}

TEST(VerifyRollback, DistinctnessHoldsAtCompileTime) {
    // The distinct-states contract is constexpr, so the mirror is pinned
    // without running it.
    static_assert(VerifyRollback::none().tokens() == 0);
    static_assert(VerifyRollback::native_default().tokens() == 2);
    static_assert(VerifyRollback::none() != VerifyRollback::native_default());
    SUCCEED();
}

} // namespace
