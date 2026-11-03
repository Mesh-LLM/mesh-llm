#include "pooling.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <type_traits>

namespace {

using skippy::Pooling;
using skippy::PoolingTag;
using skippy::with_pooling;

using ::testing::MockFunction;
using ::testing::Return;
using ::testing::StrictMock;

class WithPooling : public ::testing::TestWithParam<Pooling> {};

TEST_P(WithPooling, DispatchesToTheTagForThatMode) {
    const Pooling seen = with_pooling(GetParam(), [](auto tag) { return decltype(tag)::value; });

    EXPECT_EQ(seen, GetParam());
}

TEST_P(WithPooling, VisitorSeesTheModeAsAConstantExpression) {
    const int ok = with_pooling(GetParam(), [](auto tag) {
        constexpr Pooling mode = decltype(tag)::value;
        static_assert(std::is_same_v<std::remove_cv_t<decltype(mode)>, Pooling>);
        return static_cast<int>(mode) >= 0 ? 1 : 0;
    });

    EXPECT_EQ(ok, 1);
}

TEST_P(WithPooling, InvokesTheVisitorExactlyOnce) {
    StrictMock<MockFunction<int(Pooling)>> visits;
    EXPECT_CALL(visits, Call(GetParam())).WillOnce(Return(42));

    const int result =
        with_pooling(GetParam(), [&](auto tag) { return visits.Call(decltype(tag)::value); });

    EXPECT_EQ(result, 42);
}

INSTANTIATE_TEST_SUITE_P(AllModes, WithPooling,
                         ::testing::Values(Pooling::None, Pooling::Mean, Pooling::Cls,
                                           Pooling::Last));

TEST(WithPoolingTypes, EachModeInstantiatesADistinctOverload) {
    static_assert(!std::is_same_v<PoolingTag<Pooling::None>, PoolingTag<Pooling::Mean>>);
    static_assert(!std::is_same_v<PoolingTag<Pooling::Cls>, PoolingTag<Pooling::Last>>);

    const bool is_token_level = with_pooling(Pooling::None, [](auto tag) {
        return static_cast<bool>(decltype(tag)::value == Pooling::None);
    });

    ASSERT_TRUE(is_token_level);
    EXPECT_FALSE(with_pooling(Pooling::Mean,
                              [](auto tag) { return decltype(tag)::value == Pooling::None; }));
}

} // namespace
