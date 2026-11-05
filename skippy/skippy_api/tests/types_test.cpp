#include "types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using skippy::Token;
using skippy::detail::token_view_is_zero_copy;
using skippy::detail::tokens_from_native;

using ::testing::ElementsAre;

TEST(TokensFromNative, ValuesSurvive) {
    std::vector<std::int32_t> native{11, 22, 33};
    std::vector<Token> scratch;

    const auto tokens = tokens_from_native(native, scratch);

    EXPECT_THAT(tokens, ElementsAre(Token{11}, Token{22}, Token{33}));
}

TEST(TokensFromNative, EmptyInputIsEmptyView) {
    std::vector<std::int32_t> native;
    std::vector<Token> scratch{Token{99}};

    EXPECT_TRUE(tokens_from_native(native, scratch).empty());
}

TEST(TokensFromNative, ScratchIsReusedNotReallocated) {
    std::vector<std::int32_t> native{1, 2, 3, 4};
    std::vector<Token> scratch;
    scratch.reserve(8);
    const auto* before = scratch.data();

    tokens_from_native(native, scratch);
    tokens_from_native(native, scratch);

    if constexpr (!token_view_is_zero_copy) {
        EXPECT_EQ(scratch.data(), before);
    }
}

TEST(TokensFromNative, ZeroCopyPathStatus) {
    if constexpr (token_view_is_zero_copy) {
        std::vector<std::int32_t> native{7};
        std::vector<Token> scratch;
        const auto tokens = tokens_from_native(native, scratch);
        EXPECT_EQ(static_cast<const void*>(tokens.data()), static_cast<const void*>(native.data()));
    } else {
        GTEST_SKIP() << "std::start_lifetime_as_array unavailable on this standard library; "
                        "tokens are copied once per tokenize(). Remove the fallback when this "
                        "stops skipping.";
    }
}

} // namespace
