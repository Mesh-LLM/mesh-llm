#include "selector.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <type_traits>

namespace {

using skippy::Pooling;
using skippy::SequenceId;
using skippy::TokenIndex;
using skippy::detail::EmbeddingSelector;
using skippy::detail::NativeEmbeddingResult;
using skippy::detail::NativeEmbeddingView;

using ::testing::MockFunction;
using ::testing::Return;
using ::testing::StrictMock;

TEST(EmbeddingSelection, NoPoolingLooksUpSuppliedTokenIndex) {
    const std::array<float, 2> values = {0.6F, -0.8F};
    StrictMock<MockFunction<NativeEmbeddingResult(TokenIndex)>> lookup;

    EXPECT_CALL(lookup, Call(TokenIndex{2}))
        .WillOnce(Return(NativeEmbeddingResult{NativeEmbeddingView{values}}));

    const auto result =
        EmbeddingSelector<Pooling::None>::select(TokenIndex{2}, lookup.AsStdFunction());

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->data(), values.data());
    EXPECT_EQ(result->size(), values.size());
}

class PooledEmbeddingSelection : public ::testing::TestWithParam<Pooling> {};

TEST_P(PooledEmbeddingSelection, LooksUpSuppliedSequenceId) {
    const std::array<float, 2> values = {0.6F, -0.8F};
    StrictMock<MockFunction<NativeEmbeddingResult(SequenceId)>> lookup;

    EXPECT_CALL(lookup, Call(SequenceId{7}))
        .WillOnce(Return(NativeEmbeddingResult{NativeEmbeddingView{values}}));

    const NativeEmbeddingResult result = skippy::with_pooling(GetParam(), [&](auto tag) {
        constexpr Pooling mode = decltype(tag)::value;
        if constexpr (mode == Pooling::None) {
            ADD_FAILURE() << "pooled test instantiated with Pooling::None";
            return NativeEmbeddingResult{};
        } else {
            return EmbeddingSelector<mode>::select(SequenceId{7}, lookup.AsStdFunction());
        }
    });

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->data(), values.data());
    EXPECT_EQ(result->size(), values.size());
}

INSTANTIATE_TEST_SUITE_P(SupportedModes, PooledEmbeddingSelection,
                         ::testing::Values(Pooling::Mean, Pooling::Cls, Pooling::Last));

TEST(EmbeddingSelection, MissingTokenOutputReturnsNullopt) {
    StrictMock<MockFunction<NativeEmbeddingResult(TokenIndex)>> lookup;

    EXPECT_CALL(lookup, Call(TokenIndex{2})).WillOnce(Return(std::nullopt));

    const auto result =
        EmbeddingSelector<Pooling::None>::select(TokenIndex{2}, lookup.AsStdFunction());

    EXPECT_FALSE(result.has_value());
}

TEST(EmbeddingSelection, MissingSequenceOutputReturnsNullopt) {
    StrictMock<MockFunction<NativeEmbeddingResult(SequenceId)>> lookup;

    EXPECT_CALL(lookup, Call(SequenceId{7})).WillOnce(Return(std::nullopt));

    const auto result =
        EmbeddingSelector<Pooling::Mean>::select(SequenceId{7}, lookup.AsStdFunction());

    EXPECT_FALSE(result.has_value());
}

TEST(EmbeddingSelection, EmptyTokenOutputRemainsPresent) {
    StrictMock<MockFunction<NativeEmbeddingResult(TokenIndex)>> lookup;

    EXPECT_CALL(lookup, Call(TokenIndex{2}))
        .WillOnce(Return(NativeEmbeddingResult{NativeEmbeddingView{}}));

    const auto result =
        EmbeddingSelector<Pooling::None>::select(TokenIndex{2}, lookup.AsStdFunction());

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST(EmbeddingSelection, EmptySequenceOutputRemainsPresent) {
    StrictMock<MockFunction<NativeEmbeddingResult(SequenceId)>> lookup;

    EXPECT_CALL(lookup, Call(SequenceId{7}))
        .WillOnce(Return(NativeEmbeddingResult{NativeEmbeddingView{}}));

    const auto result =
        EmbeddingSelector<Pooling::Mean>::select(SequenceId{7}, lookup.AsStdFunction());

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

static_assert(!std::is_convertible_v<TokenIndex, SequenceId>);
static_assert(!std::is_convertible_v<SequenceId, TokenIndex>);
static_assert(!std::is_convertible_v<int, SequenceId>);

template <typename Selector, typename Id, typename Lookup>
concept SelectsWith = requires(Id id, Lookup lookup) { Selector::select(id, lookup); };

using SequenceLookup = NativeEmbeddingResult (*)(SequenceId);
using TokenLookup = NativeEmbeddingResult (*)(TokenIndex);

static_assert(SelectsWith<EmbeddingSelector<Pooling::Mean>, SequenceId, SequenceLookup>);
static_assert(!SelectsWith<EmbeddingSelector<Pooling::Mean>, TokenIndex, TokenLookup>);
static_assert(SelectsWith<EmbeddingSelector<Pooling::None>, TokenIndex, TokenLookup>);
static_assert(!SelectsWith<EmbeddingSelector<Pooling::None>, SequenceId, SequenceLookup>);

} // namespace
