#include "selected_log_probs.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

namespace {

constexpr auto masked = -std::numeric_limits<float>::infinity();

skippy::Token token(const std::int32_t id) { return skippy::Token{id}; }

} // namespace

TEST(SelectedLogProbs, OrdersScoresAndNormalises) {
    const std::array<float, 4> logits = {1000.0F, 999.0F, 998.0F, masked};
    const std::array<skippy::Token, 3> selected = {token(2), token(0), token(1)};
    std::array<float, 3> result = {7, 7, 7};

    ASSERT_TRUE(skippy::detail::selected_log_probabilities(logits, selected, result));

    EXPECT_GT(result[1], result[2]);
    EXPECT_GT(result[2], result[0]);
    EXPECT_NEAR(std::exp(result[0]) + std::exp(result[1]) + std::exp(result[2]), 1.0, 1e-6);
}

TEST(SelectedLogProbs, RejectsTokenOutsideVocabularyWithoutPublishing) {
    const std::array<float, 4> logits = {1000.0F, 999.0F, 998.0F, masked};
    const std::array<skippy::Token, 2> invalid = {token(0), token(4)};
    std::array<float, 2> result = {7, 7};

    EXPECT_FALSE(skippy::detail::selected_log_probabilities(logits, invalid, result));
    EXPECT_FLOAT_EQ(result[0], 7);
    EXPECT_FLOAT_EQ(result[1], 7);
}

TEST(SelectedLogProbs, RejectsEntirelyMaskedRow) {
    const std::array<float, 2> logits = {masked, masked};
    const std::array<skippy::Token, 2> selected = {token(0), token(1)};
    std::array<float, 1> result = {7};

    EXPECT_FALSE(skippy::detail::selected_log_probabilities(logits, selected, result));
    EXPECT_FLOAT_EQ(result[0], 7);
}

TEST(SelectedLogProbs, RejectsNaNRowEntry) {
    const std::array<float, 2> logits = {0.0F, std::numeric_limits<float>::quiet_NaN()};
    const std::array<skippy::Token, 2> selected = {token(0), token(1)};
    std::array<float, 1> result = {7};

    EXPECT_FALSE(skippy::detail::selected_log_probabilities(logits, selected, result));
    EXPECT_FLOAT_EQ(result[0], 7);
}

TEST(SelectedLogProbs, RejectsInfiniteRowEntry) {
    const std::array<float, 2> logits = {0.0F, std::numeric_limits<float>::infinity()};
    const std::array<skippy::Token, 2> selected = {token(0), token(1)};
    std::array<float, 1> result = {7};

    EXPECT_FALSE(skippy::detail::selected_log_probabilities(logits, selected, result));
    EXPECT_FLOAT_EQ(result[0], 7);
}

TEST(SelectedLogProbs, RejectsSelectedMaskedTokenWithoutPublishing) {
    const std::array<float, 4> logits = {1000.0F, 999.0F, 998.0F, masked};
    const std::array<skippy::Token, 2> selected = {token(0), token(3)};
    std::array<float, 2> result = {7, 7};

    EXPECT_FALSE(skippy::detail::selected_log_probabilities(logits, selected, result));
    EXPECT_FLOAT_EQ(result[0], 7);
    EXPECT_FLOAT_EQ(result[1], 7);
}

TEST(SelectedLogProbs, RejectsEmptyAndUndersizedOutput) {
    const std::array<float, 2> logits = {1.0F, 0.0F};
    const std::array<skippy::Token, 2> selected = {token(0), token(1)};
    std::array<float, 1> result = {7};

    EXPECT_FALSE(skippy::detail::selected_log_probabilities(logits, {}, result));
    EXPECT_FALSE(
        skippy::detail::selected_log_probabilities(std::span<const float>{}, selected, result));
    EXPECT_FALSE(skippy::detail::selected_log_probabilities(logits, selected, {}));
}

TEST(SelectedLogProbs, RejectsSelectionBeyondTheBoundedMaximum) {
    const std::array<float, 2> logits = {1.0F, 0.0F};
    const std::array<skippy::Token, skippy::detail::max_selected_tokens + 1> selected{};
    std::array<float, skippy::detail::max_selected_tokens + 1> result{};

    EXPECT_FALSE(skippy::detail::selected_log_probabilities(logits, selected, result));
    EXPECT_FLOAT_EQ(result[0], 0);
}
