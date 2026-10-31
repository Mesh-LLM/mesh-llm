//! Vocab suppress-token policy, ported from llama.cpp patch 0012. These
//! tests pin the patch's decision table: suppression overrides caller logit
//! biases, greedy selection skips suppressed tokens, exhaustion and invalid
//! ids are handled, and models without the GGUF key change nothing.

#include "sampling_suppress.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using skippy::LogitBias;
using skippy::Token;

const Token suppress_2_3[] = {Token{2}, Token{3}};

bool is_neg_inf(float value) { return std::isinf(value) && std::signbit(value); }

TEST(SamplingSuppress, VocabOverridesCallerLogitBiases) {
    // Mirror of the native test: a caller bias on a suppressed token is
    // replaced by -inf, other biases survive ahead of the appended set.
    const std::vector<LogitBias> caller{
        LogitBias{.token = Token{2}, .bias = 100.0F},
        LogitBias{.token = Token{4}, .bias = -1.0F},
        LogitBias{.token = Token{2}, .bias = 50.0F},
    };
    const auto merged = skippy::apply_vocab_suppress(caller, suppress_2_3);
    ASSERT_EQ(merged.size(), 3U);
    EXPECT_EQ(merged[0].token, Token{4});
    EXPECT_FLOAT_EQ(merged[0].bias, -1.0F);
    EXPECT_EQ(merged[1].token, Token{2});
    EXPECT_TRUE(is_neg_inf(merged[1].bias));
    EXPECT_EQ(merged[2].token, Token{3});
    EXPECT_TRUE(is_neg_inf(merged[2].bias));
}

TEST(SamplingSuppress, GreedySkipsTheSuppressedBestToken) {
    // Token 2 has the highest logit but is suppressed: the pick falls to
    // the best allowed token, exactly like the patch's fast-path fallback.
    const std::vector<float> logits{1.0F, 2.0F, 100.0F, 50.0F, 4.0F};
    EXPECT_EQ(skippy::greedy_among_allowed(logits, suppress_2_3), Token{4});
    const std::span<const Token> none;
    EXPECT_EQ(skippy::greedy_among_allowed(logits, none), Token{2});
}

TEST(SamplingSuppress, ExhaustedVocabularyYieldsNothing) {
    const std::vector<float> logits{1.0F, 2.0F, 100.0F, 50.0F, 4.0F};
    const std::vector<Token> all{Token{0}, Token{1}, Token{2}, Token{3}, Token{4}};
    EXPECT_EQ(skippy::greedy_among_allowed(logits, all), std::nullopt);
    EXPECT_EQ(skippy::greedy_among_allowed({}, suppress_2_3), std::nullopt);
}

TEST(SamplingSuppress, OutOfRangeSuppressIdsAreIgnored) {
    // A corrupt or future vocab cannot fail the selection; an id past the
    // vocabulary simply suppresses nothing.
    const std::vector<float> logits{1.0F, 2.0F, 100.0F, 50.0F, 4.0F};
    const std::vector<Token> invalid{Token{99}, Token{-4}};
    EXPECT_EQ(skippy::greedy_among_allowed(logits, invalid), Token{2});
}

TEST(SamplingSuppress, FirstAllowedTokenWinsTies) {
    const std::vector<float> logits{7.0F, 7.0F, 7.0F};
    const Token skip_first[] = {Token{0}};
    EXPECT_EQ(skippy::greedy_among_allowed(logits, skip_first), Token{1});
}

TEST(SamplingSuppress, ModelsWithoutSuppressTokensKeepCallerBiases) {
    const std::vector<LogitBias> caller{LogitBias{.token = Token{2}, .bias = 100.0F}};
    const auto merged = skippy::apply_vocab_suppress(caller, {});
    ASSERT_EQ(merged.size(), 1U);
    EXPECT_FLOAT_EQ(merged[0].bias, 100.0F);
}

TEST(SamplingSuppress, MembershipCheckMatchesTheSets) {
    EXPECT_TRUE(skippy::is_suppressed(Token{2}, suppress_2_3));
    EXPECT_TRUE(skippy::is_suppressed(Token{3}, suppress_2_3));
    EXPECT_FALSE(skippy::is_suppressed(Token{4}, suppress_2_3));
    EXPECT_FALSE(skippy::is_suppressed(Token{2}, {}));
}

// Patch 0013: the selection's boundary rule. Exhaustion is a typed
// rejection, an engaged selection (id 0 included) is a token, and the
// conversion never invents a third outcome.

TEST(SamplingSuppress, ExhaustedSelectionIsARejection) {
    const std::vector<float> logits{1.0F, 2.0F, 100.0F, 50.0F, 4.0F};
    const std::vector<Token> all{Token{0}, Token{1}, Token{2}, Token{3}, Token{4}};
    const auto rejected = skippy::eligible_token(skippy::greedy_among_allowed(logits, all));
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error(), skippy::SamplingFailure::ExhaustedVocabulary);
}

TEST(SamplingSuppress, AnEligibleSelectionPassesThroughUnchanged) {
    const std::vector<float> logits{1.0F, 2.0F, 100.0F, 50.0F, 4.0F};
    const auto eligible =
        skippy::eligible_token(skippy::greedy_among_allowed(logits, suppress_2_3));
    ASSERT_TRUE(eligible.has_value());
    EXPECT_EQ(*eligible, Token{4});
}

TEST(SamplingSuppress, TokenZeroIsEligibleNotASentinel) {
    // The polarity rule the native store path pins: zero is a token. An
    // engaged selection of id 0 passes the boundary check untouched.
    const std::optional<Token> zero{Token{0}};
    const auto eligible = skippy::eligible_token(zero);
    ASSERT_TRUE(eligible.has_value());
    EXPECT_EQ(*eligible, Token{0});
}

TEST(SamplingSuppress, NoLogitsIsExhaustionAtTheBoundary) {
    const auto rejected = skippy::eligible_token(skippy::greedy_among_allowed({}, {}));
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error(), skippy::SamplingFailure::ExhaustedVocabulary);
}

} // namespace
