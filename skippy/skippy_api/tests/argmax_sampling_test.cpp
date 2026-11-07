#include "argmax_sampling.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>

namespace {

using skippy::ArgmaxEligibility;
using skippy::ArgmaxPlan;
using skippy::LogitBias;
using skippy::SamplerKind;
using skippy::Token;

TEST(ArgmaxSampling, ZeroValuePolicyOnTheDefaultChainSelectsArgmax) {
    EXPECT_TRUE(skippy::selects_argmax(ArgmaxEligibility{}));
}

TEST(ArgmaxSampling, PositiveOrNaNTemperatureIsNotArgmax) {
    ArgmaxEligibility knobs{};
    knobs.temperature = 0.7f;
    EXPECT_FALSE(skippy::selects_argmax(knobs));
    knobs.temperature = std::numeric_limits<float>::quiet_NaN();
    // The native guard is !(temperature <= 0), so NaN fails it too.
    EXPECT_FALSE(skippy::selects_argmax(knobs));
}

TEST(ArgmaxSampling, DynatempAndMirostatDisqualify) {
    ArgmaxEligibility dynatemp{};
    dynatemp.dynatemp_range = 0.1f;
    EXPECT_FALSE(skippy::selects_argmax(dynatemp));
    ArgmaxEligibility mirostat{};
    mirostat.mirostat_mode = 2;
    EXPECT_FALSE(skippy::selects_argmax(mirostat));
}

TEST(ArgmaxSampling, CustomOrderMustIncludeTheTemperatureSampler) {
    const std::array<SamplerKind, 2> chain{SamplerKind::top_k, SamplerKind::min_p};
    EXPECT_FALSE(skippy::selects_argmax(ArgmaxEligibility{}, chain));
    const std::array<SamplerKind, 2> with_temperature{SamplerKind::top_k, SamplerKind::temperature};
    EXPECT_TRUE(skippy::selects_argmax(ArgmaxEligibility{}, with_temperature));
}

TEST(ArgmaxSampling, UnknownSamplerKindDisqualifies) {
    const auto rogue = static_cast<SamplerKind>(42);
    const std::array<SamplerKind, 2> chain{rogue, SamplerKind::temperature};
    EXPECT_FALSE(skippy::selects_argmax(ArgmaxEligibility{}, chain));
}

TEST(ArgmaxSampling, ActivePenaltiesBlockAndInertOnesPass) {
    ArgmaxEligibility active{};
    active.penalty_last_n = 64;
    active.repeat_penalty = 1.2f;
    EXPECT_FALSE(skippy::selects_argmax(active));

    // A zero window maps to -1, which disables the penalties sampler, so
    // non-zero penalty factors inside it stay eligible.
    ArgmaxEligibility inert_window{};
    inert_window.penalty_last_n = 0;
    inert_window.presence_penalty = 0.5f;
    inert_window.frequency_penalty = 0.5f;
    EXPECT_TRUE(skippy::selects_argmax(inert_window));

    // A neutral or zero repeat penalty with a live window changes nothing:
    // the chain maps zero to the neutral 1.0.
    ArgmaxEligibility neutral_repeat{};
    neutral_repeat.penalty_last_n = 64;
    EXPECT_TRUE(skippy::selects_argmax(neutral_repeat));

    ArgmaxEligibility active_repeat = neutral_repeat;
    active_repeat.repeat_penalty = 1.2f;
    EXPECT_FALSE(skippy::selects_argmax(active_repeat));
}

TEST(ArgmaxSampling, DryAndTypicalAndXtcFollowTheNativeRules) {
    ArgmaxEligibility dry_active{};
    dry_active.dry_multiplier = 0.8f;
    dry_active.dry_base = 1.5f;
    dry_active.dry_penalty_last_n = 2;
    EXPECT_FALSE(skippy::selects_argmax(dry_active));

    // Typical sampling can drop the top token: the default order skips it
    // outside (0, 1), and that exemption does not travel with a custom order.
    ArgmaxEligibility typical_default{};
    typical_default.typical_p = 0.9f;
    EXPECT_FALSE(skippy::selects_argmax(typical_default));
    ArgmaxEligibility typical_zero{};
    typical_zero.typical_p = 0.0f;
    EXPECT_TRUE(skippy::selects_argmax(typical_zero));
    const std::array<SamplerKind, 2> custom{SamplerKind::typical_p, SamplerKind::temperature};
    EXPECT_FALSE(skippy::selects_argmax(typical_zero, custom));
    ArgmaxEligibility typical_wide{};
    typical_wide.typical_p = 1.0f;
    EXPECT_TRUE(skippy::selects_argmax(typical_wide, custom));

    ArgmaxEligibility xtc_active{};
    xtc_active.xtc_probability = 0.3f;
    xtc_active.xtc_threshold = 0.2f;
    EXPECT_FALSE(skippy::selects_argmax(xtc_active));
}

TEST(ArgmaxSampling, DefaultOrderEndsWithTemperature) {
    static_assert(skippy::default_sampler_order.size() == 9);
    static_assert(skippy::default_sampler_order.back() == SamplerKind::temperature);
}

TEST(ArgmaxSampling, PlainArgmaxWithoutBiases) {
    const ArgmaxPlan plan{std::span<const LogitBias>{}, 3};
    const std::array<float, 3> logits{0.1f, 0.9f, 0.5f};
    const auto token = plan.select_unique(logits);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(*token, Token{1});
}

TEST(ArgmaxSampling, RepeatedBiasesApplyInOrderWithoutPreSumming) {
    const std::array<LogitBias, 4> biases{
        LogitBias{.token = Token{0}, .bias = -1e20f},
        LogitBias{.token = Token{1}, .bias = 0.0f},
        LogitBias{.token = Token{0}, .bias = 1.0f},
        LogitBias{.token = Token{-1}, .bias = 4.0f},
    };
    const ArgmaxPlan plan{biases, 2};
    const std::array<float, 2> logits{1e20f, 0.5f};
    const auto token = plan.select_unique(logits);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(*token, Token{0});
}

TEST(ArgmaxSampling, TiedOrNaNOrShortInputsYieldNothing) {
    const ArgmaxPlan plan{};
    const std::array<float, 3> tied{3.0f, 1.0f, 3.0f};
    EXPECT_FALSE(plan.select_unique(tied).has_value());
    const std::array<float, 2> with_nan{std::numeric_limits<float>::quiet_NaN(), 2.0f};
    EXPECT_FALSE(plan.select_unique(with_nan).has_value());
    const std::array<float, 1> short_span{1.0f};
    const std::array<LogitBias, 1> bias{LogitBias{.token = Token{0}, .bias = 0.5f}};
    const ArgmaxPlan bounded{bias, 2};
    EXPECT_FALSE(bounded.select_unique(short_span).has_value());
}

TEST(ArgmaxSampling, OutOfRangeBiasesAreSkippedAndInRangeOnesApply) {
    const std::array<LogitBias, 2> biases{
        LogitBias{.token = Token{9}, .bias = 100.0f},
        LogitBias{.token = Token{1}, .bias = 10.0f},
    };
    const ArgmaxPlan plan{biases, 2};
    const std::array<float, 2> logits{5.0f, 1.0f};
    const auto token = plan.select_unique(logits);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(*token, Token{1});
}

TEST(ArgmaxSampling, EmptyBiasTableBuildsAnEmptyPlan) {
    const ArgmaxPlan plan{std::span<const LogitBias>{}, 16};
    EXPECT_TRUE(plan.empty());
}

} // namespace
