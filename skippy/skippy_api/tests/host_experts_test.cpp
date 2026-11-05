#include "host_experts.h"

#include <gtest/gtest.h>

namespace {

using skippy::describe;
using skippy::host_expert_placement;
using skippy::HostExpertPlacement;
using skippy::LayerRange;
using skippy::max_host_expert_block_span;
using skippy::PlacementError;

TEST(HostExperts, DefaultRangeMeansOff) {
    const auto placement = host_expert_placement(LayerRange{});
    ASSERT_TRUE(placement.has_value());
    EXPECT_FALSE(placement->has_value());
}

TEST(HostExperts, EmptyNonZeroRangeIsOffNotAnError) {
    const auto placement = host_expert_placement(LayerRange{7, 7});
    ASSERT_TRUE(placement.has_value());
    EXPECT_FALSE(placement->has_value());
}

TEST(HostExperts, SingleBlockEngages) {
    const auto placement = host_expert_placement(LayerRange{0, 1});
    ASSERT_TRUE(placement.has_value());
    ASSERT_TRUE(placement->has_value());
    EXPECT_EQ((*placement)->start, 0);
    EXPECT_EQ((*placement)->end, 1);
}

TEST(HostExperts, SpanCapIsInclusive) {
    const LayerRange capped{0, max_host_expert_block_span};
    const auto placement = host_expert_placement(capped);
    ASSERT_TRUE(placement.has_value());
    ASSERT_TRUE(placement->has_value());
    EXPECT_EQ(**placement, capped);
}

TEST(HostExperts, OverspanIsRejected) {
    const auto placement = host_expert_placement(LayerRange{0, max_host_expert_block_span + 1});
    ASSERT_FALSE(placement.has_value());
    EXPECT_EQ(placement.error(), PlacementError::overspan);
}

TEST(HostExperts, NegativeStartIsRejectedBeforeEmptiness) {
    // {-1, -1} is empty by LayerRange's rule and negative by the ABI's; the
    // caller bug wins, so the range is an error, not "off".
    const auto placement = host_expert_placement(LayerRange{-1, -1});
    ASSERT_FALSE(placement.has_value());
    EXPECT_EQ(placement.error(), PlacementError::negative_start);
}

TEST(HostExperts, InvertedRangeIsRejected) {
    const auto placement = host_expert_placement(LayerRange{8, 4});
    ASSERT_FALSE(placement.has_value());
    EXPECT_EQ(placement.error(), PlacementError::inverted_range);
}

TEST(HostExperts, DescribeNamesEachRule) {
    EXPECT_STRNE(describe(PlacementError::negative_start), "");
    EXPECT_STRNE(describe(PlacementError::inverted_range), "");
    EXPECT_STRNE(describe(PlacementError::overspan), "");
}

} // namespace
