#include "stage_sink.h"

#include <gtest/gtest.h>

#include <array>
#include <string>

namespace {

using skippy::CopyFrameSink;
using skippy::DiscardSink;
using skippy::FrameBytes;
using skippy::StageSink;

// The concept is the plugin contract: a sink is anything that consumes a frame
// and reports failure as a value.
static_assert(StageSink<DiscardSink>);
static_assert(StageSink<CopyFrameSink>);

struct RejectingSink {
    [[nodiscard]] std::optional<std::string> consume(const FrameBytes&) const {
        return std::string{"refused"};
    }
};
static_assert(StageSink<RejectingSink>);

struct WrongShape {
    void consume(const FrameBytes&) const {}
};
static_assert(!StageSink<WrongShape>);

TEST(StageChain, DiscardSinkKeepsNothing) {
    DiscardSink sink;
    const std::array<std::byte, 2> bytes{std::byte{1}, std::byte{2}};
    const FrameBytes frame{bytes};
    EXPECT_FALSE(sink.consume(frame).has_value());
}

TEST(StageChain, CopySinkOwnsItsOwnFrame) {
    CopyFrameSink sink;
    const std::array<std::byte, 3> bytes{std::byte{7}, std::byte{8}, std::byte{9}};
    const FrameBytes source{bytes};
    ASSERT_FALSE(sink.consume(source).has_value());
    ASSERT_TRUE(sink.frame.has_value());
    ASSERT_EQ(sink.frame->size(), 3U);
    EXPECT_EQ(sink.frame->front(), std::byte{7});
}

TEST(StageChain, SinksMayRefuseAndSayWhy) {
    const RejectingSink sink;
    const FrameBytes frame{};
    const auto outcome = sink.consume(frame);
    ASSERT_TRUE(outcome.has_value());
    EXPECT_EQ(*outcome, "refused");
}

} // namespace
