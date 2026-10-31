#include "../tools/probe_arguments.h"

#include <gtest/gtest.h>

#include <limits>
#include <string_view>

TEST(ProbeArguments, AcceptsTheSupportedIntegerBoundaries) {
    EXPECT_EQ(skippy::probe::parse_probe_integer("0", "start", 0), 0);
    EXPECT_EQ(skippy::probe::parse_probe_integer("1", "stages", 1), 1);
    EXPECT_EQ(skippy::probe::parse_probe_integer("2147483647", "layers", 1),
              std::numeric_limits<std::int32_t>::max());
}

TEST(ProbeArguments, RejectsMalformedAndOutOfRangeArgumentsWithoutThrowing) {
    for (const std::string_view text :
         {"", "-1", "2147483648", "9223372036854775807", "4junk", " 4", "4 "}) {
        const auto parsed = skippy::probe::parse_probe_integer(text, "stages", 1);
        ASSERT_FALSE(parsed.has_value()) << text;
        EXPECT_TRUE(parsed.error().starts_with("stages must be"));
    }
    EXPECT_FALSE(skippy::probe::parse_probe_integer("0", "steps", 1).has_value());
}
