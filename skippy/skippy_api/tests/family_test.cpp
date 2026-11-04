#include "family.h"

#include <gtest/gtest.h>

namespace {

using skippy::ModelFamily;

TEST(Family, VersionedNamesMapToTheirFamily) {
    EXPECT_EQ(skippy::family_of("qwen2"), ModelFamily::Qwen);
    EXPECT_EQ(skippy::family_of("qwen2vl"), ModelFamily::Qwen);
    EXPECT_EQ(skippy::family_of("qwen3"), ModelFamily::Qwen);
    EXPECT_EQ(skippy::family_of("gemma3"), ModelFamily::Gemma);
    EXPECT_EQ(skippy::family_of("phi3"), ModelFamily::Phi);
}

TEST(Family, EncoderModelsReportBert) {
    EXPECT_EQ(skippy::family_of("bert"), ModelFamily::Bert);
    EXPECT_EQ(skippy::family_of("nomic-bert"), ModelFamily::Bert);
}

TEST(Family, UnknownArchitecturesAreNotAnError) {
    EXPECT_EQ(skippy::family_of(""), ModelFamily::Unknown);
    EXPECT_EQ(skippy::family_of("something-new"), ModelFamily::Unknown);
    EXPECT_EQ(skippy::describe(ModelFamily::Unknown), "unknown");
}

TEST(Family, RecurrentFamiliesAreNamed) {
    EXPECT_TRUE(skippy::is_recurrent(skippy::family_of("mamba")));
    EXPECT_TRUE(skippy::is_recurrent(skippy::family_of("rwkv")));
    EXPECT_FALSE(skippy::is_recurrent(skippy::family_of("qwen3")));
    EXPECT_FALSE(skippy::is_recurrent(skippy::family_of("bert")));
}

TEST(Family, DescribeCoversEveryValue) {
    for (const ModelFamily family :
         {ModelFamily::Unknown, ModelFamily::Llama, ModelFamily::Qwen, ModelFamily::Mistral,
          ModelFamily::Gemma, ModelFamily::Phi, ModelFamily::Bert, ModelFamily::Mamba,
          ModelFamily::Rwkv, ModelFamily::T5}) {
        EXPECT_FALSE(skippy::describe(family).empty());
    }
}

} // namespace
