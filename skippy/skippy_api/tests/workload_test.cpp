#include "workload.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>

#include <optional>
#include <string>

namespace {

using skippy::classify_workload;
using skippy::make_workload_info;
using skippy::NativePooling;
using skippy::Pooling;
using skippy::to_pooling;
using skippy::WorkloadKind;
using skippy::WorkloadProbe;

// --- to_pooling ---------------------------------------------------------------

TEST(ToPooling, MapsTheFourEmbeddingModes) {
    EXPECT_EQ(to_pooling(NativePooling::None), Pooling::None);
    EXPECT_EQ(to_pooling(NativePooling::Mean), Pooling::Mean);
    EXPECT_EQ(to_pooling(NativePooling::Cls), Pooling::Cls);
    EXPECT_EQ(to_pooling(NativePooling::Last), Pooling::Last);
}

TEST(ToPooling, RejectsModesTheEmbeddingPathCannotDrive) {
    EXPECT_EQ(to_pooling(NativePooling::Unspecified), std::nullopt);
    EXPECT_EQ(to_pooling(NativePooling::Rank), std::nullopt);
    EXPECT_EQ(to_pooling(static_cast<NativePooling>(42)), std::nullopt);
}

// Values are shared with the native enum by construction; pin them.
static_assert(static_cast<int>(NativePooling::Unspecified) == -1);
static_assert(static_cast<int>(NativePooling::None) == 0);
static_assert(static_cast<int>(NativePooling::Rank) == 4);
static_assert(static_cast<int>(Pooling::None) == static_cast<int>(NativePooling::None));
static_assert(static_cast<int>(Pooling::Last) == static_cast<int>(NativePooling::Last));

// --- classify_workload ----------------------------------------------------------

struct ClassificationCase {
    std::string name;
    WorkloadProbe probe;
    WorkloadKind expected;
};

class ClassifyWorkload : public ::testing::TestWithParam<ClassificationCase> {};

TEST_P(ClassifyWorkload, MatchesTheOriginalAbiRules) {
    EXPECT_EQ(classify_workload(GetParam().probe), GetParam().expected) << GetParam().name;
}

INSTANTIATE_TEST_SUITE_P(
    Rules, ClassifyWorkload,
    ::testing::Values(
        ClassificationCase{"plain decoder, no pooling",
                           {.pooling = NativePooling::None, .has_decoder = true},
                           WorkloadKind::CausalGeneration},
        ClassificationCase{"decoder with unspecified pooling",
                           {.pooling = NativePooling::Unspecified, .has_decoder = true},
                           WorkloadKind::CausalGeneration},
        ClassificationCase{"decoder with mean pooling",
                           {.pooling = NativePooling::Mean, .has_decoder = true},
                           WorkloadKind::Embedding},
        ClassificationCase{"decoder with last-token pooling",
                           {.pooling = NativePooling::Last, .has_decoder = true},
                           WorkloadKind::Embedding},
        ClassificationCase{"encoder-only, per-token outputs",
                           {.pooling = NativePooling::None, .has_encoder = true},
                           WorkloadKind::Embedding},
        ClassificationCase{"encoder-only, cls pooling",
                           {.pooling = NativePooling::Cls, .has_encoder = true},
                           WorkloadKind::Embedding},
        ClassificationCase{
            "encoder and decoder",
            {.pooling = NativePooling::None, .has_encoder = true, .has_decoder = true},
            WorkloadKind::EncoderDecoder},
        ClassificationCase{
            "encoder and decoder with pooling still enc-dec",
            {.pooling = NativePooling::Mean, .has_encoder = true, .has_decoder = true},
            WorkloadKind::EncoderDecoder},
        ClassificationCase{"rank pooling",
                           {.pooling = NativePooling::Rank, .has_encoder = true},
                           WorkloadKind::Rerank},
        ClassificationCase{
            "classifier head without rank pooling",
            {.pooling = NativePooling::Cls, .has_encoder = true, .classifier_outputs = 1},
            WorkloadKind::Rerank},
        ClassificationCase{
            "rerank wins over encoder-decoder",
            {.pooling = NativePooling::Rank, .has_encoder = true, .has_decoder = true},
            WorkloadKind::Rerank}),
    [](const ::testing::TestParamInfo<ClassificationCase>& info) {
        std::string id = info.param.name;
        std::ranges::replace_if(id, [](unsigned char c) { return !std::isalnum(c); }, '_');
        return id;
    });

// --- make_workload_info ----------------------------------------------------------

TEST(MakeWorkloadInfo, EmbeddingReportsWidthAndPooling) {
    const auto info = make_workload_info(
        {.pooling = NativePooling::Mean, .has_decoder = true, .embedding_dimensions = 768});

    EXPECT_EQ(info.kind, WorkloadKind::Embedding);
    EXPECT_EQ(info.embedding_pooling, Pooling::Mean);
    EXPECT_EQ(info.output_dimensions, 768U);
    EXPECT_EQ(info.classifier_outputs, 0U);
    EXPECT_FALSE(info.has_encoder);
    EXPECT_TRUE(info.has_decoder);
}

TEST(MakeWorkloadInfo, ClassifierCountOutranksPooling) {
    // A classifier head makes it a reranker whatever the pooling says, so the
    // embedding width is withheld and the classifier count is reported.
    const auto info = make_workload_info({.pooling = NativePooling::Mean,
                                          .has_decoder = true,
                                          .embedding_dimensions = 768,
                                          .classifier_outputs = 3});

    EXPECT_EQ(info.kind, WorkloadKind::Rerank);
    EXPECT_EQ(info.embedding_pooling, std::nullopt);
    EXPECT_EQ(info.output_dimensions, 0U);
    EXPECT_EQ(info.classifier_outputs, 3U);
}

TEST(MakeWorkloadInfo, RerankReportsClassifierCountOnly) {
    const auto info = make_workload_info({.pooling = NativePooling::Rank,
                                          .has_encoder = true,
                                          .embedding_dimensions = 1024,
                                          .classifier_outputs = 1});

    EXPECT_EQ(info.kind, WorkloadKind::Rerank);
    EXPECT_EQ(info.pooling, NativePooling::Rank);
    EXPECT_EQ(info.embedding_pooling, std::nullopt); // embed() must refuse this context
    EXPECT_EQ(info.output_dimensions, 0U);
    EXPECT_EQ(info.classifier_outputs, 1U);
}

TEST(MakeWorkloadInfo, GenerationReportsNeither) {
    const auto info = make_workload_info(
        {.pooling = NativePooling::None, .has_decoder = true, .embedding_dimensions = 4096});

    EXPECT_EQ(info.kind, WorkloadKind::CausalGeneration);
    EXPECT_EQ(info.embedding_pooling, std::nullopt);
    EXPECT_EQ(info.output_dimensions, 0U);
    EXPECT_EQ(info.classifier_outputs, 0U);
}

TEST(MakeWorkloadInfo, EncoderOnlyPerTokenKeepsNonePooling) {
    const auto info = make_workload_info(
        {.pooling = NativePooling::None, .has_encoder = true, .embedding_dimensions = 384});

    EXPECT_EQ(info.kind, WorkloadKind::Embedding);
    EXPECT_EQ(info.embedding_pooling, Pooling::None); // per-token path, a token gets selected
    EXPECT_EQ(info.output_dimensions, 384U);
}

// The whole thing is constexpr: a probe known at compile time classifies at compile time.
static_assert(classify_workload({.pooling = NativePooling::Rank}) == WorkloadKind::Rerank);
static_assert(make_workload_info({.pooling = NativePooling::Mean, .embedding_dimensions = 8})
                  .output_dimensions == 8);

} // namespace
