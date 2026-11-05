#include "workload_profile.h"

#include <gtest/gtest.h>

namespace {

using skippy::classify;
using skippy::DeclaredPooling;
using skippy::ModelFacts;

TEST(WorkloadProfile, ABareCausalDecoderIsNotAnEmbeddingWorkload) {
    const auto profile = classify(ModelFacts{.has_decoder = true});
    EXPECT_FALSE(profile.embedding_workload);
    EXPECT_FALSE(profile.classifier_workload);
    EXPECT_FALSE(profile.encoder_decoder);
    EXPECT_FALSE(profile.lanes_must_serialize);
}

TEST(WorkloadProfile, AnEncoderOnlyModelIsAnEmbeddingWorkload) {
    // BERT-style: encoder, no decoder, no classifier head, no declared pooling.
    const auto profile = classify(ModelFacts{.has_encoder = true});
    EXPECT_TRUE(profile.embedding_workload);
    EXPECT_FALSE(profile.classifier_workload);
}

TEST(WorkloadProfile, AClassifierHeadWithNoDeclaredPoolingIsAReranker) {
    // The reranker case from the wild: cls head present, pooling unspecified.
    // RANK pooling must be forced for the score to be readable.
    const auto profile = classify(ModelFacts{.classifier_outputs = 1});
    EXPECT_TRUE(profile.classifier_workload);
    EXPECT_TRUE(profile.embedding_workload);
}

TEST(WorkloadProfile, AClassifierHeadWithDeclaredPoolingIsNotAReranker) {
    // MiniLM carries classifier labels but declares mean pooling: the declared
    // pooling wins, so it stays an embedder. This is the classification that
    // first-generation probing got wrong in both directions.
    const auto profile =
        classify(ModelFacts{.classifier_outputs = 1, .declared_pooling = DeclaredPooling::Mean});
    EXPECT_FALSE(profile.classifier_workload);
    EXPECT_TRUE(profile.embedding_workload);
}

TEST(WorkloadProfile, DeclaredRankPoolingIsARerankerEvenWithoutLabels) {
    const auto profile = classify(ModelFacts{.declared_pooling = DeclaredPooling::Rank});
    EXPECT_FALSE(profile.classifier_workload); // no head to force
    EXPECT_TRUE(profile.embedding_workload);
}

TEST(WorkloadProfile, DeclaredNonePoolingStillLeavesTheHeadInCharge) {
    // NONE means no pooling was declared, not "never pool": a classifier head
    // with no pooling to honour is a reranker, exactly as patch 0011 forces
    // RANK for any `model->cls` present.
    const auto profile = classify(ModelFacts{
        .has_decoder = true, .classifier_outputs = 3, .declared_pooling = DeclaredPooling::None});
    EXPECT_TRUE(profile.classifier_workload);
    EXPECT_TRUE(profile.embedding_workload);
}

TEST(WorkloadProfile, ABareDecoderWithDeclaredNoneIsNotAnEmbeddingWorkload) {
    // The disabling case: no head, no encoder, no usable pooling.
    const auto profile =
        classify(ModelFacts{.has_decoder = true, .declared_pooling = DeclaredPooling::None});
    EXPECT_FALSE(profile.embedding_workload);
    EXPECT_FALSE(profile.classifier_workload);
}

TEST(WorkloadProfile, AnEncoderDecoderSerializesAndDoesNotEmbedd) {
    // Patch 0011's lane rule: encoder output is context-global, so shared
    // state must serialize. The embeddings flag stays off because the decode
    // half consumes them, matching the native patch.
    const auto profile = classify(ModelFacts{.has_encoder = true, .has_decoder = true});
    EXPECT_TRUE(profile.encoder_decoder);
    EXPECT_TRUE(profile.lanes_must_serialize);
    EXPECT_FALSE(profile.embedding_workload);
}

TEST(WorkloadProfile, DeclaredPoolingValuesAreConstrained) {
    // Every known value classifies; the table must not reject any of them.
    for (const auto pooling :
         {DeclaredPooling::None, DeclaredPooling::Mean, DeclaredPooling::Cls, DeclaredPooling::Last,
          DeclaredPooling::Rank, DeclaredPooling::Unspecified}) {
        const auto profile = classify(ModelFacts{.declared_pooling = pooling});
        EXPECT_NO_FATAL_FAILURE(static_cast<void>(profile));
    }
}

} // namespace
