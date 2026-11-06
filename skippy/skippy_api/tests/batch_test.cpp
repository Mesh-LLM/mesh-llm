#include "batch.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

using skippy::SequenceId;
using skippy::Token;
using skippy::detail::fill_embedding_batch;
using skippy::detail::TokenBatch;

using ::testing::Each;
using ::testing::ElementsAre;

struct FakeBatch {
    explicit FakeBatch(std::size_t capacity)
        : token_(capacity), pos_(capacity), n_seq_id_(capacity),
          seq_storage_(capacity, std::array<std::int32_t, 1>{-1}), seq_id_(capacity),
          logits_(capacity), token(token_.data()), pos(pos_.data()), n_seq_id(n_seq_id_.data()),
          seq_id(seq_id_.data()), logits(logits_.data()) {
        std::ranges::transform(seq_storage_, seq_id_.begin(),
                               [](auto& slot) { return slot.data(); });
    }

    std::vector<std::int32_t> token_;
    std::vector<std::int32_t> pos_;
    std::vector<std::int32_t> n_seq_id_;
    std::vector<std::array<std::int32_t, 1>> seq_storage_;
    std::vector<std::int32_t*> seq_id_;
    std::vector<std::int8_t> logits_;

    std::int32_t n_tokens = -1;
    std::int32_t* token;
    std::int32_t* pos;
    std::int32_t* n_seq_id;
    std::int32_t** seq_id;
    std::int8_t* logits;
};
static_assert(TokenBatch<FakeBatch>);

struct NotABatch {
    int n_tokens;
};
static_assert(!TokenBatch<NotABatch>);

class FillEmbeddingBatch : public ::testing::Test {
protected:
    const std::vector<Token> tokens_{Token{11}, Token{22}, Token{33}};
    static constexpr SequenceId sequence_{7};
    FakeBatch batch_{3};

    void SetUp() override { fill_embedding_batch(batch_, tokens_, sequence_); }
};

TEST_F(FillEmbeddingBatch, PreservesTokenCountAndIds) {
    EXPECT_EQ(batch_.n_tokens, 3);
    EXPECT_THAT(batch_.token_, ElementsAre(11, 22, 33));
}

TEST_F(FillEmbeddingBatch, AssignsConsecutivePositionsStartingAtZero) {
    EXPECT_THAT(batch_.pos_, ElementsAre(0, 1, 2));
}

TEST_F(FillEmbeddingBatch, AssignsSuppliedSequenceIdToEveryToken) {
    EXPECT_THAT(batch_.n_seq_id_, Each(1));
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(batch_.seq_id_[i][0], 7) << "at index " << i;
    }
}

TEST_F(FillEmbeddingBatch, RequestsOutputForEveryToken) { EXPECT_THAT(batch_.logits_, Each(1)); }

TEST(FillEmbeddingBatchCapacity, WritesOnlyAsManyEntriesAsTokens) {
    FakeBatch batch{5};
    const std::vector<Token> tokens{Token{1}, Token{2}};

    fill_embedding_batch(batch, tokens, SequenceId{3});

    EXPECT_EQ(batch.n_tokens, 2);
    EXPECT_THAT(batch.token_, ElementsAre(1, 2, 0, 0, 0));
    EXPECT_THAT(batch.logits_, ElementsAre(1, 1, 0, 0, 0));
}

TEST(FillEmbeddingBatchCapacity, MaxTokensIsInt32Max) {
    EXPECT_EQ(skippy::detail::max_batch_tokens, 2147483647U);
}

} // namespace
