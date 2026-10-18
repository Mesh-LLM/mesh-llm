#include "embed.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <memory>

namespace {

using skippy::EmbeddingOutput;
using skippy::EmbeddingView;
using skippy::F32Format;
using skippy::Model;
using skippy::ModelConfig;
using skippy::NativeRuntime;
using skippy::RuntimeError;
using skippy::Session;
using skippy::SessionConfig;
using skippy::Token;
using skippy::WorkloadError;
using skippy::detail::NativeEmbeddingView;

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Return;
using ::testing::StrictMock;

using NativeResult = std::expected<NativeEmbeddingView, WorkloadError>;

class MockEmbeddingSession {
public:
    MOCK_METHOD(std::size_t, embedding_dimensions, (), (const));
    MOCK_METHOD(NativeResult, embed, (std::span<const Token>));
    MOCK_METHOD(NativeResult, embed_many, (std::span<const std::span<const Token>>));
};
static_assert(skippy::EmbeddingSession<StrictMock<MockEmbeddingSession>>);

struct EmbeddingBackend {
    struct ModelResource {};
    using SessionResource = StrictMock<MockEmbeddingSession>;

    static std::expected<std::unique_ptr<ModelResource>, RuntimeError>
    load_model(const ModelConfig& /*unused*/) {
        return std::make_unique<ModelResource>();
    }
    static std::expected<std::unique_ptr<SessionResource>, RuntimeError>
    create_session(ModelResource& /*unused*/, const SessionConfig& /*unused*/) {
        return std::make_unique<SessionResource>();
    }
};
static_assert(skippy::RuntimeBackend<EmbeddingBackend>);

MATCHER_P(TokensAre, expected, "") {
    return std::vector<Token>(arg.begin(), arg.end()) == expected;
}

class EmbedTest : public ::testing::Test {
protected:
    NativeRuntime<EmbeddingBackend> runtime_{std::in_place};
    Model model_ = *runtime_.load_model(ModelConfig{.path = "fake"});
    Session session_ = *runtime_.create_session(model_);
    StrictMock<MockEmbeddingSession>& native_ =
        *runtime_.with_session(session_, [](auto& s) { return &s; }).value();

    const std::vector<Token> tokens_{Token{11}, Token{22}, Token{33}};
    std::array<float, 4> native_values_{1.0F, -2.0F, 2.0F, -4.0F}; // |v| = 5
    std::array<float, 6> buffer_{99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F};

    EmbeddingOutput<F32Format> output(std::size_t capacity) {
        return {
            .values = std::span<float>{buffer_}.first(capacity),
            .metadata = {},
        };
    }

    void expect_dimensions(std::size_t dimensions) {
        EXPECT_CALL(native_, embedding_dimensions()).WillRepeatedly(Return(dimensions));
    }

    std::expected<EmbeddingView<F32Format>, WorkloadError> embed(std::span<const Token> tokens,
                                                                 std::size_t capacity) {
        return skippy::embed(runtime_, session_, tokens, output(capacity));
    }

    std::expected<skippy::EmbeddingBatchView<F32Format>, WorkloadError>
    embed_batch(std::span<const std::span<const Token>> inputs, std::size_t capacity) {
        return skippy::embed_batch(runtime_, session_, inputs, output(capacity));
    }
};

TEST_F(EmbedTest, EmptyInputIsRejectedWithoutTouchingTheSession) {
    const auto result = embed({}, 4);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::EmptyInput);
}

TEST_F(EmbedTest, InvalidSessionHandleIsRejected) {
    ASSERT_FALSE(runtime_.release_session(session_).has_value());

    const auto result = embed(tokens_, 4);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::InvalidSession);
}

TEST_F(EmbedTest, ZeroDimensionsIsRejectedBeforeDecoding) {
    expect_dimensions(0);

    const auto result = embed(tokens_, 4);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::NoEmbeddingDimensions);
}

TEST_F(EmbedTest, OutputTooSmallIsRejectedBeforeDecoding) {
    expect_dimensions(4);

    const auto result = embed(tokens_, 3);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::OutputTooSmall);
    EXPECT_THAT(buffer_, ElementsAre(99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F));
}

TEST_F(EmbedTest, PassesTokensThroughUnchanged) {
    expect_dimensions(4);
    EXPECT_CALL(native_, embed(TokensAre(tokens_)))
        .WillOnce(Return(NativeResult{NativeEmbeddingView{native_values_}}));

    EXPECT_TRUE(embed(tokens_, 4).has_value());
}

TEST_F(EmbedTest, NativeErrorPropagatesAndLeavesOutputUntouched) {
    expect_dimensions(4);
    EXPECT_CALL(native_, embed(_))
        .WillOnce(Return(NativeResult{std::unexpected(WorkloadError::DecodeFailed)}));

    const auto result = embed(tokens_, 4);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::DecodeFailed);
    EXPECT_THAT(buffer_, ElementsAre(99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F));
}

TEST_F(EmbedTest, NativeWidthMismatchIsRejected) {
    expect_dimensions(4);
    EXPECT_CALL(native_, embed(_))
        .WillOnce(Return(NativeResult{NativeEmbeddingView{native_values_}.first(3)}));

    const auto result = embed(tokens_, 4);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::DimensionMismatch);
    EXPECT_THAT(buffer_, ElementsAre(99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F));
}

TEST_F(EmbedTest, NormalisesIntoTheCallerBufferAndReturnsAViewOfIt) {
    expect_dimensions(4);
    EXPECT_CALL(native_, embed(_))
        .WillOnce(Return(NativeResult{NativeEmbeddingView{native_values_}}));

    const auto result = embed(tokens_, 6);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->dimensions, 4U);
    EXPECT_EQ(result->values.data(), buffer_.data());
    EXPECT_EQ(result->values.size(), 4U);
    EXPECT_THAT(result->values, ElementsAre(0.2F, -0.4F, 0.4F, -0.8F));
    EXPECT_FLOAT_EQ(buffer_[4], 99.0F);
    EXPECT_FLOAT_EQ(buffer_[5], 99.0F);
}

TEST_F(EmbedTest, ExactCapacityIsEnough) {
    expect_dimensions(4);
    EXPECT_CALL(native_, embed(_))
        .WillOnce(Return(NativeResult{NativeEmbeddingView{native_values_}}));

    const auto result = embed(tokens_, 4);

    ASSERT_TRUE(result.has_value());
    EXPECT_THAT(result->values, ElementsAre(0.2F, -0.4F, 0.4F, -0.8F));
}

TEST_F(EmbedTest, NativeViewDoesNotEscape) {
    expect_dimensions(4);
    EXPECT_CALL(native_, embed(_))
        .WillOnce(Return(NativeResult{NativeEmbeddingView{native_values_}}));

    const auto result = embed(tokens_, 4);

    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->values.data(), native_values_.data());
    // the buffer is owned by llama. Ensure untouched.
    EXPECT_THAT(native_values_, ElementsAre(1.0F, -2.0F, 2.0F, -4.0F));
}

struct NotAnEmbeddingSession {};
static_assert(!skippy::EmbeddingSession<NotAnEmbeddingSession>);

struct WrongReturnType {
    [[nodiscard]] static std::size_t embedding_dimensions() { return 1; }
    static std::span<const float> embed(std::span<const Token> /*unused*/) {
        return {};
    } // not an expected
};
static_assert(!skippy::EmbeddingSession<WrongReturnType>);

} // namespace

// --- batched embedding ------------------------------------------------------------------

TEST_F(EmbedTest, BatchNormalisesEveryRowIntoItsOwnSlot) {
    expect_dimensions(4);
    const std::array<float, 8> rows{1.0F, -2.0F, 2.0F, -4.0F, // |v| = 5
                                    0.0F, 3.0F,  0.0F, 4.0F}; // |v| = 5
    std::array<float, 8> buffer{};
    std::vector<std::span<const Token>> inputs{tokens_, tokens_};

    EXPECT_CALL(native_, embed_many(_)).WillOnce(Return(NativeEmbeddingView{rows}));

    const auto result = skippy::embed_batch(
        runtime_, session_, inputs, EmbeddingOutput<F32Format>{.values = buffer, .metadata = {}});

    ASSERT_TRUE(result.has_value()) << static_cast<int>(result.error());
    EXPECT_EQ(result->count, 2U);
    EXPECT_EQ(result->dimensions, 4U);
    EXPECT_THAT(buffer, ElementsAre(0.2F, -0.4F, 0.4F, -0.8F, 0.0F, 0.6F, 0.0F, 0.8F));
}

TEST_F(EmbedTest, BatchRejectsAnEmptyRowBeforeTheBackend) {
    expect_dimensions(4);
    const std::vector<Token> empty;
    const std::vector<std::span<const Token>> inputs{tokens_, empty};

    const auto result = skippy::embed_batch(runtime_, session_, inputs, output(8));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::EmptyInput);
}

TEST_F(EmbedTest, BatchRejectsAShortBuffer) {
    expect_dimensions(4);
    const std::vector<std::span<const Token>> inputs{tokens_, tokens_};

    const auto result = skippy::embed_batch(runtime_, session_, inputs, output(7));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::OutputTooSmall);
}

TEST_F(EmbedTest, BatchRejectsAWidthMismatch) {
    expect_dimensions(4);
    const std::array<float, 7> rows{}; // one value short of two rows
    std::array<float, 8> buffer{};
    const std::vector<std::span<const Token>> inputs{tokens_, tokens_};

    EXPECT_CALL(native_, embed_many(_)).WillOnce(Return(NativeEmbeddingView{rows}));

    const auto result = skippy::embed_batch(
        runtime_, session_, inputs, EmbeddingOutput<F32Format>{.values = buffer, .metadata = {}});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::DimensionMismatch);
}
