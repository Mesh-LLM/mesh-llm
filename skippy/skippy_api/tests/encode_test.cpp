#include "encode.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <expected>
#include <memory>
#include <span>
#include <vector>

namespace {

using skippy::Model;
using skippy::ModelConfig;
using skippy::NativePooling;
using skippy::NativeRuntime;
using skippy::RuntimeError;
using skippy::Session;
using skippy::SessionConfig;
using skippy::Token;
using skippy::WorkloadError;
using skippy::WorkloadFailure;
using skippy::WorkloadInfo;
using skippy::WorkloadKind;

using ::testing::_;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrictMock;

using EncodeResult = std::expected<Token, WorkloadFailure>;

class MockEncoderSession {
public:
    MOCK_METHOD(const WorkloadInfo&, workload, (), (const));
    MOCK_METHOD(EncodeResult, encode, (std::span<const Token>));
};
static_assert(skippy::EncoderSession<StrictMock<MockEncoderSession>>);

struct EncoderBackend {
    struct ModelResource {};
    using SessionResource = StrictMock<MockEncoderSession>;

    static std::expected<std::unique_ptr<ModelResource>, skippy::LoadFailure>
    load_model(const ModelConfig& /*unused*/) {
        return std::make_unique<ModelResource>();
    }
    static std::expected<std::unique_ptr<SessionResource>, RuntimeError>
    create_session(ModelResource& /*unused*/, const SessionConfig& /*unused*/) {
        return std::make_unique<SessionResource>();
    }
};
static_assert(skippy::RuntimeBackend<EncoderBackend>);

MATCHER_P(TokensAre, expected, "") {
    return std::vector<Token>(arg.begin(), arg.end()) == expected;
}

constexpr WorkloadInfo encoder_decoder{
    .kind = WorkloadKind::EncoderDecoder,
    .pooling = NativePooling::None,
    .embedding_pooling = std::nullopt,
    .output_dimensions = 0,
    .classifier_outputs = 0,
    .has_encoder = true,
    .has_decoder = true,
};

class EncodePromptTest : public ::testing::Test {
protected:
    NativeRuntime<EncoderBackend> runtime_{std::in_place};
    Model model_ = *runtime_.load_model(ModelConfig{.path = "fake"});
    Session session_ = *runtime_.create_session(model_);
    StrictMock<MockEncoderSession>& native_ =
        *runtime_.with_session(session_, [](auto& s) { return &s; }).value();

    WorkloadInfo workload_ = encoder_decoder;
    const std::vector<Token> tokens_{Token{3}, Token{4}};

    void expect_workload() {
        EXPECT_CALL(native_, workload()).WillRepeatedly(ReturnRef(workload_));
    }

    std::expected<Token, WorkloadFailure> encode(std::span<const Token> tokens) {
        return skippy::encode_prompt(runtime_, session_, tokens);
    }
};

TEST_F(EncodePromptTest, EmptyInputIsRejectedWithoutTouchingTheSession) {
    const auto result = encode({});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, WorkloadError::EmptyInput);
}

TEST_F(EncodePromptTest, InvalidSessionHandleIsRejected) {
    ASSERT_TRUE(runtime_.release_session(session_).has_value());

    const auto result = encode(tokens_);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, WorkloadError::InvalidSession);
}

TEST_F(EncodePromptTest, NonEncoderDecoderWorkloadIsRejectedBeforeEncoding) {
    workload_.kind = WorkloadKind::Embedding;
    expect_workload();

    const auto result = encode(tokens_);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, WorkloadError::UnsupportedWorkload);
}

TEST_F(EncodePromptTest, PassesTokensThroughAndReturnsDecoderStart) {
    expect_workload();
    EXPECT_CALL(native_, encode(TokensAre(tokens_))).WillOnce(Return(EncodeResult{Token{1}}));

    const auto result = encode(tokens_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, Token{1});
}

TEST_F(EncodePromptTest, NativeErrorPropagates) {
    expect_workload();
    EXPECT_CALL(native_, encode(_))
        .WillOnce(Return(EncodeResult{
            std::unexpected(WorkloadFailure::code_only(WorkloadError::EncodeFailed))}));

    const auto result = encode(tokens_);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, WorkloadError::EncodeFailed);
}

struct MissingEncode {
    [[nodiscard]] static const WorkloadInfo& workload() { return encoder_decoder; }
};
static_assert(!skippy::EncoderSession<MissingEncode>);

} // namespace