#include "rerank.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using skippy::Model;
using skippy::ModelConfig;
using skippy::NativePooling;
using skippy::NativeRuntime;
using skippy::RerankPromptFormat;
using skippy::RerankResult;
using skippy::RuntimeError;
using skippy::Session;
using skippy::SessionConfig;
using skippy::Token;
using skippy::WorkloadError;
using skippy::WorkloadInfo;
using skippy::WorkloadKind;

using ::testing::_;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrictMock;

using TokenizeResult = std::expected<std::span<const Token>, WorkloadError>;
using ScoreResult = std::expected<float, WorkloadError>;
using ScoreManyResult = std::expected<std::span<const float>, WorkloadError>;

// Mocked at the concept boundary: no virtuals, no llama. The static_assert is
// the proof this mock is what rerank() compiles against.
class MockRerankSession {
public:
    MOCK_METHOD(const WorkloadInfo&, workload, (), (const));
    MOCK_METHOD(RerankPromptFormat, rerank_prompt_format, (), (const));
    MOCK_METHOD(TokenizeResult, tokenize, (std::string_view));
    MOCK_METHOD(ScoreResult, score, (std::span<const Token>));
    MOCK_METHOD(ScoreManyResult, score_many, (std::span<const std::span<const Token>>));
};
static_assert(skippy::RerankSession<StrictMock<MockRerankSession>>);

struct RerankBackend {
    struct ModelResource {};
    using SessionResource = StrictMock<MockRerankSession>;

    static std::expected<std::unique_ptr<ModelResource>, RuntimeError>
    load_model(const ModelConfig& /*unused*/) {
        return std::make_unique<ModelResource>();
    }
    static std::expected<std::unique_ptr<SessionResource>, RuntimeError>
    create_session(ModelResource& /*unused*/, const SessionConfig& /*unused*/) {
        return std::make_unique<SessionResource>();
    }
};
static_assert(skippy::RuntimeBackend<RerankBackend>);

MATCHER_P(TokensAre, expected, "") {
    return std::vector<Token>(arg.begin(), arg.end()) == expected;
}

constexpr WorkloadInfo reranker{
    .kind = WorkloadKind::Rerank,
    .pooling = NativePooling::Rank,
    .embedding_pooling = std::nullopt,
    .output_dimensions = 0,
    .classifier_outputs = 1,
    .has_encoder = true,
    .has_decoder = false,
};

class RerankTest : public ::testing::Test {
protected:
    NativeRuntime<RerankBackend> runtime_{std::in_place};
    Model model_ = *runtime_.load_model(ModelConfig{.path = "fake"});
    Session session_ = *runtime_.create_session(model_);
    StrictMock<MockRerankSession>& native_ =
        *runtime_.with_session(session_, [](auto& s) { return &s; }).value();

    WorkloadInfo workload_ = reranker;
    const std::vector<Token> tokens_{Token{5}, Token{6}, Token{7}};

    void expect_workload() {
        EXPECT_CALL(native_, workload()).WillRepeatedly(ReturnRef(workload_));
    }
    void expect_format(RerankPromptFormat format) {
        EXPECT_CALL(native_, rerank_prompt_format()).WillRepeatedly(Return(format));
    }

    std::expected<RerankResult, WorkloadError> rerank(std::string_view query,
                                                      std::string_view document) {
        return skippy::rerank(runtime_, session_, query, document);
    }

    std::expected<skippy::RerankBatchResult, WorkloadError>
    rerank_batch(std::string_view query, std::span<const std::string_view> documents,
                 std::span<float> scores) {
        return skippy::rerank_batch(runtime_, session_, query, documents, scores);
    }
};

// --- checks that never reach the backend ------------------------------------------------

TEST_F(RerankTest, InvalidSessionHandleIsRejected) {
    ASSERT_FALSE(runtime_.release_session(session_).has_value());

    const auto result = rerank("q", "d");

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::InvalidSession);
}

TEST_F(RerankTest, NonRerankWorkloadIsRejectedBeforeTokenizing) {
    workload_.kind = WorkloadKind::Embedding;
    expect_workload();

    const auto result = rerank("q", "d");

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::UnsupportedWorkload);
}

TEST_F(RerankTest, MultipleClassifierOutputsAreRejected) {
    workload_.classifier_outputs = 2;
    expect_workload();

    const auto result = rerank("q", "d");

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::UnsupportedClassifierCount);
}

// --- prompt assembly reaches the tokenizer -----------------------------------------------

TEST_F(RerankTest, UsesTheModelTemplateWhenItHasOne) {
    expect_workload();
    expect_format({.prompt_template = "Q: {query}\nD: {document}",
                   .eos_piece = "</s>",
                   .sep_piece = "[SEP]"});
    EXPECT_CALL(native_, tokenize(std::string_view{"Q: hello\nD: world"}))
        .WillOnce(Return(TokenizeResult{std::span<const Token>{tokens_}}));
    EXPECT_CALL(native_, score(_)).WillOnce(Return(ScoreResult{0.5F}));

    EXPECT_TRUE(rerank("hello", "world").has_value());
}

TEST_F(RerankTest, FallsBackToQueryEosSepDocument) {
    expect_workload();
    expect_format({.prompt_template = std::nullopt, .eos_piece = "</s>", .sep_piece = "[SEP]"});
    EXPECT_CALL(native_, tokenize(std::string_view{"hello</s>[SEP]world"}))
        .WillOnce(Return(TokenizeResult{std::span<const Token>{tokens_}}));
    EXPECT_CALL(native_, score(_)).WillOnce(Return(ScoreResult{0.5F}));

    EXPECT_TRUE(rerank("hello", "world").has_value());
}

// --- the native boundary --------------------------------------------------------------------

TEST_F(RerankTest, TokenizeErrorPropagatesAndNothingIsScored) {
    expect_workload();
    expect_format({});
    EXPECT_CALL(native_, tokenize(_))
        .WillOnce(Return(TokenizeResult{std::unexpected(WorkloadError::TokenizationFailed)}));

    const auto result = rerank("q", "d");

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::TokenizationFailed);
}

TEST_F(RerankTest, EmptyTokenizationIsAFailureNotAScore) {
    expect_workload();
    expect_format({});
    EXPECT_CALL(native_, tokenize(_)).WillOnce(Return(TokenizeResult{std::span<const Token>{}}));

    const auto result = rerank("q", "d");

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::TokenizationFailed);
}

TEST_F(RerankTest, ScoreReceivesTheTokenizedPrompt) {
    expect_workload();
    expect_format({});
    EXPECT_CALL(native_, tokenize(_))
        .WillOnce(Return(TokenizeResult{std::span<const Token>{tokens_}}));
    EXPECT_CALL(native_, score(TokensAre(tokens_))).WillOnce(Return(ScoreResult{0.25F}));

    EXPECT_TRUE(rerank("q", "d").has_value());
}

TEST_F(RerankTest, ScoreErrorPropagates) {
    expect_workload();
    expect_format({});
    EXPECT_CALL(native_, tokenize(_))
        .WillOnce(Return(TokenizeResult{std::span<const Token>{tokens_}}));
    EXPECT_CALL(native_, score(_))
        .WillOnce(Return(ScoreResult{std::unexpected(WorkloadError::DecodeFailed)}));

    const auto result = rerank("q", "d");

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::DecodeFailed);
}

// --- the happy path ---------------------------------------------------------------------------

TEST_F(RerankTest, ReturnsScoreAndTokenCount) {
    expect_workload();
    expect_format({});
    EXPECT_CALL(native_, tokenize(_))
        .WillOnce(Return(TokenizeResult{std::span<const Token>{tokens_}}));
    EXPECT_CALL(native_, score(_)).WillOnce(Return(ScoreResult{-1.75F}));

    const auto result = rerank("q", "d");

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->score, -1.75F);
    EXPECT_EQ(result->token_count, 3U);
}

// --- compile-time contract --------------------------------------------------------------------

struct MissingScore {
    [[nodiscard]] static const WorkloadInfo& workload() { return reranker; }
    [[nodiscard]] static RerankPromptFormat rerank_prompt_format() { return {}; }
    static TokenizeResult tokenize(std::string_view /*unused*/) { return {}; }
};
static_assert(!skippy::RerankSession<MissingScore>);

} // namespace

// --- batched scoring --------------------------------------------------------------------

TEST_F(RerankTest, BatchComposesOnePromptPerDocumentAndCopiesEveryScore) {
    expect_workload();
    expect_format(
        RerankPromptFormat{.prompt_template = std::nullopt, .eos_piece = "", .sep_piece = ""});

    const std::array<Token, 3> first{Token{5}, Token{6}, Token{7}};
    const std::array<Token, 2> second{Token{8}, Token{9}};
    EXPECT_CALL(native_, tokenize(std::string_view{"qdoc one"}))
        .WillOnce(Return(std::span<const Token>{first}));
    EXPECT_CALL(native_, tokenize(std::string_view{"qdoc two"}))
        .WillOnce(Return(std::span<const Token>{second}));

    const std::array<float, 2> native_scores{0.75F, -1.5F};
    EXPECT_CALL(native_, score_many(_)).WillOnce(Return(std::span<const float>{native_scores}));

    const std::array<std::string_view, 2> documents{"doc one", "doc two"};
    std::array<float, 2> scores{};

    const auto result = rerank_batch("q", documents, scores);

    ASSERT_TRUE(result.has_value()) << static_cast<int>(result.error());
    EXPECT_EQ(result->count, 2U);
    EXPECT_EQ(result->token_count, 5U);
    EXPECT_THAT(scores, ::testing::ElementsAre(0.75F, -1.5F));
}

TEST_F(RerankTest, BatchRejectsAShortScoreBufferBeforeTouchingTheBackend) {
    const std::array<std::string_view, 2> documents{"a", "b"};
    std::array<float, 1> scores{};

    const auto result = rerank_batch("q", documents, scores);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::OutputTooSmall);
}

TEST_F(RerankTest, BatchRejectsAnEmptyDocumentGroup) {
    const std::array<std::string_view, 0> documents{};
    std::array<float, 1> scores{};

    const auto result = rerank_batch("q", documents, scores);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), WorkloadError::EmptyInput);
}
