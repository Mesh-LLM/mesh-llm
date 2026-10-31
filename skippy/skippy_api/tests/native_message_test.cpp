// P1-2: the backend's failure text travels in the failure value, so OOM and
// a Metal failure reach the caller as distinguishable WorkloadFailures. The
// doubles model what LlamaSession does: a native pass failure returns
// unexpected(WorkloadFailure::native<...>(what())) — never a bare code with
// the reason parked in a member.
#include "embed.h"
#include "encode.h"
#include "rerank.h"
#include "types.h"
#include "workload.h"
#include "workload_error.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

const skippy::Token one[]{skippy::Token{1}};

using skippy::EmbeddingOutput;
using skippy::F32Format;
using skippy::ModelConfig;
using skippy::NativeRuntime;
using skippy::RuntimeError;
using skippy::Session;
using skippy::SessionConfig;
using skippy::Token;
using skippy::WorkloadError;
using skippy::WorkloadFailure;
using skippy::WorkloadInfo;
using skippy::WorkloadKind;
using skippy::detail::NativeEmbeddingView;

constexpr std::string_view kMetalFailure = "metal: command buffer failed";

template <typename SessionDouble> struct BackendOf {
    struct ModelResource {};
    using SessionResource = SessionDouble;

    static std::expected<std::unique_ptr<ModelResource>, skippy::LoadFailure>
    load_model(const ModelConfig& /*unused*/) {
        return std::make_unique<ModelResource>();
    }
    static std::expected<std::unique_ptr<SessionResource>, RuntimeError>
    create_session(ModelResource& /*unused*/, const SessionConfig& /*unused*/) {
        return std::make_unique<SessionDouble>();
    }
};

// embed / embed_batch: the decode pass fails with the backend's own text.
class MetalEmbeddingSession {
public:
    std::size_t embedding_dimensions() const { return 3; }
    std::expected<NativeEmbeddingView, WorkloadFailure> embed(std::span<const Token>) {
        return failed();
    }
    std::expected<NativeEmbeddingView, WorkloadFailure>
    embed_many(std::span<const std::span<const Token>>) {
        return failed();
    }

private:
    static std::unexpected<WorkloadFailure> failed() {
        return std::unexpected(
            WorkloadFailure::native<WorkloadError::DecodeFailed>(std::string{kMetalFailure}));
    }
};
static_assert(skippy::EmbeddingSession<MetalEmbeddingSession>);

// A pass that failed before a message could be recorded (bad_alloc while
// recording it): the code still arrives, with an empty message.
class UnexplainedDecodeSession {
public:
    std::size_t embedding_dimensions() const { return 3; }
    std::expected<NativeEmbeddingView, WorkloadFailure> embed(std::span<const Token>) {
        return std::unexpected(WorkloadFailure::native<WorkloadError::DecodeFailed>(std::string{}));
    }
    std::expected<NativeEmbeddingView, WorkloadFailure>
    embed_many(std::span<const std::span<const Token>>) {
        return std::unexpected(WorkloadFailure::native<WorkloadError::DecodeFailed>(std::string{}));
    }
};
static_assert(skippy::EmbeddingSession<UnexplainedDecodeSession>);

// rerank: the score pass fails with the backend's own text.
class MetalRerankSession {
public:
    const WorkloadInfo& workload() const {
        static constexpr WorkloadInfo reranker{
            .kind = WorkloadKind::Rerank,
            .classifier_outputs = 1,
        };
        return reranker;
    }
    skippy::RerankPromptFormat rerank_prompt_format() const { return {}; }
    std::expected<std::span<const Token>, WorkloadFailure> tokenize(std::string_view) {
        return std::span<const Token>{tokens_};
    }
    std::expected<float, WorkloadFailure> score(std::span<const Token>) {
        return std::unexpected(
            WorkloadFailure::native<WorkloadError::DecodeFailed>(std::string{kMetalFailure}));
    }
    std::expected<std::span<const float>, WorkloadFailure>
    score_many(std::span<const std::span<const Token>>) {
        return std::unexpected(
            WorkloadFailure::native<WorkloadError::DecodeFailed>(std::string{kMetalFailure}));
    }

private:
    std::vector<Token> tokens_{Token{7}};
};
static_assert(skippy::RerankSession<MetalRerankSession>);

// encode_prompt: the encode pass fails with the backend's own text.
class MetalEncoderSession {
public:
    const WorkloadInfo& workload() const {
        static constexpr WorkloadInfo encoder_decoder{
            .kind = WorkloadKind::EncoderDecoder,
        };
        return encoder_decoder;
    }
    std::expected<Token, WorkloadFailure> encode(std::span<const Token>) {
        return std::unexpected(
            WorkloadFailure::native<WorkloadError::EncodeFailed>(std::string{kMetalFailure}));
    }
};
static_assert(skippy::EncoderSession<MetalEncoderSession>);

TEST(NativeMessage, EmbedCarriesTheBackendText) {
    NativeRuntime<BackendOf<MetalEmbeddingSession>> runtime{std::in_place};
    const ModelConfig config{.path = "fake"};
    auto model = *runtime.load_model(config);
    const Session session = *runtime.create_session(model);

    std::array<float, 3> values{};
    const auto result = skippy::embed(runtime, session, std::span<const Token>{one},
                                      EmbeddingOutput<F32Format>{values, {}});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, WorkloadError::DecodeFailed);
    EXPECT_EQ(result.error().message, kMetalFailure);
}

TEST(NativeMessage, EmbedBatchCarriesTheBackendText) {
    NativeRuntime<BackendOf<MetalEmbeddingSession>> runtime{std::in_place};
    const ModelConfig config{.path = "fake"};
    auto model = *runtime.load_model(config);
    const Session session = *runtime.create_session(model);

    std::array<float, 3> values{};
    const skippy::Token input[]{skippy::Token{1}};
    const std::span<const Token> inputs[]{std::span<const Token>{input}};
    const auto result =
        skippy::embed_batch(runtime, session, std::span<const std::span<const Token>>{inputs},
                            EmbeddingOutput<F32Format>{values, {}});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, WorkloadError::DecodeFailed);
    EXPECT_EQ(result.error().message, kMetalFailure);
}

TEST(NativeMessage, RerankCarriesTheBackendText) {
    NativeRuntime<BackendOf<MetalRerankSession>> runtime{std::in_place};
    const ModelConfig config{.path = "fake"};
    auto model = *runtime.load_model(config);
    const Session session = *runtime.create_session(model);

    const auto result = skippy::rerank(runtime, session, "query", "document");

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, WorkloadError::DecodeFailed);
    EXPECT_EQ(result.error().message, kMetalFailure);
}

TEST(NativeMessage, EncodePromptCarriesTheBackendText) {
    NativeRuntime<BackendOf<MetalEncoderSession>> runtime{std::in_place};
    const ModelConfig config{.path = "fake"};
    auto model = *runtime.load_model(config);
    const Session session = *runtime.create_session(model);

    const auto result = skippy::encode_prompt(runtime, session, std::span<const Token>{one});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, WorkloadError::EncodeFailed);
    EXPECT_EQ(result.error().message, kMetalFailure);
}

TEST(NativeMessage, AnUnrecordedPassStillDeliversTheCode) {
    NativeRuntime<BackendOf<UnexplainedDecodeSession>> runtime{std::in_place};
    const ModelConfig config{.path = "fake"};
    auto model = *runtime.load_model(config);
    const Session session = *runtime.create_session(model);

    std::array<float, 3> values{};
    const auto result = skippy::embed(runtime, session, std::span<const Token>{one},
                                      EmbeddingOutput<F32Format>{values, {}});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, WorkloadError::DecodeFailed);
    EXPECT_TRUE(result.error().message.empty());
}

} // namespace