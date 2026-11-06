#pragma once
// autocxx seed: the facade with its shared types spelled out in plain C++,
// since autocxx reads headers and cannot see types a cxx bridge would define.
#include <cstddef>
#include <cstdint>
#include <memory>

namespace skippy { class Model; class Session; }

namespace skippy::ffi {

enum class RuntimeStatus : std::uint8_t {
    Ok, InvalidModel, InvalidSession, ModelInUse, AllocationFailed, LoadFailed,
    SessionCreationFailed, UnsupportedConfiguration, CapacityExceeded, InvalidBackendResult, InternalError,
};
enum class WorkloadStatus : std::uint8_t {
    Ok, InvalidSession, UnsupportedWorkload, UnsupportedClassifierCount, EmptyInput, TooManyTokens,
    TokenizationFailed, NoEmbeddingDimensions, OutputTooSmall, AllocationFailed, DecodeFailed,
    EncodeFailed, NoDecoderStartToken, NoOutput, DimensionMismatch, InternalError,
};
enum class WorkloadKind : std::int8_t { CausalGeneration, Embedding, Rerank, EncoderDecoder };
enum class NativePooling : std::int8_t { Unspecified = -1, None = 0, Mean = 1, Cls = 2, Last = 3, Rank = 4 };

struct SessionConfig { std::uint32_t max_tokens; };
struct WorkloadInfo {
    WorkloadKind kind; NativePooling pooling; NativePooling embedding_pooling;
    std::size_t output_dimensions; std::size_t classifier_outputs; bool has_encoder; bool has_decoder;
};
struct ModelResult { RuntimeStatus status; std::unique_ptr<skippy::Model> model; };
struct SessionResult { RuntimeStatus status; std::unique_ptr<skippy::Session> session; };
struct WorkloadResult { RuntimeStatus status; WorkloadInfo info; };
struct EmbedResult { WorkloadStatus status; std::size_t dimensions; };
struct RerankResult { WorkloadStatus status; float score; std::size_t token_count; };
struct EncodeResult { WorkloadStatus status; std::int32_t decoder_start; };

class Bridge {
public:
    Bridge();
    ~Bridge();
    ModelResult load_model(const char* path, std::size_t length) noexcept;
    RuntimeStatus unload_model(skippy::Model& model) noexcept;
    SessionResult create_session(const skippy::Model& model, SessionConfig config) noexcept;
    RuntimeStatus release_session(skippy::Session& session) noexcept;
    WorkloadResult workload(const skippy::Session& session) const noexcept;
    EmbedResult embed(const skippy::Session& session, const std::int32_t* tokens, std::size_t count,
                      float* output, std::size_t capacity) noexcept;
    RerankResult rerank(const skippy::Session& session, const char* query, std::size_t query_length,
                        const char* document, std::size_t document_length) noexcept;
    EncodeResult encode(const skippy::Session& session, const std::int32_t* tokens, std::size_t count) noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<Bridge> new_bridge() noexcept;

} // namespace skippy::ffi
