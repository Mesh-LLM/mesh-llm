#ifndef SKIPPY_API_LLAMA_STAGED_H
#define SKIPPY_API_LLAMA_STAGED_H

/// Staged models: a slice of a model, opened through the skippy ABI.
///
/// This is a second model type, not a flag on the plain one. A plain `Model`
/// wraps `llama_model*` and decodes on a `llama_context*`; a staged model
/// wraps `skippy_model*` and runs workloads on a `skippy_session*`. They share
/// no ownership rules, so keeping them apart keeps both honest.

#include "stage_load.h"

namespace skippy::llama {
using StageSpec = StageLoadConfig::Spec;
}
#include "stage_plan.h"

#include "skippy.h"

#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace skippy::llama {

/// Why a staged operation failed.
struct StagedFailure {
    /// Status the native call returned.
    int status = 0;
    /// The backend's own explanation, when it produced one.
    std::string message;
};

class StagedSession;

/// Owns one stage of a model.
class StagedModel {
public:
    StagedModel() = default;
    ~StagedModel();
    StagedModel(StagedModel&& other) noexcept;
    StagedModel& operator=(StagedModel&&) = delete;
    StagedModel(const StagedModel&) = delete;
    StagedModel& operator=(const StagedModel&) = delete;

    /// Opens `path` as the stage described by `spec`.
    [[nodiscard]] static std::expected<StagedModel, StagedFailure> open(const std::string& path,
                                                                        const StageSpec& spec);

    [[nodiscard]] std::expected<StagedSession, StagedFailure> create_session() const;

    [[nodiscard]] LayerRange layers() const noexcept { return layers_; }
    [[nodiscard]] skippy_model* native() const noexcept { return model_; }

private:
    StagedModel(skippy_model* model, LayerRange layers) noexcept : model_(model), layers_(layers) {}

    skippy_model* model_ = nullptr;
    LayerRange layers_{};
};

/// Owns one session on a stage.
class StagedSession {
public:
    StagedSession() = default;
    ~StagedSession();
    StagedSession(StagedSession&& other) noexcept;
    StagedSession& operator=(StagedSession&&) = delete;
    StagedSession(const StagedSession&) = delete;
    StagedSession& operator=(const StagedSession&) = delete;

    /// Embeds `tokens` into `output`, which must hold `dimensions` values.
    [[nodiscard]] std::expected<std::uint32_t, StagedFailure>
    embed(std::span<const llama_token> tokens, std::span<float> output);

    [[nodiscard]] skippy_session* native() const noexcept { return session_; }

private:
    friend class StagedModel;
    explicit StagedSession(skippy_session* session) noexcept : session_(session) {}

    skippy_session* session_ = nullptr;
};

/// Converts a planned loader config into an owning spec.
[[nodiscard]] StageSpec own(const StageLoadConfig& config);

} // namespace skippy::llama

#endif // SKIPPY_API_LLAMA_STAGED_H
