#include "llama/staged.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace skippy::llama {
namespace {

StagedFailure failure(skippy_status status, skippy_error* error) {
    StagedFailure result{.status = static_cast<int>(status), .message = {}};
    if (error != nullptr) {
        if (error->message != nullptr) result.message = error->message;
        skippy_error_free(error);
    }
    return result;
}

} // namespace

StageSpec own(const StageLoadConfig& config) {
    StageSpec spec;
    spec.stage_index = config.stage_index;
    spec.layers = config.layers;
    spec.execution_contract.assign(config.execution_contract);
    spec.resident_identities.reserve(config.resident_identities.size());
    for (const std::string_view value : config.resident_identities) {
        spec.resident_identities.emplace_back(value);
    }
    const auto owned = [](std::span<const std::string_view> values) {
        std::vector<std::string> copy;
        copy.reserve(values.size());
        std::ranges::transform(values, std::back_inserter(copy),
                               [](std::string_view value) { return std::string{value}; });
        return copy;
    };
    spec.import_identities = owned(config.activation_import_identities);
    spec.import_bindings = owned(config.activation_import_bindings);
    spec.export_identities = owned(config.activation_export_identities);
    spec.export_bindings = owned(config.activation_export_bindings);
    return spec;
}

StagedModel::~StagedModel() {
    if (model_ != nullptr) {
        skippy_error* error = nullptr;
        (void)skippy_model_free(model_, &error);
        if (error != nullptr) skippy_error_free(error);
    }
}

StagedModel::StagedModel(StagedModel&& other) noexcept
    : model_(std::exchange(other.model_, nullptr)), layers_(other.layers_) {}

std::expected<StagedModel, StagedFailure> StagedModel::open(const std::string& path,
                                                            const StageSpec& spec) {
    std::vector<const char*> residents;
    residents.reserve(spec.resident_identities.size());
    for (const std::string& value : spec.resident_identities)
        residents.push_back(value.c_str());

    std::vector<const char*> imports;
    imports.reserve(spec.import_identities.size());
    for (const std::string& value : spec.import_identities)
        imports.push_back(value.c_str());
    std::vector<const char*> import_bindings;
    import_bindings.reserve(spec.import_bindings.size());
    for (const std::string& value : spec.import_bindings)
        import_bindings.push_back(value.c_str());
    std::vector<const char*> exports;
    exports.reserve(spec.export_identities.size());
    for (const std::string& value : spec.export_identities)
        exports.push_back(value.c_str());
    std::vector<const char*> export_bindings;
    export_bindings.reserve(spec.export_bindings.size());
    for (const std::string& value : spec.export_bindings)
        export_bindings.push_back(value.c_str());

    auto config = skippy_runtime_config_default();
    config.stage_index = spec.stage_index;
    config.layer_start = spec.layers.start;
    config.layer_end = spec.layers.end;
    config.ctx_size = 512;
    config.n_batch = 512;
    config.n_ubatch = 512;
    config.n_gpu_layers = 999;
    config.load_mode = SKIPPY_LOAD_MODE_LAYER_PACKAGE;
    config.execution_contract = spec.execution_contract.c_str();
    config.resident_tensor_names = residents.data();
    config.resident_tensor_name_count = residents.size();
    config.activation_import_identities = imports.empty() ? nullptr : imports.data();
    config.activation_import_identity_count = imports.size();
    config.activation_import_bindings = import_bindings.empty() ? nullptr : import_bindings.data();
    config.activation_export_identities = exports.empty() ? nullptr : exports.data();
    config.activation_export_identity_count = exports.size();
    config.activation_export_bindings = export_bindings.empty() ? nullptr : export_bindings.data();

    skippy_model* model = nullptr;
    skippy_error* error = nullptr;
    const skippy_status status = skippy_model_open(path.c_str(), &config, &model, &error);
    if (status != SKIPPY_STATUS_OK) return std::unexpected(failure(status, error));
    return StagedModel{model, spec.layers};
}

std::expected<StagedSession, StagedFailure> StagedModel::create_session() const {
    skippy_session* session = nullptr;
    skippy_error* error = nullptr;
    const skippy_status status = skippy_session_create(model_, &session, &error);
    if (status != SKIPPY_STATUS_OK) return std::unexpected(failure(status, error));
    return StagedSession{session};
}

StagedSession::~StagedSession() {
    if (session_ != nullptr) {
        skippy_error* error = nullptr;
        (void)skippy_session_free(session_, &error);
        if (error != nullptr) skippy_error_free(error);
    }
}

StagedSession::StagedSession(StagedSession&& other) noexcept
    : session_(std::exchange(other.session_, nullptr)) {}

std::expected<std::uint32_t, StagedFailure>
StagedSession::embed(std::span<const llama_token> tokens, std::span<float> output) {
    std::size_t written = 0;
    skippy_error* error = nullptr;
    const skippy_status status = skippy_session_embed(
        session_, tokens.data(), tokens.size(), output.data(), output.size(), &written, &error);
    if (status != SKIPPY_STATUS_OK) return std::unexpected(failure(status, error));
    return static_cast<std::uint32_t>(written);
}

} // namespace skippy::llama
