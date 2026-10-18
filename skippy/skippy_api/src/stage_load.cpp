#include "stage_load.h"

#include <algorithm>

namespace skippy {

std::expected<StageLoadConfig, PlanError> stage_load_config(const StagePlan& plan,
                                                            const ProfileId& profile) {
    const auto found = std::ranges::find_if(
        plan.profiles, [&](const PlanProfile& entry) { return entry.id == profile; });
    if (found == plan.profiles.end()) return std::unexpected(PlanError::ProfileIdentityMismatch);
    if (plan.resident_identities.empty()) return std::unexpected(PlanError::ProfileCountMismatch);
    if (plan.execution_contract.empty()) {
        return std::unexpected(PlanError::ActivationFrontierMismatch);
    }

    StageLoadConfig config;
    config.layers = plan.range;
    config.execution_contract = plan.execution_contract;

    config.resident_identities.reserve(plan.resident_identities.size());
    std::ranges::transform(plan.resident_identities, std::back_inserter(config.resident_identities),
                           [&](const TensorSlice& tensor) { return plan.view(tensor.slice); });

    // Identities and bindings are paired: the loader indexes them together.
    config.activation_import_identities.reserve(found->activation_imports.size());
    config.activation_import_bindings.reserve(found->activation_imports.size());
    for (const FrontierValue& value : found->activation_imports) {
        config.activation_import_identities.push_back(plan.view(value.identity.slice));
        config.activation_import_bindings.push_back(plan.view(value.binding.slice));
    }
    if (const auto semantics = summarize(plan, profile); semantics.has_value()) {
        config.profile_id = found->id.view();
        config.graph_identity = found->graph_identity.view();
        config.profile_identity = found->profile_identity.view();
        config.slice_identity = found->slice_identity.view();
        config.source_snapshot = found->source_snapshot.view();
        config.graph_configuration = found->graph_configuration.view();
        config.backend = found->backend.view();
        config.frontier_values = semantics->frontier_values;
        config.state_effects = found->state_effects.size();
        config.written_state_effects = semantics->written;
        config.recurrent_state = semantics->recurrent;
    }

    config.activation_export_identities.reserve(found->activation_exports.size());
    config.request_input_identities.reserve(found->request_inputs.size());
    config.request_input_bindings.reserve(found->request_inputs.size());
    for (const auto& input : found->request_inputs) {
        config.request_input_identities.push_back(plan.view(input.identity.slice));
        config.request_input_bindings.push_back(plan.view(input.binding.slice));
    }
    config.activation_export_bindings.reserve(found->activation_exports.size());
    for (const FrontierValue& value : found->activation_exports) {
        config.activation_export_identities.push_back(plan.view(value.identity.slice));
        config.activation_export_bindings.push_back(plan.view(value.binding.slice));
    }
    return config;
}

} // namespace skippy
