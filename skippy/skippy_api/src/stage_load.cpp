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
    const auto import_values = plan.activation_imports_of(*found);
    config.activation_import_identities.reserve(import_values.size());
    config.activation_import_bindings.reserve(import_values.size());
    for (const FrontierValue& value : import_values) {
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
        config.state_effects = plan.state_effects_of(*found).size();
        config.written_state_effects = semantics->written;
        config.recurrent_state = semantics->recurrent;
    }

    const auto export_values = plan.activation_exports_of(*found);
    config.activation_export_identities.reserve(export_values.size());
    const auto request_values = plan.request_inputs_of(*found);
    config.request_input_identities.reserve(request_values.size());
    config.request_input_bindings.reserve(request_values.size());
    for (const auto& input : request_values) {
        config.request_input_identities.push_back(plan.view(input.identity.slice));
        config.request_input_bindings.push_back(plan.view(input.binding.slice));
    }
    config.activation_export_bindings.reserve(export_values.size());
    for (const FrontierValue& value : export_values) {
        config.activation_export_identities.push_back(plan.view(value.identity.slice));
        config.activation_export_bindings.push_back(plan.view(value.binding.slice));
    }
    return config;
}

} // namespace skippy
