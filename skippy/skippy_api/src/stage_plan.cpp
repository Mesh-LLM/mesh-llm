#include "stage_plan.h"

#include <algorithm>
#include <iterator>
#include <ranges>
#include <span>

namespace skippy {
namespace {

std::vector<ValueIdentity> identities_of(const StagePlan& plan,
                                         std::span<const FrontierValue> values) {
    std::vector<ValueIdentity> identities;
    identities.reserve(values.size());
    std::ranges::transform(values, std::back_inserter(identities), [&](const FrontierValue& value) {
        return ValueIdentity{std::string{plan.view(value.identity.slice)}};
    });
    return identities;
}

// Guards that must be identical for adjacent stages to hand off a frontier.
bool guards_match(const PlanProfile& left, const PlanProfile& right) {
    return left.id == right.id && left.graph_identity == right.graph_identity &&
           left.profile_identity == right.profile_identity;
}

} // namespace

std::vector<ValueIdentity> import_identities(const StagePlan& plan, const PlanProfile& profile) {
    return identities_of(plan, plan.activation_imports_of(profile));
}

std::vector<ValueIdentity> export_identities(const StagePlan& plan, const PlanProfile& profile) {
    return identities_of(plan, plan.activation_exports_of(profile));
}

std::optional<PlanFailure> validate_inputs(const StagePlannerInputs& inputs) {
    if (inputs.package.empty()) {
        return PlanFailure{
            .code = PlanError::PackageMismatch,
            .message = "package id is empty",
        };
    }
    if (inputs.shard_paths.empty()) {
        return PlanFailure{
            .code = PlanError::Rejected,
            .message = "a planner needs at least one shard",
        };
    }
    if (inputs.profiles.empty()) {
        return PlanFailure{
            .code = PlanError::ProfileCountMismatch,
            .message = "a planner needs at least one profile",
        };
    }
    return std::nullopt;
}

std::expected<PlanSemantics, PlanError> summarize(const StagePlan& plan, const ProfileId& profile) {
    const auto found = std::ranges::find_if(
        plan.profiles, [&](const PlanProfile& entry) { return entry.id == profile; });
    if (found == plan.profiles.end()) return std::unexpected(PlanError::ProfileIdentityMismatch);

    PlanSemantics semantics;
    for (const StateEffect& effect : plan.state_effects_of(*found)) {
        if (std::ranges::find(semantics.state_kinds, effect.kind) == semantics.state_kinds.end()) {
            semantics.state_kinds.push_back(effect.kind);
        }
        if (effect.kind == StateKind::RecurrentConvolution ||
            effect.kind == StateKind::RecurrentSsm) {
            semantics.recurrent = true;
        }
        if (effect.access == StateAccess::Write) ++semantics.written;
        semantics.first_layer = semantics.first_layer < 0
                                    ? effect.layer
                                    : std::min(semantics.first_layer, effect.layer);
        semantics.last_layer = std::max(semantics.last_layer, effect.layer);
    }
    std::ranges::sort(semantics.state_kinds);
    semantics.frontier_values =
        plan.activation_imports_of(*found).size() + plan.activation_exports_of(*found).size();
    return semantics;
}

std::vector<StateEffect> state_effects_at(const StagePlan& plan, const PlanProfile& profile,
                                          std::int32_t layer) {
    std::vector<StateEffect> effects;
    std::ranges::copy_if(plan.state_effects_of(profile), std::back_inserter(effects),
                         [layer](const StateEffect& effect) { return effect.layer == layer; });
    return effects;
}

std::optional<PlanError> validate(const StagePlan& plan) {
    if (plan.layer_count <= 0 || plan.range.start < 0 || plan.range.start >= plan.range.end ||
        plan.range.end > plan.layer_count) {
        return PlanError::InvalidRange;
    }
    if (plan.package.empty()) return PlanError::PackageMismatch;
    if (plan.profiles.empty()) return PlanError::ProfileCountMismatch;
    const bool slice_matches = std::ranges::all_of(plan.profiles, [&](const PlanProfile& profile) {
        return profile.slice_range == plan.range;
    });
    if (!slice_matches) return PlanError::ProfileIdentityMismatch;
    return std::nullopt;
}

namespace {

bool ranges_join(std::span<const StagePlan> stages, std::size_t index) {
    const StagePlan& stage = stages[index];
    if (index == 0) return stage.range.start == 0;
    return stage.range.start == stages[index - 1].range.end;
}

bool ends_at_model_end(const StagePlan& stage) { return stage.range.end == stage.layer_count; }

std::optional<PlanError> validate_handoff(const StagePlan& previous, const StagePlan& stage) {
    if (stage.profiles.size() != previous.profiles.size()) {
        return PlanError::ProfileCountMismatch;
    }
    for (const std::size_t index : std::views::iota(std::size_t{0}, stage.profiles.size())) {
        const PlanProfile& profile = stage.profiles[index];
        const PlanProfile& handoff = previous.profiles[index];
        if (!guards_match(handoff, profile)) {
            return PlanError::ProfileIdentityMismatch;
        }
        // What the earlier stage exports must be what this one imports, in
        // order: the frontier is a list of values, not a set.
        if (export_identities(previous, handoff) != import_identities(stage, profile)) {
            return PlanError::ActivationFrontierMismatch;
        }
    }
    return std::nullopt;
}

std::optional<PlanError> validate_link(std::span<const StagePlan> stages, std::size_t index,
                                       const PackageId& package) {
    const StagePlan& stage = stages[index];
    if (stage.package != package) return PlanError::PackageMismatch;
    if (stage.layer_count != stages.front().layer_count) {
        return PlanError::LayerCountMismatch;
    }
    if (!ranges_join(stages, index)) return PlanError::IncompletePartition;
    const bool last = index + 1 == stages.size();
    if (last && !ends_at_model_end(stage)) {
        return PlanError::IncompletePartition;
    }
    if (index == 0) return std::nullopt;
    return validate_handoff(stages[index - 1], stage);
}

} // namespace

std::string_view describe(PlanError error) noexcept {
    switch (error) {
    case PlanError::EmptyChain: return "empty chain";
    case PlanError::InvalidRange: return "invalid range";
    case PlanError::IncompletePartition: return "incomplete partition";
    case PlanError::LayerCountMismatch: return "layer count mismatch";
    case PlanError::PackageMismatch: return "package mismatch";
    case PlanError::ProfileCountMismatch: return "profile count mismatch";
    case PlanError::ProfileIdentityMismatch: return "profile identity mismatch";
    case PlanError::ActivationFrontierMismatch: return "activation frontier mismatch";
    case PlanError::Rejected: return "rejected";
    case PlanError::BackendFailed: return "backend failed";
    }
    return "unknown";
}

std::optional<PlanError> validate_chain(std::span<const StagePlan> stages) {
    if (stages.empty()) return PlanError::EmptyChain;

    for (const StagePlan& stage : stages) {
        if (auto failure = validate(stage); failure.has_value()) return failure;
    }

    const PackageId package = stages.front().package;
    for (const std::size_t index : std::views::iota(std::size_t{0}, stages.size())) {
        if (auto failure = validate_link(stages, index, package); failure.has_value())
            return failure;
    }
    return std::nullopt;
}

void StagePlan::reserve_totals(std::size_t imports, std::size_t exports, std::size_t requests,
                               std::size_t effects) {
    activation_imports.reserve(imports);
    activation_exports.reserve(exports);
    request_inputs.reserve(requests);
    state_effects.reserve(effects);
}

namespace {

/// Appends bytes to the arena and returns the slice that names them.
ArenaSlice arena_put(std::string& arena, std::string_view bytes) {
    const ArenaSlice slice{
        .offset = static_cast<std::uint32_t>(arena.size()),
        .length = static_cast<std::uint32_t>(bytes.size()),
    };
    arena.append(bytes);
    return slice;
}

FrontierValue into_arena(std::string& arena, const OwnedValue& value) {
    return FrontierValue{
        .identity = ValueIdentitySlice{arena_put(arena, value.identity.value)},
        .binding = ValueBindingSlice{arena_put(arena, value.binding.value)},
    };
}

} // namespace

void StagePlan::append(OwnedProfile profile) {
    const auto imports = static_cast<std::size_t>(activation_imports.size());
    const auto exports = static_cast<std::size_t>(activation_exports.size());
    const auto requests = static_cast<std::size_t>(request_inputs.size());
    const auto effects = static_cast<std::size_t>(state_effects.size());

    identity_arena.reserve(identity_arena.size() + (profile.activation_imports.size() * 64) +
                           (profile.activation_exports.size() * 64) +
                           (profile.request_inputs.size() * 64));
    std::ranges::transform(
        profile.activation_imports, std::back_inserter(activation_imports),
        [&](const OwnedValue& value) { return into_arena(identity_arena, value); });
    std::ranges::transform(
        profile.activation_exports, std::back_inserter(activation_exports),
        [&](const OwnedValue& value) { return into_arena(identity_arena, value); });
    std::ranges::transform(
        profile.request_inputs, std::back_inserter(request_inputs),
        [&](const OwnedValue& value) { return into_arena(identity_arena, value); });
    state_effects.insert(state_effects.end(),
                         std::make_move_iterator(profile.state_effects.begin()),
                         std::make_move_iterator(profile.state_effects.end()));

    profiles.push_back(PlanProfile{
        .id = std::move(profile.id),
        .graph_identity = std::move(profile.graph_identity),
        .profile_identity = std::move(profile.profile_identity),
        .slice_identity = std::move(profile.slice_identity),
        .source_snapshot = std::move(profile.source_snapshot),
        .graph_configuration = std::move(profile.graph_configuration),
        .backend = std::move(profile.backend),
        .slice_range = profile.slice_range,
        .tokens = profile.tokens,
        .sequences = profile.sequences,
        .outputs = profile.outputs,
        .recurrent_rollback_sequences = profile.recurrent_rollback_sequences,
        .activation_imports =
            ValueRange{
                .offset = static_cast<std::uint32_t>(imports),
                .count = static_cast<std::uint32_t>(activation_imports.size() - imports),
            },
        .activation_exports =
            ValueRange{
                .offset = static_cast<std::uint32_t>(exports),
                .count = static_cast<std::uint32_t>(activation_exports.size() - exports),
            },
        .request_inputs =
            ValueRange{
                .offset = static_cast<std::uint32_t>(requests),
                .count = static_cast<std::uint32_t>(request_inputs.size() - requests),
            },
        .state_effects =
            ValueRange{
                .offset = static_cast<std::uint32_t>(effects),
                .count = static_cast<std::uint32_t>(state_effects.size() - effects),
            },
    });
}

namespace {
std::vector<OwnedValue> owned_values(const StagePlan& plan, std::span<const FrontierValue> values) {
    std::vector<OwnedValue> owned;
    owned.reserve(values.size());
    for (const FrontierValue& value : values) {
        owned.push_back(OwnedValue{
            .identity = ValueIdentity{std::string{plan.view(value.identity.slice)}},
            .binding = ValueBinding{std::string{plan.view(value.binding.slice)}},
        });
    }
    return owned;
}
} // namespace

StagePlan StagePlan::clone() const {
    StagePlan copy;
    copy.package = package;
    copy.id = id;
    copy.layer_count = layer_count;
    copy.range = range;
    copy.execution_contract = execution_contract;
    // The arena is copied wholesale, so every slice offset in the copied
    // profiles and residents still points at the same bytes — no replay needed.
    copy.identity_arena = identity_arena;
    copy.activation_imports = activation_imports;
    copy.activation_exports = activation_exports;
    copy.request_inputs = request_inputs;
    copy.state_effects = state_effects;
    copy.resident_identities = resident_identities; // slices; arena copied above
    // Re-derive each profile's spans against the copy's own (now separate)
    // storage by replaying the same offsets.
    std::size_t imports = 0;
    std::size_t exports = 0;
    std::size_t requests = 0;
    std::size_t effects = 0;
    copy.profiles.reserve(profiles.size());
    for (const PlanProfile& profile : profiles) {
        copy.append(OwnedProfile{
            .id = profile.id,
            .graph_identity = profile.graph_identity,
            .profile_identity = profile.profile_identity,
            .slice_identity = profile.slice_identity,
            .source_snapshot = profile.source_snapshot,
            .graph_configuration = profile.graph_configuration,
            .backend = profile.backend,
            .slice_range = profile.slice_range,
            .tokens = profile.tokens,
            .sequences = profile.sequences,
            .outputs = profile.outputs,
            .recurrent_rollback_sequences = profile.recurrent_rollback_sequences,
            .activation_imports = owned_values(*this, activation_imports_of(profile)),
            .activation_exports = owned_values(*this, activation_exports_of(profile)),
            .request_inputs = owned_values(*this, request_inputs_of(profile)),
            .state_effects = {state_effects_of(profile).begin(), state_effects_of(profile).end()},
        });
        imports += profile.activation_imports.count;
        exports += profile.activation_exports.count;
        requests += profile.request_inputs.count;
        effects += profile.state_effects.count;
    }
    (void)imports;
    (void)exports;
    (void)requests;
    (void)effects;
    return copy;
}
} // namespace skippy
