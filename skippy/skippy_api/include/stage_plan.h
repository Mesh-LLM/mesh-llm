#ifndef SKIPPY_API_STAGE_PLAN_H
#define SKIPPY_API_STAGE_PLAN_H

/**
 * @file stage_plan.h
 * @brief Typed, pointer-free description of a realized staged plan.
 *
 * A stage plan is the metadata answer to "which layers does this stage hold,
 * which activation values cross its boundaries, and which persistent state
 * does it touch, for each execution profile". It is pure data: describing and
 * validating a plan needs no llama and no backend, which is what makes it
 * testable without a model.
 *
 * The native planner produces these values; this header is the typed shape
 * they take once they enter the facade. Layer ranges are half-open:
 * `[start, end)`.
 */

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace skippy {

/// A string that means one specific thing. Tags stop ids from being swapped.
template <typename Tag> struct StrongString {
    std::string value;

    [[nodiscard]] std::string_view view() const noexcept { return value; }
    [[nodiscard]] bool empty() const noexcept { return value.empty(); }

    friend bool operator==(const StrongString&, const StrongString&) = default;
    friend auto operator<=>(const StrongString&, const StrongString&) = default;
};

struct PackageIdTag;
struct PlanIdTag;
struct ProfileIdTag;
struct GraphIdentityTag;
struct ProfileIdentityTag;
struct SliceIdentityTag;
struct SourceSnapshotIdentityTag;
struct GraphConfigurationIdTag;
struct BackendIdTag;
struct TensorIdTag;
struct ValueIdentityTag;
struct ValueBindingTag;

using PackageId = StrongString<PackageIdTag>;
using PlanId = StrongString<PlanIdTag>;
using ProfileId = StrongString<ProfileIdTag>;
using GraphIdentity = StrongString<GraphIdentityTag>;
using ProfileIdentity = StrongString<ProfileIdentityTag>;
using SliceIdentity = StrongString<SliceIdentityTag>;
using SourceSnapshotIdentity = StrongString<SourceSnapshotIdentityTag>;
using GraphConfigurationId = StrongString<GraphConfigurationIdTag>;
using BackendId = StrongString<BackendIdTag>;
using TensorId = StrongString<TensorIdTag>;
/// Semantic identity of an activation value crossing a stage boundary.
/// A slice into a StagePlan's identity arena: offset+length, never a pointer,
/// so slices survive the plan's vector moves and a clone rebuilds them by
/// replaying the same offsets into the clone's own arena.
struct ArenaSlice {
    std::uint32_t offset = 0;
    std::uint32_t length = 0;
};

template <typename Tag> struct StrongSlice {
    ArenaSlice slice;

    friend bool operator==(const StrongSlice&, const StrongSlice&) = default;
    friend auto operator<=>(const StrongSlice&, const StrongSlice&) = default;
};

using TensorSlice = StrongSlice<TensorIdTag>;
using ValueIdentitySlice = StrongSlice<ValueIdentityTag>;
using ValueBindingSlice = StrongSlice<ValueBindingTag>;

using ValueIdentity = StrongString<ValueIdentityTag>;
/// Where that value lives in the stage's tensor namespace.
using ValueBinding = StrongString<ValueBindingTag>;

/// Half-open layer range, `[start, end)`, within `[0, layer_count)`.
struct LayerRange {
    std::int32_t start = 0;
    std::int32_t end = 0;

    [[nodiscard]] bool empty() const noexcept { return start >= end; }
    [[nodiscard]] std::int32_t count() const noexcept { return end - start; }
    [[nodiscard]] bool contains(std::int32_t layer) const noexcept {
        return layer >= start && layer < end;
    }

    friend bool operator==(const LayerRange&, const LayerRange&) = default;
};

/// Which side of a profile boundary a value identity names.
enum class ValueKind : std::uint8_t {
    ActivationImport,
    ActivationExport,
    RequestInput,
};

/// Kind of persistent state a stage reads or writes.
enum class StateKind : std::uint8_t {
    KvKey,
    KvValue,
    RecurrentConvolution,
    RecurrentSsm,
};

enum class StateAccess : std::uint8_t {
    Read,
    Write,
};

enum class StateResidency : std::uint8_t {
    LayerLocal,
    PerStage,
};

/// One persistent-state effect of a profile.
/// What a stage does to one piece of persistent state. The identity bytes are
/// deliberately absent: no consumer reads them, and materialising one string per
/// effect made up over half of the copy cost of a plan.
struct StateEffect {
    StateKind kind = StateKind::KvKey;
    StateAccess access = StateAccess::Read;
    std::int32_t layer = 0;
    StateResidency residency = StateResidency::LayerLocal;
    std::int64_t write_ordinal = 0;
};

/// One activation value crossing a stage boundary, borrowing its identity
/// strings from the owning plan's arena.
struct FrontierValue {
    ValueIdentitySlice identity;
    ValueBindingSlice binding;
};

/// The assembly-input form: owns its strings until `StagePlan::append` moves
/// them into the plan's arena.
struct OwnedValue {
    ValueIdentity identity;
    ValueBinding binding;
};

/// A profile with its own value storage — the assembly input for a plan.
/// `StagePlan::assemble` moves these values into the plan's flat storage and
/// hands back span-backed profiles that borrow from it.
struct OwnedProfile {
    ProfileId id;
    GraphIdentity graph_identity;
    ProfileIdentity profile_identity;
    SliceIdentity slice_identity;
    SourceSnapshotIdentity source_snapshot;
    GraphConfigurationId graph_configuration;
    BackendId backend;

    /// The slice this profile was built for; must equal the owning stage range.
    LayerRange slice_range;

    std::uint32_t tokens = 0;
    std::uint32_t sequences = 0;
    std::uint32_t outputs = 0;
    std::uint32_t recurrent_rollback_sequences = 0;

    std::vector<OwnedValue> activation_imports;
    std::vector<OwnedValue> activation_exports;
    std::vector<OwnedValue> request_inputs;
    std::vector<StateEffect> state_effects;
};

/// Guards and geometry for one execution profile of a stage. The value spans
/// borrow from the owning StagePlan's flat storage, so a profile is only ever
/// obtained from a plan and never outlives it.
/// A position in a plan's flat value storage: where a profile's values
/// begin, and how many. Resolved to a span by the owning plan on access.
struct ValueRange {
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
};

struct PlanProfile {
    ProfileId id;
    GraphIdentity graph_identity;
    ProfileIdentity profile_identity;
    SliceIdentity slice_identity;
    SourceSnapshotIdentity source_snapshot;
    GraphConfigurationId graph_configuration;
    BackendId backend;

    /// The slice this profile was built for; must equal the owning stage range.
    LayerRange slice_range;

    std::uint32_t tokens = 0;
    std::uint32_t sequences = 0;
    std::uint32_t outputs = 0;
    std::uint32_t recurrent_rollback_sequences = 0;

    /// Positions in the owning plan's flat storage, not pointers. A span
    /// stored here dangles the moment a later append grows the vector;
    /// a position cannot.
    ValueRange activation_imports;
    ValueRange activation_exports;
    ValueRange request_inputs;
    ValueRange state_effects;
};

struct StagePlan;

/// Identities only: the frontier contract is about semantics, not storage.
/// Free functions because resolving a slice needs the owning plan's arena.
[[nodiscard]] std::vector<ValueIdentity> import_identities(const StagePlan& plan,
                                                           const PlanProfile& profile);
[[nodiscard]] std::vector<ValueIdentity> export_identities(const StagePlan& plan,
                                                           const PlanProfile& profile);

/// One realized, immutable stage.
///
/// Move-only by design: profiles hold spans into this plan's own flat value
/// storage, and a vector move transfers the buffer (element addresses are
/// stable), while a copy would reallocate and leave every span dangling. Use
/// `clone()` where a second copy of a plan is genuinely needed.
struct StagePlan {
    PackageId package;
    PlanId id;
    std::int32_t layer_count = 0;
    LayerRange range;

    /// Opaque native execution contract, carried to the runtime unchanged.
    std::string execution_contract;

    /// One allocation holding every resident and frontier identity string.
    /// Slices index into it; view() is the only way to read one.
    std::string identity_arena;

    /// Flat per-plan storage; each profile spans into these, so a plan with
    /// many profiles pays four value allocations in total instead of four per
    /// profile.
    std::vector<FrontierValue> activation_imports;
    std::vector<FrontierValue> activation_exports;
    std::vector<FrontierValue> request_inputs;
    std::vector<StateEffect> state_effects;

    std::vector<PlanProfile> profiles;
    std::vector<TensorSlice> resident_identities;

    /// A profile's values as a span, resolved from the recorded position.
    /// Spans are never stored: a stored span dangles when a later append
    /// grows the flat vector underneath it.
    [[nodiscard]] std::span<const FrontierValue>
    activation_imports_of(const PlanProfile& profile) const noexcept {
        return std::span{activation_imports}.subspan(profile.activation_imports.offset,
                                                     profile.activation_imports.count);
    }
    [[nodiscard]] std::span<const FrontierValue>
    activation_exports_of(const PlanProfile& profile) const noexcept {
        return std::span{activation_exports}.subspan(profile.activation_exports.offset,
                                                     profile.activation_exports.count);
    }
    [[nodiscard]] std::span<const FrontierValue>
    request_inputs_of(const PlanProfile& profile) const noexcept {
        return std::span{request_inputs}.subspan(profile.request_inputs.offset,
                                                 profile.request_inputs.count);
    }
    [[nodiscard]] std::span<const StateEffect>
    state_effects_of(const PlanProfile& profile) const noexcept {
        return std::span{state_effects}.subspan(profile.state_effects.offset,
                                                profile.state_effects.count);
    }

    /// The bytes a slice refers to. The only reader for arena identities.
    /// Pointer arithmetic, not substr: a corrupt slice yields a short or empty
    /// view rather than throwing, and slices are plan-produced so this cannot
    /// happen in practice.
    [[nodiscard]] std::string_view view(ArenaSlice of) const noexcept {
        const char* const bytes = identity_arena.data() + of.offset;
        return std::string_view{bytes, of.length};
    }

    StagePlan() = default;
    StagePlan(const StagePlan&) = delete;
    StagePlan& operator=(const StagePlan&) = delete;
    StagePlan(StagePlan&&) = default;
    StagePlan& operator=(StagePlan&&) = default;

    /// Pre-reserves the flat value storage. Call once, before the first
    /// `append`, when the totals are known (the planner knows them from the
    /// profile descriptors); `append` itself never reserves, because reserving
    /// to an exact cumulative size per profile would reallocate on every call.
    void reserve_totals(std::size_t imports, std::size_t exports, std::size_t requests,
                        std::size_t effects);

    /// Moves each profile's values into this plan's flat storage and appends a
    /// span-backed profile.
    void append(OwnedProfile profile);

    /// A deep copy with spans rebuilt into the copy's own storage.
    [[nodiscard]] StagePlan clone() const;

    [[nodiscard]] bool terminal() const noexcept { return range.end == layer_count; }
};

/// What a caller asks the planner to guard for one execution profile.
struct ProfileSpec {
    ProfileId id;
    std::uint32_t tokens = 0;
    std::uint32_t sequences = 1;
    std::uint32_t outputs = 0;
    std::uint32_t recurrent_rollback_sequences = 0;
};

/// Why a plan, or a chain of plans, is not usable.
enum class PlanError : std::uint8_t {
    EmptyChain,
    InvalidRange,
    IncompletePartition,
    LayerCountMismatch,
    PackageMismatch,
    ProfileCountMismatch,
    ProfileIdentityMismatch,
    ActivationFrontierMismatch,
    /// The backend refused the request; `PlanFailure::message` says why.
    Rejected,
    /// The backend failed after accepting the request.
    BackendFailed,
};

/// A plan error plus the backend's explanation, when there is one.
struct PlanFailure {
    PlanError code = PlanError::BackendFailed;
    std::string message;
};

/// What a plan's state effects say about the stage, without naming a family.
///
/// This is the facade's reading of the per-model graph semantics: a stage that
/// touches recurrent SSM state is recurrent whatever architecture produced it.
struct PlanSemantics {
    /// Distinct state kinds this stage touches, in ascending enum order.
    std::vector<StateKind> state_kinds;
    /// True when any layer holds recurrent (convolution or SSM) state.
    bool recurrent = false;
    /// Lowest and highest layer with a state effect; -1 when there are none.
    std::int32_t first_layer = -1;
    std::int32_t last_layer = -1;
    /// State effects that are written, not just read.
    std::size_t written = 0;
    /// Values crossing a stage boundary in either direction.
    std::size_t frontier_values = 0;
};

/// Summarizes one profile's semantics; the plan must have exactly one profile
/// for this to be unambiguous.
[[nodiscard]] std::expected<PlanSemantics, PlanError> summarize(const StagePlan& plan,
                                                                const ProfileId& profile);

/// State effects of one profile at one layer, in plan order.
[[nodiscard]] std::vector<StateEffect>
state_effects_at(const StagePlan& plan, const PlanProfile& profile, std::int32_t layer);

/// Exact package tensor binding the planner needs.
struct PackageTensor {
    TensorId id;
    std::string native_name;
    std::int32_t ggml_type = 0;
    std::uint32_t dimension_count = 0;
    std::array<std::int64_t, 4> dimensions{};
    std::uint32_t split_no = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t stored_length = 0;
};

/// Everything a planner needs to build a metadata-only model.
struct StagePlannerInputs {
    PackageId package;
    std::vector<std::string> shard_paths;
    std::vector<PackageTensor> tensors;
    std::vector<ProfileSpec> profiles;
    GraphConfigurationId graph_configuration;
    BackendId backend;
};

/// Rejects inputs no planner could use, before any backend work starts.
[[nodiscard]] std::optional<PlanFailure> validate_inputs(const StagePlannerInputs& inputs);

/// Checks one plan on its own: range is non-empty and inside the model.
[[nodiscard]] std::optional<PlanError> validate(const StagePlan& plan);

/// Checks that `stages` form one complete, compatible chain.
///
/// Rules, in the order they are reported: a non-empty chain; every plan
/// individually valid; ranges partition `[0, layer_count)` in order; adjacent
/// stages agree on model, profile count and per-profile guard; and each
/// stage's exports are exactly the next stage's imports.
[[nodiscard]] std::optional<PlanError> validate_chain(std::span<const StagePlan> stages);

/// Stable name of a rejection reason, for logs and error messages.
[[nodiscard]] std::string_view describe(PlanError error) noexcept;

} // namespace skippy

#endif // SKIPPY_API_STAGE_PLAN_H
