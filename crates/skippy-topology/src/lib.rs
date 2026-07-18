use serde::{Deserialize, Serialize};

mod artifact_diagnostics;
mod edge_order;
mod family_capability;
mod planning;
mod validation;

pub use edge_order::StageEdgeSignal;
pub use family_capability::{
    STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS, deepseek2_capability, deepseek2ocr_capability,
    deepseek3_capability, dense_attention_layers, dense_family_capability, falcon_h1_capability,
    falcon_h1_layers, gemma2_capability, gemma3_capability, gemma3n_capability,
    gemma4_a4b_capability, gemma4_e4b_capability, glm4_capability, glm47_flash_capability,
    infer_family_capability, kimi_linear_capability, llama_capability, minimax_m27_capability,
    olmo_capability, qwen2moe_capability, qwen3_dense_capability, qwen3moe_capability,
    qwen3next_capability, qwen3next_layers, recurrent_family_capability,
    reviewed_capability_for_identity, reviewed_capability_records, rwkv6_capability,
    rwkv7_capability,
};
pub use planning::{
    classify_layers, plan_contiguous_with_splits, plan_even_contiguous,
    plan_package_aware_contiguous, plan_package_aware_contiguous_with_signals,
    plan_package_aware_contiguous_with_transport, plan_weighted_contiguous,
    wire_payload_bytes_per_token,
};
pub use validation::PlanError;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TopologyPlanRequest {
    pub topology_id: String,
    pub model_id: String,
    pub layers: Vec<LayerSpec>,
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub family: Option<FamilyCapabilityRecord>,
    #[serde(default)]
    pub policy: PlannerPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LayerSpec {
    pub index: u32,
    #[serde(default)]
    pub attention: bool,
    #[serde(default)]
    pub recurrent: bool,
    #[serde(default)]
    pub parameter_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct NodeSpec {
    pub node_id: String,
    #[serde(default)]
    pub cached_slice_bytes: u64,
    #[serde(default)]
    pub vram_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct NodePlacementSignal {
    pub node_id: String,
    #[serde(default)]
    pub cached_slice_bytes: u64,
    #[serde(default)]
    pub missing_artifact_bytes: u64,
    #[serde(default)]
    pub rtt_ms: Option<u32>,
    #[serde(default)]
    pub artifact_transfer_supported: bool,
    #[serde(default)]
    pub availability_score: u32,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct PlannerPolicy {
    pub allow_recurrent_state_transfer: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TopologyPlan {
    pub topology_id: String,
    pub model_id: String,
    #[serde(default)]
    pub family_id: Option<String>,
    pub stages: Vec<StagePlan>,
    #[serde(default)]
    pub boundaries: Vec<BoundaryPlan>,
    pub diagnostics: Vec<PlanDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StagePlan {
    pub stage_id: String,
    pub stage_index: u32,
    pub node_id: String,
    #[serde(default)]
    pub roles: Vec<StageRole>,
    pub layer_start: u32,
    pub layer_end: u32,
    pub layer_count: u32,
    pub parameter_bytes: u64,
    pub state_affinity: StateAffinity,
    pub migration_policy: MigrationPolicy,
    #[serde(default)]
    pub reason_codes: Vec<PlanReasonCode>,
    #[serde(default)]
    pub cached_slice_bytes: u64,
    #[serde(default)]
    pub missing_artifact_bytes: u64,
    #[serde(default)]
    pub rtt_ms: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageRole {
    Driver,
    Embedding,
    Intermediate,
    Readout,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct BoundaryPlan {
    pub producer_stage_index: u32,
    pub consumer_stage_index: u32,
    pub layer_boundary: u32,
    pub decision: BoundaryDecision,
    pub wire_dtype: WireDType,
    pub raw_activation_bytes_per_token: u64,
    pub wire_payload_bytes_per_token: u64,
    #[serde(default)]
    pub reason_codes: Vec<PlanReasonCode>,
    #[serde(default)]
    pub messages: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryDecision {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateAffinity {
    Stateless,
    AttentionKv,
    Recurrent,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationPolicy {
    FreelyMovable,
    CostedKv,
    StickyRecurrentOwner,
    RecurrentStateTransferAllowed,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PlanDiagnostic {
    pub severity: DiagnosticSeverity,
    pub code: PlanReasonCode,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReasonCode {
    ActivationOnlyBoundary,
    AttentionKvCosted,
    RecurrentOwnerSticky,
    RecurrentStateTransferAllowed,
    RecurrentStateTransferRejected,
    SharedKvRegionCut,
    TokenSidebandRequired,
    ActivationSidebandRequired,
    DefaultWireDtypeF16,
    Q8WireValidated,
    Q8WireRejected,
    ExactStateMobilityAccepted,
    ExactStateMobilityRejected,
    CacheLocalityPreferred,
    ArtifactTransferPenalty,
    NetworkPipelineCost,
    PeerAvailabilityPreferred,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct FamilyCapabilityRecord {
    pub family_id: String,
    pub layer_count: u32,
    pub activation_width: u32,
    pub default_wire_dtype: WireDType,
    pub q8_wire_validation: WireValidation,
    pub exact_state_mobility: ExactStateMobility,
    #[serde(default)]
    pub recurrent_ranges: Vec<LayerRange>,
    #[serde(default)]
    pub split_constraints: Vec<SplitConstraint>,
    #[serde(default)]
    pub sidebands: Vec<SidebandRequirement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireDType {
    F32,
    F16,
    Q8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireValidation {
    Untested,
    Validated,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExactStateMobility {
    Untested,
    Accepted,
    RejectedTooLarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
}

impl LayerRange {
    pub fn contains_layer(self, layer: u32) -> bool {
        self.start <= layer && layer < self.end
    }

    pub fn contains_boundary(self, boundary: u32) -> bool {
        self.start < boundary && boundary < self.end
    }

    pub fn intersects(self, start: u32, end: u32) -> bool {
        self.start < end && start < self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SplitConstraint {
    pub kind: SplitConstraintKind,
    pub range: LayerRange,
    #[serde(default)]
    pub forbidden_boundaries: Vec<u32>,
    pub reject_boundary_inside: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitConstraintKind {
    SharedKvProducerConsumer,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SidebandRequirement {
    pub kind: SidebandKind,
    pub first_required_layer: u32,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SidebandKind {
    TokenIds,
    Rwkv7VFirst,
    Gemma3nAltup,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ReviewedCapabilityRecord {
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub source_repo: Option<String>,
    #[serde(default)]
    pub source_revision: Option<String>,
    #[serde(default)]
    pub source_file: Option<String>,
    #[serde(default)]
    pub canonical_ref: Option<String>,
    #[serde(default)]
    pub distribution_id: Option<String>,
    #[serde(default)]
    pub selector: Option<String>,
    pub capability: FamilyCapabilityRecord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageRuntimeFamilyExpectation {
    pub llama_architecture: &'static str,
    pub family_id: &'static str,
    pub recurrent_or_hybrid: bool,
}

#[cfg(test)]
mod tests;
