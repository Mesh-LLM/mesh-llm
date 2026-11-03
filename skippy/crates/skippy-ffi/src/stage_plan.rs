//! Ergonomic aliases for the generated stage-plan ABI types.

use crate::bindings;

pub const STAGE_PLANNER_CONFIG_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_STAGE_PLANNER_CONFIG_V1_ABI_VERSION;
pub const STAGE_PLANNER_TENSOR_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_STAGE_PLANNER_TENSOR_V1_ABI_VERSION;
pub const STAGE_PLANNER_PROFILE_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_STAGE_PLANNER_PROFILE_V1_ABI_VERSION;
pub const STAGE_PLAN_DESC_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_STAGE_PLAN_DESC_V1_ABI_VERSION;
pub const STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION;
pub const STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION;
pub const STAGE_PLAN_STATE_DESC_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_STAGE_PLAN_STATE_DESC_V1_ABI_VERSION;
pub const STAGE_PLAN_MAX_DIMS: usize = bindings::skippy::SKIPPY_STAGE_PLAN_MAX_DIMS as usize;

pub type StagePlanner = bindings::skippy::skippy_stage_planner;
pub type StagePlan = bindings::skippy::skippy_stage_plan;
pub type StagePlanStringRefV1 = bindings::skippy::skippy_stage_plan_string_ref_v1;
pub type StagePlannerTensorV1 = bindings::skippy::skippy_stage_planner_tensor_v1;
pub type StagePlannerProfileV1 = bindings::skippy::skippy_stage_planner_profile_v1;
pub type StagePlannerConfigV1 = bindings::skippy::skippy_stage_planner_config_v1;
pub type StagePlanValueKind = bindings::skippy::skippy_stage_plan_value_kind;
pub type StagePlanStateKind = bindings::skippy::skippy_stage_plan_state_kind;
pub type StagePlanStateAccess = bindings::skippy::skippy_stage_plan_state_access;
pub type StagePlanStateResidency = bindings::skippy::skippy_stage_plan_state_residency;
pub type StagePlanDescV1 = bindings::skippy::skippy_stage_plan_desc_v1;
pub type StagePlanProfileDescV1 = bindings::skippy::skippy_stage_plan_profile_desc_v1;
pub type StagePlanValueDescV1 = bindings::skippy::skippy_stage_plan_value_desc_v1;
pub type StagePlanStateDescV1 = bindings::skippy::skippy_stage_plan_state_desc_v1;
