#[path = "../src/ci_plan/plan_bytes.rs"]
pub(crate) mod plan_bytes;
mod ci_plan {
    pub(crate) use crate::plan_bytes;
}
#[path = "../src/repository/python_text.rs"]
#[expect(
    dead_code,
    reason = "The isolated parity target consumes only document separator semantics"
)]
pub(crate) mod python_text;
mod repository {
    pub(crate) use crate::python_text;
}
#[expect(
    dead_code,
    unused_imports,
    reason = "Parity exercises receipt aggregation, not writing or rendering"
)]
#[path = "../src/automation/canary_receipts/mod.rs"]
mod canary_receipts;

#[path = "migration_canary_receipts_parity/aggregate.rs"]
mod aggregate;
#[path = "migration_canary_receipts_parity/support.rs"]
mod support;
