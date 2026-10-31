#![cfg(feature = "automation-contracts")]
pub(crate) use ci_controller::plan::plan_bytes;
mod ci_plan {
    pub(crate) use crate::plan_bytes;
}
mod repository {
    pub(crate) use ci_controller::repository::text;
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
