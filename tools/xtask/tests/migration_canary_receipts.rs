#[path = "../src/ci_plan/plan_bytes.rs"]
pub(crate) mod plan_bytes;
mod ci_plan {
    pub(crate) use crate::plan_bytes;
}
#[path = "../src/repository/text.rs"]
#[expect(
    dead_code,
    reason = "The isolated target consumes only document separator semantics"
)]
pub(crate) mod text;
mod repository {
    pub(crate) use crate::text;
}
#[expect(
    dead_code,
    unused_imports,
    reason = "Package admission is exercised by the dedicated parity target"
)]
#[path = "../src/automation/canary_receipts/mod.rs"]
mod canary_receipts;

#[path = "migration_canary_receipts/aggregation.rs"]
mod aggregation;
#[path = "migration_canary_receipts/duplicate_keys.rs"]
mod duplicate_keys;
#[path = "migration_canary_receipts/inputs.rs"]
mod inputs;
#[path = "migration_canary_receipts/limits.rs"]
mod limits;
#[path = "migration_canary_receipts/results.rs"]
mod results;
#[path = "migration_canary_receipts/separators.rs"]
mod separators;
#[path = "migration_canary_receipts/support.rs"]
mod support;
#[path = "migration_canary_receipts/top_level_duplicates.rs"]
mod top_level_duplicates;
#[path = "migration_canary_receipts/writer.rs"]
mod writer;

#[path = "migration_canary_receipts/handoff_cli.rs"]
mod handoff_cli;
