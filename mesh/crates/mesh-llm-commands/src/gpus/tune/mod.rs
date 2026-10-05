// Module root for the `tune` submodule. Each child module is a real
// submodule so visibility and the flat `crate::gpus::tune::*` re-exports
// below can be controlled explicitly. Child modules add `use super::*;`
// so shared symbol lookups continue to resolve.
//
// Items in this module are reached by lib code, command handlers,
// `tune_hardware`, the embedded `benchmark` modules, and test scaffolding.
// `dead_code` is suppressed because callers live across several targets
// (lib, bins, test scaffolding) and clippy cannot unify them all.
#![allow(dead_code)]

pub mod benchmark;
pub(crate) mod benchmark_progress;
pub(crate) mod benchmark_selection;
pub(crate) mod matrix;
pub(crate) mod metadata;
pub(crate) mod output_report;
pub(crate) mod output_types;
pub(crate) mod output_values;
pub(crate) mod types;
pub(crate) use benchmark::*;
pub(crate) mod output_emit;
pub(crate) mod output_launch;
pub(crate) mod output_render;
pub(crate) mod planning;
pub(crate) mod recommendation;
pub(crate) mod recommendation_existing;
pub(crate) mod recommendation_reports;
pub(crate) mod recommendation_writes;

// Re-exports preserve the previous flat `crate::gpus::tune::*` API consumed
// by `tune_hardware`, the embedded `benchmark` modules, and the command
// handlers. Without these re-exports, callers like
// `crate::gpus::tune::TuneDiagnostic` would need deep `module::type` paths.
pub(crate) use benchmark_progress::*;
pub(crate) use benchmark_selection::*;
pub(crate) use metadata::*;
pub(crate) use output_emit::*;
pub(crate) use output_launch::*;
pub(crate) use output_render::*;
pub(crate) use output_report::*;
pub(crate) use output_types::*;
pub(crate) use output_values::*;
pub(crate) use planning::*;
pub(crate) use recommendation::*;
pub(crate) use recommendation_existing::*;
pub(crate) use recommendation_reports::*;
pub(crate) use recommendation_writes::*;
pub(crate) use types::*;

#[cfg(test)]
pub(crate) mod apply_collision_tests;
#[cfg(test)]
pub(crate) mod apply_test_support;
#[cfg(test)]
pub(crate) mod apply_write_tests;
#[cfg(test)]
pub(crate) mod metadata_tests;
#[cfg(test)]
pub(crate) mod output_tests;
#[cfg(test)]
pub(crate) mod recommendation_defaults_tests;
#[cfg(test)]
pub(crate) mod recommendation_failure_tests;
#[cfg(test)]
pub(crate) mod recommendation_tests;
#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
pub(crate) use apply_test_support::{appended_target, configured_target, write_local_gguf_file};
#[cfg(test)]
pub(crate) use recommendation_tests::{
    assert_applied_batch, assert_applied_context, assert_applied_fit_target,
    assert_applied_flash_attention, assert_applied_gpu_layers, assert_applied_kv,
    assert_applied_ubatch, assert_preserved, gib, gpu_hardware, recommendation_target,
    sample_metadata, status_for, survey_with_gpu,
};
#[cfg(test)]
pub(crate) use tests::sample_target;
