//! Native event benchmark matrix with owned worker admission and retained lifecycle.
mod health_log;
mod http_measurement;
mod manifest_output;
mod measurement_worker;
mod options;
mod paired_execution;
mod plan;
mod stream_metrics;
mod trial_contract;
mod trial_environment;

mod admission;
mod command;
mod evidence_io;
mod identity_worker;
mod lifecycle_budget;
mod matrix;
mod preflight;
mod probes;
mod trial_cell;
mod trial_owner;
mod trial_profile;
mod trial_receipt;
mod worker_frontends;
pub(crate) use command::run;
