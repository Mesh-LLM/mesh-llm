//! Run admitted paired cells and retain each cell's actual environment and cleanup.
use super::{
    admission, evidence_io, manifest_output, options, paired_execution, plan, preflight,
    trial_cell, trial_environment, trial_profile,
};
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::Path,
    time::{Duration, Instant},
};

pub(super) struct Execution {
    pub batch: paired_execution::Batch,
    pub environments: [Option<BTreeMap<String, trial_environment::Entry>>; 2],
    pub inheritance: [Option<trial_profile::Inheritance>; 2],
}

fn provenance(directory: &Path, trial: &trial_cell::Trial) -> DynResult<()> {
    evidence_io::publish(
        &directory.join("trial-provenance.json"),
        &json!({
            "schema_version": 1,
            "environment": trial.environment,
            "inheritance": trial.inheritance,
            "worker_status": trial.worker_status,
            "cleanup_complete": trial.cleanup_complete,
            "cleanup_forced": trial.cleanup_forced,
            "capture_complete": trial.capture_complete,
            "health_capture_complete": trial.health_capture_complete,
            "health_observation_error": trial.health_observation_error,
        }),
        evidence_io::RECEIPT_BYTES,
    )
}

pub(super) fn execute(
    command: &options::Command,
    metadata: &admission::Metadata,
    start: Instant,
    cancellation: &Cancellation,
) -> DynResult<Execution> {
    execute_with(
        command,
        metadata,
        start,
        cancellation,
        preflight::revalidate,
        trial_cell::execute,
    )
}

fn execute_with(
    command: &options::Command,
    metadata: &admission::Metadata,
    start: Instant,
    cancellation: &Cancellation,
    mut revalidate: impl FnMut(&admission::Metadata, &Path, Duration, &Cancellation) -> DynResult<()>,
    mut trial: impl FnMut(&trial_cell::Input<'_>, &Cancellation) -> DynResult<trial_cell::Trial>,
) -> DynResult<Execution> {
    if !metadata.runtime_packages_verified {
        return Err("benchmark execution requires native runtime package verification".into());
    }
    let entries = plan::build(&command.spec, &metadata.sides)?;
    let inherited = std::env::vars_os().collect::<BTreeMap<OsString, OsString>>();
    let mut environments = [None, None];
    let mut inheritance = [None, None];
    let deadline = Duration::from_secs(command.execution_secs);
    let batch = paired_execution::run(&entries, &metadata.sides, |side, entry| {
        if cancellation.is_cancelled() {
            return Err("benchmark matrix interrupted before next cell".into());
        }
        let index = usize::from(side.side_id == metadata.sides[1].side_id);
        let directory = command.output_dir.join(entry.log_stem(&side.side_id));
        let validation = command
            .output_dir
            .join(format!("{}-identity", entry.log_stem(&side.side_id)));
        std::fs::create_dir(&validation)?;
        revalidate(
            metadata,
            &validation,
            deadline.saturating_sub(start.elapsed()),
            cancellation,
        )?;
        let mut trial = trial(
            &trial_cell::Input {
                side,
                entry,
                binary_sha256: &metadata.binaries[index].sha256,
                model: &metadata.model,
                model_sha256: &metadata.source_model_sha256,
                native_runtime_root: &metadata.runtime_roots[index],
                directory: &directory,
                max_tokens: command.max_tokens,
                readiness: Duration::from_secs(command.readiness_secs),
                request: Duration::from_secs(command.request_secs),
                poll: Duration::from_millis(200),
                shutdown: Duration::from_secs(command.shutdown_secs),
                remaining: deadline.saturating_sub(start.elapsed()),
                inherited_profile: &inherited,
            },
            cancellation,
        )?;
        // Publication failure is an evidence failure, but preserve the observed row.
        if let Err(error) = provenance(&directory, &trial) {
            let prior = trial.outcome.error.take();
            trial.outcome.error = Some(format!(
                "{}trial provenance publication failed: {error}",
                prior.map_or_else(String::new, |value| format!("{value}; "))
            ));
        }
        environments[index] = Some(std::mem::take(&mut trial.environment));
        inheritance[index] = Some(std::mem::take(&mut trial.inheritance));
        Ok(trial.into_outcome())
    })?;
    Ok(Execution {
        batch,
        environments,
        inheritance,
    })
}

pub(super) fn publish(
    command: &options::Command,
    prepared: &preflight::Prepared,
    execution: &Execution,
    generated_at: &str,
) -> DynResult<BTreeMap<String, String>> {
    let context = manifest_output::Context {
        spec: &command.spec,
        model: prepared
            .metadata
            .model
            .to_str()
            .ok_or("benchmark model path is not UTF8")?,
        source_model_sha256: &prepared.metadata.source_model_sha256,
        attempt: command.attempt,
        generated_at,
        host: &prepared.metadata.host,
        thermal_state: &prepared.thermal_state,
    };
    let mut paths = BTreeMap::new();
    for index in 0..2 {
        let side = &prepared.metadata.sides[index];
        let mut document = manifest_output::build_normalized(
            &context,
            side,
            &prepared.metadata.binaries[index],
            execution.environments[index].as_ref(),
            execution.inheritance[index].as_ref(),
            &execution.batch,
            index,
        )?;
        document["model_metadata"] = prepared.metadata.model_metadata.clone();
        document["native_runtime_packages_verified"] = Value::Bool(true);
        let path = command
            .output_dir
            .join(format!("manifest-{}.json", side.side_id));
        let text = path
            .to_str()
            .ok_or("benchmark manifest path is not UTF8")?
            .to_owned();
        evidence_io::publish(&path, &document, evidence_io::MANIFEST_BYTES)?;
        paths.insert(side.side_id.clone(), text);
    }
    Ok(paths)
}

#[cfg(all(test, unix))]
#[path = "matrix_command_tests.rs"]
mod command_tests;
