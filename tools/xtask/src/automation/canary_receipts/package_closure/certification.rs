//! One package-bound worker certification; retained battery owns actual native lanes.
#[path = "certification/guard.rs"]
mod guard;
#[path = "certification/host_lock.rs"]
mod host_lock;
#[path = "certification/observation.rs"]
mod observation;
#[cfg(test)]
#[path = "certification/tests.rs"]
mod tests;
use super::{process, producer_receipt::Context, restoring};
use crate::{
    automation::{
        canary_receipts::{Digest, PackageVerification, verify_package},
        canary_source_plan,
    },
    command::DynResult,
    process::{OutputFiles, ProcessSpec, Value},
};
use serde::Deserialize;
use serde_json::{Value as Json, json};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    context: Context,
    root: PathBuf,
    package: PathBuf,
    identity_sha256: Digest,
    shard_index: u64,
    memory_tier: String,
    evidence: PathBuf,
    #[serde(default = "budget")]
    max_seconds: u64,
}
fn budget() -> u64 {
    43200
}

pub(super) fn execute(input: &Input) -> DynResult<Json> {
    if !input.root.is_absolute()
        || !input.package.is_absolute()
        || !input.evidence.is_absolute()
        || !(1..=43200).contains(&input.max_seconds)
    {
        return Err("invalid certify paths/budget".into());
    }
    let root = input.root.canonicalize()?;
    if input.evidence.starts_with(&root) {
        return Err("certify evidence must be outside consumer source".into());
    }
    fs::create_dir_all(&input.evidence)?;
    let evidence = input.evidence.canonicalize()?;
    if evidence.starts_with(&root) {
        return Err("certify evidence alias escapes role separation".into());
    }
    let mut report = json!({"family":"unknown","reserve_percent":10,"status":"failed"});
    let result = execute_bound(input, &root, &evidence, &mut report);
    if let Err(error) = &result {
        report["status"] = json!("failed");
        report["error"] = json!(error.to_string());
    }
    fs::write(
        evidence.join("memory-admission.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    result
}

fn execute_bound(
    input: &Input,
    root: &std::path::Path,
    evidence: &std::path::Path,
    report: &mut Json,
) -> DynResult<Json> {
    let started = Instant::now();
    let deadline = started
        .checked_add(Duration::from_secs(input.max_seconds))
        .ok_or("certification deadline overflow")?;
    input.context.validate()?;
    verify_package(
        &input.package,
        PackageVerification {
            expected_identity_sha256: input.identity_sha256.clone(),
            current_run_id: input.context.run_id.clone(),
            current_run_attempt: input.context.run_attempt.clone(),
            controller_revision: Some(input.context.controller_revision.clone()),
            selected_source: input.context.selected_source.clone(),
        },
    )?;
    let plan = fs::read(input.package.join("plan.json"))?;
    let matrix = serde_json::to_value(canary_source_plan::placement::project(&plan)?)?;
    let rows = matrix["include"]
        .as_array()
        .ok_or("invalid certification scheduling matrix")?
        .iter()
        .filter(|row| row["shard_index"].as_u64() == Some(input.shard_index))
        .collect::<Vec<_>>();
    if rows.len() != 1 {
        return Err("unplanned family shard".into());
    }
    let row = rows[0];
    if row["memory_tier"].as_str() != Some(input.memory_tier.as_str()) {
        return Err("scheduled memory tier differs from verified family estimate".into());
    }
    let family = row["families"].as_str().ok_or("missing family identity")?;
    report["family"] = json!(family);
    for key in [
        "resident_model_bytes",
        "runtime_allowance_bytes",
        "estimated_peak_bytes",
        "minimum_runner_memory_gib",
        "memory_tier",
    ] {
        if let Some(value) = row.get(key) {
            report[key] = value.clone();
        }
    }
    let request = restoring::Input {
        context: input.context.clone(),
        root: root.to_owned(),
        package: input.package.clone(),
        identity_sha256: input.identity_sha256.clone(),
    };
    let capture = evidence.join(format!("before-{}.diff", std::process::id()));
    restoring::verify_consumer(&request, &capture)?;
    let battery = root
        .join("scripts/skippy-family-battery.sh")
        .canonicalize()?;
    if !battery.starts_with(root) || !fs::symlink_metadata(&battery)?.is_file() {
        return Err("selected battery escapes consumer source".into());
    }
    let (_lock, contended, wait) = host_lock::acquire(deadline)?;
    report["host_lock_contended"] = json!(contended);
    report["host_lock_wait_seconds"] = json!(wait.as_secs_f64());
    let initial = observation::observe(root, &process::cancellation())?;
    let peak = row["estimated_peak_bytes"]
        .as_u64()
        .ok_or("missing family peak memory")?;
    let reserve = observation::admission(initial, peak)?;
    report["physical_bytes"] = json!(initial.total);
    report["initial_available_bytes"] = json!(initial.available);
    report["reserve_bytes"] = json!(reserve);
    report["minimum_available_bytes"] = json!(initial.available);
    let spec = ProcessSpec {
        executable: "/usr/bin/arch".into(),
        arguments: [
            "-arm64".into(),
            battery.into(),
            "--skip-build".into(),
            "--plan".into(),
            input.package.canonicalize()?.join("plan.json").into(),
            "--shard-index".into(),
            input.shard_index.to_string().into(),
        ]
        .into_iter()
        .map(Value::Public)
        .collect(),
        cwd: root.to_owned(),
        environment: environment(),
    };
    let observed = guard::run(
        &spec,
        deadline,
        &process::cancellation(),
        initial,
        reserve,
        OutputFiles {
            stdout: Some(evidence.join("certify.stdout.log")),
            stderr: Some(evidence.join("certify.stderr.log")),
        },
    )?;
    report["minimum_available_bytes"] = json!(observed.minimum);
    report["exit_code"] = json!(observed.process.status.and_then(|status| status.code()));
    report["supervisor_outcome"] = json!(format!("{:?}", observed.process.outcome));
    report["cleanup_complete"] = json!(observed.process.cleanup.complete);
    if let Some(error) = observed.guard_error {
        return Err(error.into());
    }
    if !observed.process.success()
        || observed.process.stdout.truncated
        || observed.process.stderr.truncated
    {
        return Err(format!(
            "family certification failed: {:?}, {:?}; cleanup={:?}",
            observed.process.outcome, observed.process.status, observed.process.cleanup
        )
        .into());
    }
    restoring::verify_consumer(
        &request,
        &evidence.join(format!("after-{}.diff", std::process::id())),
    )?;
    process::check()?;
    report["status"] = json!("passed");
    Ok(
        json!({"status":"passed","family":family,"shard_index":input.shard_index,"identity_sha256":input.identity_sha256}),
    )
}
fn environment() -> BTreeMap<std::ffi::OsString, Value> {
    std::env::vars_os()
        .filter(|(key, _)| {
            !["GH_TOKEN", "GITHUB_TOKEN", "CANARY_REPAIR_TOKEN"]
                .iter()
                .any(|name| key == name)
        })
        .map(|(key, value)| {
            let name = key.to_string_lossy();
            let secret = !value.is_empty()
                && (name.ends_with("_TOKEN")
                    || name.ends_with("_KEY")
                    || name.ends_with("_SECRET")
                    || name.ends_with("_PASSWORD"));
            (
                key,
                if secret {
                    Value::Secret(value)
                } else {
                    Value::Public(value)
                },
            )
        })
        .collect()
}
