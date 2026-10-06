//! Alternating old/new rounds; every arm is a separately owned mixed-cell child.
use super::{
    acceptance::Version,
    adaptive_identity as identity, mixed_cell, mixed_summary as summary, mixed_worker,
    mixed_workload::{self, Manifest, PromptRecord, Role, Shape},
    options, publish,
};
use crate::{
    command::DynResult,
    process::{self, Cancellation, Value as Arg},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u64,
    old: identity::Input,
    new: identity::Input,
    shape: Shape,
    split: bool,
    manifest: Option<Manifest>,
    request_timeout_secs: f64,
    worker_timeout_secs: u64,
    suppressed_token_ids: Vec<u32>,
    timeout_secs: u64,
    cell_timeout_secs: u64,
    startup_timeout_secs: u64,
}
fn matched(old: &identity::Input, new: &identity::Input) -> bool {
    old.native_profile == new.native_profile
        && old.model_sha256 == new.model_sha256
        && old.model_id == new.model_id
        && old.ctx_size == new.ctx_size
        && old.split_layer == new.split_layer
        && old.layer_end == new.layer_end
        && old.n_gpu_layers == new.n_gpu_layers
        && old.adaptive_target_ms == new.adaptive_target_ms
}
impl Input {
    fn validate(&self) -> DynResult<()> {
        self.old.validate()?;
        self.new.validate()?;
        self.shape.validate()?;
        if let Some(manifest) = &self.manifest {
            manifest.validate(&self.shape)?;
        }
        if self.schema_version != 1
            || self.old.version != Version::Old
            || self.new.version != Version::New
            || self.old.model != self.new.model
            || !matched(&self.old, &self.new)
            || !(1..=128).contains(&self.shape.rounds)
            || !self.request_timeout_secs.is_finite()
            || self.request_timeout_secs <= 0.0
            || self.request_timeout_secs > 86400.0
            || !(1..=86400).contains(&self.worker_timeout_secs)
            || self.suppressed_token_ids.len() > 256
            || !(13..=86400).contains(&self.timeout_secs)
            || !(10..=86400).contains(&self.cell_timeout_secs)
            || self.startup_timeout_secs == 0
            || self.startup_timeout_secs >= self.cell_timeout_secs
        {
            return Err("mixed matrix bounds or old/new model/config profile differ".into());
        }
        Ok(())
    }
}
pub(super) fn order(round: u64) -> [Version; 2] {
    if round % 2 == 1 {
        [Version::Old, Version::New]
    } else {
        [Version::New, Version::Old]
    }
}
fn label(version: Version) -> &'static str {
    if version == Version::Old {
        "old"
    } else {
        "new"
    }
}
fn clean(report: &process::ProcessReport) -> bool {
    report.outcome == process::Outcome::Exited
        && report
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code)
            == Some(0)
        && report.failure.is_none()
        && report.cleanup.complete
        && !report.cleanup.forced
        && !report.cleanup.graceful_signal_failed
        && report.cleanup.failure.is_none()
        && report.stdout.line_capture_complete
        && report.stderr.line_capture_complete
}
fn invoke(
    input: &mixed_cell::Input,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Value> {
    let bytes = serde_json::to_vec(input)?;
    let hash = identity::digest(&bytes);
    let request = directory.with_extension("input.json");
    identity::fresh(&request, &bytes)?;
    let execution = until
        .saturating_duration_since(Instant::now())
        .saturating_sub(Duration::from_secs(3));
    if execution < Duration::from_secs(input.timeout_secs) {
        return Err("mixed child cannot fit overall deadline with cleanup".into());
    }
    let report = process::supervise(
        &process::ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: vec![
                Arg::Public("automation".into()),
                Arg::Public("waiting-prefix".into()),
                Arg::Public("mixed-cell".into()),
                Arg::Public("--input".into()),
                Arg::Public(request.into_os_string()),
                Arg::Public("--output-directory".into()),
                Arg::Public(directory.as_os_str().to_owned()),
            ],
            cwd: directory.parent().ok_or("matrix parent absent")?.into(),
            environment: ["PATH", "SYSTEMROOT", "WINDIR"]
                .into_iter()
                .filter_map(|key| {
                    std::env::var_os(key).map(|value| (key.into(), Arg::Public(value)))
                })
                .collect(),
        },
        &process::Limits {
            execution,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        cancel,
        process::OutputFiles {
            stdout: Some(directory.with_extension("stdout.log")),
            stderr: Some(directory.with_extension("stderr.log")),
        },
    )?;
    let receipt: Value = serde_json::from_slice(&identity::bounded(
        &directory.join("cell.json"),
        16 * 1024 * 1024,
    )?)?;
    if !clean(&report)
        || receipt["schema_version"] != 1
        || receipt["request_sha256"] != hash
        || receipt["status"] != "mixed_cell_admitted"
    {
        return Err("mixed arm child or correlated receipt refused".into());
    }
    let arm: identity::Input = serde_json::from_value(receipt["identity"]["admitted"].clone())?;
    if !matched(&input.arm, &arm)
        || input.arm.binary_sha256 != arm.binary_sha256
        || input.arm.commit != arm.commit
        || input.arm.native_build_sha256 != arm.native_build_sha256
        || input.arm.version != arm.version
        || input.arm.round != arm.round
        || input.arm.model.canonicalize()? != arm.model
        || input.arm.binary.canonicalize()? != arm.binary
        || input.arm.native_build.canonicalize()? != arm.native_build
    {
        return Err("mixed arm receipt differs from assigned identity".into());
    }
    if receipt["shape"] != serde_json::to_value(&input.shape)? || receipt["split"] != input.split {
        return Err("mixed cell receipt profile differs from assignment".into());
    }
    let measured = &receipt["requests"];
    if measured["workload_sha256"] != input.worker.workload_sha256
        || measured["round"] != input.arm.round
        || measured["version"] != serde_json::to_value(input.arm.version)?
        || measured["model"] != input.arm.model_id
    {
        return Err("mixed cell receipt worker workload identity differs".into());
    }
    Ok(receipt)
}
fn worker(input: &Input, arm: &identity::Input, round: u64) -> DynResult<mixed_worker::Input> {
    let mut worker = mixed_worker::Input {
        schema_version: 1,
        round,
        version: arm.version,
        base_url: format!("http://127.0.0.1:{}/v1", arm.openai_port),
        model: arm.model_id.clone(),
        request_timeout_secs: input.request_timeout_secs,
        timeout_secs: input.worker_timeout_secs,
        readiness_timeout_secs: input.startup_timeout_secs,
        warmup: PromptRecord {
            family: "synthetic-warmup".into(),
            prompt: mixed_workload::stable_prompt(16, -1, Role::Prefill)?,
            provenance: serde_json::Map::new(),
        },
        requests: mixed_workload::requests(&input.shape, round, input.manifest.as_ref())?,
        suppressed_token_ids: input.suppressed_token_ids.clone(),
        manifest_metadata: input
            .manifest
            .as_ref()
            .map(|m| m.metadata.clone())
            .unwrap_or_default(),
        provenance: json!({"shape":input.shape,"split":input.split})
            .as_object()
            .unwrap()
            .clone(),
        workload_sha256: String::new(),
    };
    worker.workload_sha256 = worker.workload_sha()?;
    worker.validate()?;
    Ok(worker)
}
fn execute_with<F>(
    input: &Input,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
    mut launch: F,
) -> Value
where
    F: FnMut(&mixed_cell::Input, &Path, Instant, &Cancellation) -> DynResult<Value>,
{
    let mut cells = Vec::new();
    let mut cell_bytes = 0_usize;
    let result = (|| -> DynResult<Value> {
        input.validate()?;
        for round in 1..=input.shape.rounds {
            for version in order(round) {
                if cancel.is_cancelled() {
                    return Err("mixed matrix interrupted".into());
                }
                let budget = until
                    .saturating_duration_since(Instant::now())
                    .saturating_sub(Duration::from_secs(3))
                    .as_secs()
                    .min(input.cell_timeout_secs);
                if budget < 10 || input.startup_timeout_secs >= budget {
                    return Err("mixed matrix deadline exhausted before next arm".into());
                }
                let mut arm = if version == Version::Old {
                    input.old.clone()
                } else {
                    input.new.clone()
                };
                arm.round = round;
                let cell_input = mixed_cell::Input {
                    schema_version: 1,
                    arm: arm.clone(),
                    shape: input.shape.clone(),
                    split: input.split,
                    worker: worker(input, &arm, round)?,
                    timeout_secs: budget,
                    startup_timeout_secs: input.startup_timeout_secs,
                };
                let mut cell = launch(
                    &cell_input,
                    &directory.join(format!("round-{round}-{}", label(version))),
                    until,
                    cancel,
                )?;
                cell["round"] = json!(round);
                cell["version"] = json!(label(version));
                cell["worker_receipt"] = cell["requests"].clone();
                cell["requests"] = cell["worker_receipt"]["requests"].clone();
                cell_bytes = cell_bytes
                    .checked_add(serde_json::to_vec(&cell)?.len())
                    .ok_or("mixed receipt size overflow")?;
                if cell_bytes > 32 * 1024 * 1024 {
                    return Err("mixed complete cell receipts exceed 32MiB budget".into());
                }
                cells.push(cell);
            }
        }
        if cancel.is_cancelled() || Instant::now() >= until {
            return Err("mixed matrix cancelled or expired before comparison".into());
        }
        let comparison = summary::compare(&cells, input.shape.rounds)?;
        if cancel.is_cancelled() || Instant::now() >= until {
            return Err("mixed deadline expired during bounded comparison".into());
        }
        Ok(comparison)
    })();
    let mut output = json!({"schema_version":1,"metadata":{"old":input.old,"new":input.new,"shape":input.shape,"split":input.split,"manifest":input.manifest,"discarded_calibration_requests_per_cell":1,"kv_cache":"original-mixed-default-no-explicit-disable","native_profile":"standalone-static-skippy-server","source_binding":"declared-commit-and-native-build-tree-not-binary-build-attestation"},"cells":cells,"error":null});
    match result {
        Ok(value) => {
            output["comparison"] = value;
            if output["comparison"]["output_parity"]["exact_matches"]
                != output["comparison"]["output_parity"]["comparable_requests"]
            {
                output["error"] = json!("mixed output parity mismatch");
            }
        }
        Err(e) => output["error"] = json!(e.to_string().chars().take(1024).collect::<String>()),
    };
    output
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(
        args,
        &["--input", "--output-directory"],
        &["--input", "--output-directory"],
    )?;
    let input: Input = serde_json::from_slice(&identity::bounded(
        Path::new(opts["--input"]),
        16 * 1024 * 1024,
    )?)?;
    input.validate()?;
    let directory = std::path::absolute(opts["--output-directory"])?;
    std::fs::create_dir(&directory)?;
    let directory: PathBuf = directory.canonicalize()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let output = execute_with(
        &input,
        &directory,
        Instant::now() + Duration::from_secs(input.timeout_secs),
        &interrupt.cancellation(),
        invoke,
    );
    let publication = (|| -> DynResult<()> {
        publish(
            &directory.join("comparison.json"),
            &serde_json::to_vec_pretty(&output)?,
        )?;
        let report = if output["comparison"].is_object() {
            summary::render(&output["comparison"])
        } else {
            format!("Mixed comparison unavailable: {}\n", output["error"])
        };
        publish(&directory.join("report.md"), report.as_bytes())
    })();
    let finished = interrupt.finish();
    publication?;
    finished?;
    if !output["error"].is_null() {
        return Err("mixed matrix failed; partial comparison retained".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "mixed_matrix_tests.rs"]
mod tests;
