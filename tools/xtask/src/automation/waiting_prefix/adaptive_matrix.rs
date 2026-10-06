//! Alternating old/new rounds; every arm is a separately owned adaptive-cell child.
use super::{
    acceptance::Version, adaptive_cell, adaptive_identity as identity, adaptive_summary as summary,
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
    worker: Value,
    rounds: u64,
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
        if self.schema_version != 1
            || self.old.version != Version::Old
            || self.new.version != Version::New
            || self.old.model != self.new.model
            || !matched(&self.old, &self.new)
            || !(1..=128).contains(&self.rounds)
            || !(13..=86400).contains(&self.timeout_secs)
            || !(10..=86400).contains(&self.cell_timeout_secs)
            || self.startup_timeout_secs == 0
            || self.startup_timeout_secs >= self.cell_timeout_secs
        {
            return Err("adaptive matrix bounds or old/new model/config profile differ".into());
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
    input: &adaptive_cell::Input,
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
        return Err("adaptive child cannot fit overall deadline with cleanup".into());
    }
    let report = process::supervise(
        &process::ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: vec![
                Arg::Public("automation".into()),
                Arg::Public("waiting-prefix".into()),
                Arg::Public("adaptive-cell".into()),
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
        || receipt["status"] != "adaptive_cell_admitted"
    {
        return Err("adaptive arm child or correlated receipt refused".into());
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
        return Err("adaptive arm receipt differs from assigned identity".into());
    }
    Ok(receipt)
}
fn execute_with<F>(
    input: &Input,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
    mut launch: F,
) -> Value
where
    F: FnMut(&adaptive_cell::Input, &Path, Instant, &Cancellation) -> DynResult<Value>,
{
    let mut cells = Vec::new();
    let mut cell_bytes = 0_usize;
    let result = (|| -> DynResult<Value> {
        input.validate()?;
        for round in 1..=input.rounds {
            for version in order(round) {
                if cancel.is_cancelled() {
                    return Err("adaptive matrix interrupted".into());
                }
                let budget = until
                    .saturating_duration_since(Instant::now())
                    .saturating_sub(Duration::from_secs(3))
                    .as_secs()
                    .min(input.cell_timeout_secs);
                if budget < 10 || input.startup_timeout_secs >= budget {
                    return Err("adaptive matrix deadline exhausted before next arm".into());
                }
                let mut arm = if version == Version::Old {
                    input.old.clone()
                } else {
                    input.new.clone()
                };
                arm.round = round;
                let cell_input = adaptive_cell::Input {
                    schema_version: 1,
                    arm,
                    worker: input.worker.clone(),
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
                cell["summary"] = summary::summarize(&cell)?;
                cell_bytes = cell_bytes
                    .checked_add(serde_json::to_vec(&cell)?.len())
                    .ok_or("adaptive receipt size overflow")?;
                if cell_bytes > 32 * 1024 * 1024 {
                    return Err("adaptive complete cell receipts exceed 32MiB budget".into());
                }
                cells.push(cell);
            }
        }
        if cancel.is_cancelled() || Instant::now() >= until {
            return Err("adaptive matrix cancelled or expired before comparison".into());
        }
        let comparison = summary::compare(&cells, input.rounds)?;
        if cancel.is_cancelled() || Instant::now() >= until {
            return Err("adaptive deadline expired during bounded comparison".into());
        }
        Ok(comparison)
    })();
    let mut output = json!({"schema_version":1,"metadata":{"old":input.old,"new":input.new,"rounds":input.rounds,"discarded_calibration_requests_per_cell":1,"worker":input.worker,"cache":"disabled","native_profile":"standalone-static-skippy-server","source_binding":"declared-commit-and-native-build-tree-not-binary-build-attestation"},"cells":cells,"error":null});
    match result {
        Ok(value) => {
            output["comparison"] = value;
            if output["comparison"]["output_parity"]["exact_matches"]
                != output["comparison"]["output_parity"]["comparable_requests"]
            {
                output["error"] = json!("adaptive output parity mismatch");
            }
        }
        Err(e) => output["error"] = json!(e.to_string().chars().take(1024).collect::<String>()),
    };
    output
}
/// Complete the signal/deadline boundary before publishing admission. Observed
/// cells/comparison remain available on failure and are explicitly unadmitted.
fn finalize(
    output: &mut Value,
    finished: DynResult<()>,
    cancellation: &Cancellation,
    deadline: Instant,
) -> DynResult<()> {
    let terminal = if cancellation.is_cancelled() {
        Some("adaptive terminal cancellation")
    } else if finished.is_err() {
        Some("adaptive interrupt finalization failed")
    } else if Instant::now() >= deadline {
        Some("adaptive terminal deadline exhausted")
    } else {
        None
    };
    output["terminal_error"] = terminal.map_or(Value::Null, |reason| json!(reason));
    if output["error"].is_null()
        && let Some(reason) = terminal
    {
        output["error"] = json!(reason);
    }
    let admitted = output["error"].is_null() && output["comparison"].is_object();
    output["comparison_admitted"] = json!(admitted);
    output["status"] = json!(if admitted {
        "adaptive_matrix_admitted"
    } else {
        "adaptive_matrix_failed"
    });
    if admitted {
        Ok(())
    } else {
        Err("adaptive matrix failed; partial observations retained without admission".into())
    }
}
fn rendered(output: &Value) -> String {
    let heading = if output["comparison_admitted"] == true {
        "Adaptive comparison admitted.\n\n".to_owned()
    } else {
        format!(
            "Adaptive comparison NOT admitted: {}. Retained values are observations.\n\n",
            output["error"]
        )
    };
    if output["comparison"].is_object() {
        heading + &summary::render(&output["comparison"])
    } else {
        heading + "Adaptive comparison unavailable.\n"
    }
}
fn prepare_input(input: &mut Input) -> DynResult<()> {
    input.validate()?;
    let worker = input
        .worker
        .as_object_mut()
        .ok_or("adaptive worker must be an object")?;
    worker.insert("round".into(), json!(1));
    worker.insert("version".into(), json!("old"));
    worker.insert("model".into(), json!(input.old.model_id));
    worker.insert(
        "base_url".into(),
        json!(format!("http://127.0.0.1:{}/v1", input.old.openai_port)),
    );
    worker.insert(
        "readiness_timeout_secs".into(),
        json!(input.startup_timeout_secs),
    );
    super::sequential_cell::bind_manifest(&mut input.worker)?;
    let typed: super::sequential_cell::Input = serde_json::from_value(input.worker.clone())?;
    typed.validate()
}
pub(super) fn prepare(args: &[String]) -> DynResult<()> {
    let opts = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let mut input: Input = serde_json::from_slice(&identity::bounded(
        Path::new(opts["--input"]),
        16 * 1024 * 1024,
    )?)?;
    prepare_input(&mut input)?;
    // Preserve the closed original fields; only the existing worker projection changes.
    let output = json!({"schema_version":input.schema_version,"old":input.old,"new":input.new,"worker":input.worker,"rounds":input.rounds,"timeout_secs":input.timeout_secs,"cell_timeout_secs":input.cell_timeout_secs,"startup_timeout_secs":input.startup_timeout_secs});
    identity::fresh(
        Path::new(opts["--output"]),
        &serde_json::to_vec_pretty(&output)?,
    )
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
    let cancellation = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut output = execute_with(&input, &directory, deadline, &cancellation, invoke);
    let finished = interrupt.finish().map_err(Into::into);
    let terminal = finalize(&mut output, finished, &cancellation, deadline);
    let publication = (|| -> DynResult<()> {
        publish(
            &directory.join("comparison.json"),
            &serde_json::to_vec_pretty(&output)?,
        )?;
        let report = rendered(&output);
        publish(&directory.join("report.md"), report.as_bytes())
    })();
    publication?;
    terminal
}
#[cfg(test)]
#[path = "adaptive_matrix_tests.rs"]
mod tests;
