//! Bounded local candidate coordinator; the retained harness owns certification.
#[path = "parity_toolkits.rs"]
mod toolkits;
use super::parity_local_plan;
use crate::{
    command::DynResult,
    process::{self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Argument},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Write as _,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    plan: parity_local_plan::Input,
    evidence: PathBuf,
    harness_sha256: String,
    max_seconds: u64,
    candidate_seconds: u64,
    stop_on_failure: bool,
    stage_build_dir: Option<PathBuf>,
    path: String,
    #[serde(default)]
    toolkit_dirs: toolkits::Input,
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    let [flag, path] = args else {
        return Err("parity-local-run requires --input PATH".into());
    };
    if flag != "--input" {
        return Err("parity-local-run requires --input PATH".into());
    }
    let bytes = parity_local_plan::document(Path::new(path))?;
    let input: Input = serde_json::from_slice(&bytes)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = execute(&input, &interrupt.cancellation());
    let finish = interrupt.finish();
    let measured = result?;
    let output = measured.publish(finish.map_err(Into::into))?;
    crate::repository::check_report::CheckReport::success(format!(
        "{}\n",
        serde_json::to_string_pretty(&output)?
    ))
    .emit()
}
fn digest(path: &Path) -> DynResult<String> {
    if !fs::symlink_metadata(path)?.is_file() {
        return Err("harness/receipt must be regular".into());
    }
    Ok(crate::product::digest::file_sha256(path).map_err(|error| error.error)?)
}
fn write_new(path: &Path, value: &Value) -> DynResult<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.sync_all()?;
    Ok(())
}
fn prepare(
    input: &Input,
    deadline: Instant,
    cancellation: &process::Cancellation,
) -> DynResult<(PathBuf, PathBuf, Value)> {
    if !input.evidence.is_absolute()
        || !(4..=86400).contains(&input.max_seconds)
        || !(1..=43200).contains(&input.candidate_seconds)
        || input.path.is_empty()
        || input.path.contains(['\0', '\n', '\r'])
    {
        return Err("invalid local execution paths/budgets".into());
    }
    let root = input.plan.source_root.canonicalize()?;
    let harness = root.join("scripts/family-certify.sh");
    if checked_digest(&harness, deadline, cancellation)? != input.harness_sha256 {
        return Err("local harness differs from declared source digest".into());
    }
    if input.evidence.starts_with(root.join("scripts")) {
        return Err("evidence must not replace harness source".into());
    }
    if input
        .stage_build_dir
        .as_ref()
        .is_some_and(|p| !p.is_absolute() || !p.is_dir())
    {
        return Err("stage build directory must exist and be absolute".into());
    }
    let plan =
        parity_local_plan::plan_with_guard(&input.plan, &mut || guard(deadline, cancellation))?;
    if plan["invocations"]
        .as_array()
        .ok_or("invocations")?
        .is_empty()
    {
        return Err("local certification selected no executable candidates".into());
    }
    let parent = input.evidence.parent().ok_or("evidence parent")?;
    if !fs::symlink_metadata(parent)?.is_dir() {
        return Err("evidence parent must be regular directory".into());
    }
    // A fresh owned root makes every coordinator output new; never resume by overwrite.
    fs::create_dir(&input.evidence)?;
    let evidence = input.evidence.canonicalize()?;
    write_new(&evidence.join("plan.json"), &plan)?;
    Ok((harness, evidence, plan))
}
// Only these reviewed scalar build/device settings cross the private child boundary.
// Toolchain/library/build-root path aliases require separately typed custody.
const PROFILE_NAMES: &[&str] = &[
    "LLAMA_STAGE_BACKEND",
    "SKIPPY_LLAMA_BACKEND",
    "LLAMA_STAGE_LINK_MODE",
    "SKIPPY_LLAMA_LINK_MODE",
    "LLAMA_STAGE_CUDA_ARCHITECTURES",
    "SKIPPY_CUDA_ARCHITECTURES",
    "LLAMA_STAGE_AMDGPU_TARGETS",
    "SKIPPY_AMDGPU_TARGETS",
    "LLAMA_STAGE_GGML_NATIVE",
    "SKIPPY_GGML_NATIVE",
    "GGML_CUDA_NO_VMM",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
];
fn environment(
    input: &Input,
    toolkits: &toolkits::Admitted,
) -> DynResult<(BTreeMap<std::ffi::OsString, Argument>, Value)> {
    let mut env = BTreeMap::new();
    for name in [
        "SystemRoot",
        "WINDIR",
        "TMP",
        "TEMP",
        "HOME",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "CC",
        "CXX",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        "CMAKE_GENERATOR",
    ] {
        if let Some(value) = std::env::var_os(name) {
            env.insert(name.into(), Argument::Public(value));
        }
    }
    let mut profile = serde_json::Map::new();
    for name in PROFILE_NAMES {
        if let Some(value) = std::env::var_os(name) {
            let text = value
                .to_str()
                .ok_or("local build/device setting must be UTF-8")?;
            if text.len() > 4096 || text.contains(['\0', '\n', '\r']) {
                return Err("invalid local build/device setting".into());
            }
            profile.insert((*name).into(), json!(text));
            env.insert((*name).into(), Argument::Public(value));
        }
    }
    for (key, value) in [("GIT_MASTER", "1"), ("GIT_OPTIONAL_LOCKS", "0")] {
        env.insert(key.into(), Argument::Public(value.into()));
    }
    env.insert("PATH".into(), Argument::Public(input.path.clone().into()));
    if let Some(path) = &input.stage_build_dir {
        env.insert(
            "LLAMA_STAGE_BUILD_DIR".into(),
            Argument::Public(path.clone().into()),
        );
    }
    // Retained harness native control helpers use this exact running controller.
    env.insert(
        "MESH_LLM_AUTOMATION_BIN".into(),
        Argument::Public(std::env::current_exe()?.into()),
    );
    env.extend(toolkits.environment.clone());
    let snapshot = json!({"scope":"effective_local_nonsecret_build_device_settings_not_build_custody","inherited_settings":profile,"path":input.path,"stage_build_dir":input.stage_build_dir,"git_master":"1","git_optional_locks":"0","toolkit_observations":toolkits.observation});
    Ok((env, snapshot))
}
struct Execution<'a> {
    deadline: Instant,
    cancellation: &'a process::Cancellation,
    environment: &'a BTreeMap<std::ffi::OsString, Argument>,
    toolkits: &'a toolkits::Admitted,
}
fn execute(input: &Input, cancellation: &process::Cancellation) -> DynResult<Measured> {
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(input.max_seconds))
        .ok_or("deadline overflow")?;
    let toolkits = toolkits::admit(&input.toolkit_dirs, &mut || guard(deadline, cancellation))?;
    let (harness, evidence, plan) = prepare(input, deadline, cancellation)?;
    let (environment, effective_profile) = environment(input, &toolkits)?;
    write_new(&evidence.join("effective-profile.json"), &effective_profile)?;
    let execution = Execution {
        deadline,
        cancellation,
        environment: &environment,
        toolkits: &toolkits,
    };
    let mut receipts = Vec::new();
    let mut failed = false;
    for (index, invocation) in plan["invocations"]
        .as_array()
        .ok_or("invocations")?
        .iter()
        .enumerate()
    {
        if cancellation.is_cancelled() {
            failed = true;
            break;
        }
        let directory = evidence.join(format!("candidate-{index:04}"));
        fs::create_dir(&directory)?;
        let result = one(input, &harness, &plan, invocation, &directory, &execution);
        let receipt = match result {
            Ok(receipt) => receipt,
            Err(error) => {
                json!({"index":index,"status":"refused","error":error.to_string(),"invocation":invocation})
            }
        };
        let success = receipt["status"] == "process_completed_manifest_bound";
        write_new(&directory.join("receipt.json"), &receipt)?;
        receipts.push(receipt);
        failed |= !success;
        if (!success && input.stop_on_failure)
            || cancellation.is_cancelled()
            || Instant::now() >= deadline
        {
            break;
        }
    }
    let output = json!({"schema_version":1,"scope":"local_harness_execution_observation_not_family_promotion","status":if cancellation.is_cancelled(){"cancelled"}else if Instant::now()>=deadline{"deadline"}else if failed{"failed"}else{"completed"},"planned_candidates":plan["invocations"].as_array().ok_or("invocations")?.len(),"completed_receipts":receipts.len(),"harness_sha256":input.harness_sha256,"effective_profile":effective_profile,"receipts":receipts});
    let mut checkpoint = output.clone();
    checkpoint["observed_status"] = output["status"].clone();
    checkpoint["status"] = "observed_pending_final_admission".into();
    write_new(&evidence.join("observed-receipt.json"), &checkpoint)?;
    Ok(Measured {
        evidence,
        output,
        deadline,
        cancellation: cancellation.clone(),
    })
}
fn one(
    input: &Input,
    harness: &Path,
    plan: &Value,
    invocation: &Value,
    directory: &Path,
    execution: &Execution<'_>,
) -> DynResult<Value> {
    let Execution {
        deadline,
        cancellation,
        environment,
        toolkits,
    } = *execution;
    if toolkits::admit(&input.toolkit_dirs, &mut || guard(deadline, cancellation))?.observation
        != toolkits.observation
    {
        return Err("toolkit directory observation changed before harness launch".into());
    }
    // Re-admit bytes/shape immediately before every launch; no saved plan trust.
    if parity_local_plan::plan_with_guard(&input.plan, &mut || guard(deadline, cancellation))?
        != *plan
        || checked_digest(harness, deadline, cancellation)? != input.harness_sha256
    {
        return Err("local plan/harness changed before launch".into());
    }
    let budget = deadline
        .saturating_duration_since(Instant::now())
        .checked_sub(Duration::from_secs(3))
        .filter(|d| !d.is_zero())
        .ok_or("local matrix deadline exhausted before launch")?
        .min(Duration::from_secs(input.candidate_seconds));
    let mut arguments: Vec<_> = invocation["arguments"]
        .as_array()
        .ok_or("arguments")?
        .iter()
        .map(|v| {
            v.as_str()
                .map(|s| Argument::Public(s.into()))
                .ok_or("argument must be text")
        })
        .collect::<Result<_, _>>()?;
    let cert_root = directory.join("harness");
    arguments.extend([
        Argument::Public("--cert-root".into()),
        Argument::Public(cert_root.clone().into()),
    ]);
    let report = process::supervise(
        &ProcessSpec {
            executable: harness.into(),
            arguments,
            cwd: input.plan.source_root.canonicalize()?,
            environment: environment.clone(),
        },
        &Limits {
            execution: budget,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 8 * 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        OutputFiles {
            stdout: Some(directory.join("stdout.log")),
            stderr: Some(directory.join("stderr.log")),
        },
    )?;
    let complete = report.success()
        && !report.cleanup.forced
        && !report.stdout.truncated
        && !report.stderr.truncated
        && report.stdout.line_capture_complete
        && report.stderr.line_capture_complete;
    let mut receipt = json!({"status":"failed","invocation":invocation,"outcome":format!("{:?}",report.outcome),"exit_code":report.status.and_then(|s|s.code()),"cleanup_complete":report.cleanup.complete,"cleanup_forced":report.cleanup.forced,"stdout_suppressed_lines":report.stdout.suppressed_lines,"stderr_suppressed_lines":report.stderr.suppressed_lines,"stdout_bytes_seen":report.stdout.bytes_seen,"stderr_bytes_seen":report.stderr.bytes_seen,"stdout_sha256":digest(&directory.join("stdout.log"))?,"stderr_sha256":digest(&directory.join("stderr.log"))?});
    // Post-run admission may refuse cancellation or changed inputs. Keep the owned
    // process outcome and cleanup evidence even when that admission fails.
    if let Err(error) = admit_toolkits_after_harness(input, execution) {
        receipt["status"] = "refused".into();
        receipt["error"] = error.to_string().into();
        return Ok(receipt);
    }
    if complete {
        match manifest(&cert_root, invocation) {
            Ok((path, sha)) => {
                receipt["status"] = "process_completed_manifest_bound".into();
                receipt["harness_manifest"] = json!(path);
                receipt["harness_manifest_sha256"] = sha.into();
            }
            Err(error) => receipt["error"] = error.to_string().into(),
        }
    }
    Ok(receipt)
}
fn admit_toolkits_after_harness(input: &Input, execution: &Execution<'_>) -> DynResult<()> {
    let after = toolkits::admit(&input.toolkit_dirs, &mut || {
        guard(execution.deadline, execution.cancellation)
    })?;
    if after.observation != execution.toolkits.observation {
        return Err(
            "toolkit directory observation changed across harness execution; logs retained".into(),
        );
    }
    Ok(())
}
fn manifest(root: &Path, invocation: &Value) -> DynResult<(PathBuf, String)> {
    let mut pending = vec![(root.to_path_buf(), 0usize)];
    let mut found = Vec::new();
    let mut count = 0usize;
    while let Some((directory, depth)) = pending.pop() {
        if depth > 8 || !fs::symlink_metadata(&directory)?.is_dir() {
            return Err("harness artifact directory invalid".into());
        }
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            count += 1;
            if count > 4096 {
                return Err("harness artifact census exceeds bound".into());
            }
            let metadata = entry.file_type()?;
            if metadata.is_dir() {
                pending.push((entry.path(), depth + 1));
            } else if metadata.is_file() {
                if entry.file_name() == "manifest.json" {
                    found.push(entry.path());
                }
            } else {
                return Err("harness artifact contains link/special entry".into());
            }
        }
    }
    if found.len() != 1 {
        return Err("harness must emit exactly one owned manifest".into());
    }
    let path = found.pop().ok_or("manifest")?;
    let bytes = parity_local_plan::document(&path)?;
    let value: Value = serde_json::from_slice(&bytes)?;
    let args: Vec<_> = invocation["arguments"]
        .as_array()
        .ok_or("arguments")?
        .iter()
        .map(|v| v.as_str().ok_or("argument"))
        .collect::<Result<_, _>>()?;
    let arg = |name| {
        args.windows(2)
            .find(|pair| pair[0] == name)
            .map(|pair| pair[1])
            .ok_or("identity argument")
    };
    if value["family"] != arg("--family")?
        || value["target_model"] != arg("--target-model")?
        || value["model_id"] != arg("--model-id")?
        || value["run_id"] != arg("--run-id")?
        || value["output_dir"] != json!(path.parent().ok_or("manifest parent")?)
    {
        return Err("harness manifest identity differs from planned candidate".into());
    }
    for (key, flag) in [
        ("layer_end", "--layer-end"),
        ("split_layer", "--split-layer"),
        ("splits", "--splits"),
        ("activation_width", "--activation-width"),
        ("ctx_size", "--ctx-size"),
        ("n_gpu_layers", "--n-gpu-layers"),
    ] {
        if value["correctness"][key] != arg(flag)? {
            return Err("harness correctness metadata differs from invocation".into());
        }
    }
    let capability = path
        .parent()
        .ok_or("manifest parent")?
        .join("capability-draft.json");
    if value["capability_draft"] != json!(capability) {
        return Err("harness capability path differs from owned output".into());
    }
    let capability_value: Value =
        serde_json::from_slice(&parity_local_plan::document(&capability)?)?;
    if capability_value["generated_by"] != "scripts/family-certify.sh"
        || capability_value["family"] != value["family"]
        || capability_value["target_model"] != value["target_model"]
        || capability_value["model_id"] != value["model_id"]
    {
        return Err("harness capability identity differs".into());
    }
    Ok((path.clone(), digest(&path)?))
}
fn guard(deadline: Instant, cancellation: &process::Cancellation) -> std::io::Result<()> {
    if cancellation.is_cancelled() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "local parity admission cancelled",
        ));
    }
    if Instant::now() >= deadline {
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "local parity admission deadline",
        ));
    }
    Ok(())
}
fn checked_digest(
    path: &Path,
    deadline: Instant,
    cancellation: &process::Cancellation,
) -> DynResult<String> {
    guard(deadline, cancellation)?;
    if !fs::symlink_metadata(path)?.is_file() {
        return Err("harness must be regular".into());
    }
    crate::product::digest::file_sha256_with_guard(path, &mut || guard(deadline, cancellation))
        .map_err(|error| error.error.into())
}

pub(super) fn execute_document(
    value: &Value,
    cancellation: &process::Cancellation,
) -> DynResult<Measured> {
    let input: Input = serde_json::from_value(value.clone())?;
    execute(&input, cancellation)
}

/// Measured observations survive the outer authority and interrupt admission.
pub(super) struct Measured {
    evidence: PathBuf,
    pub(super) output: Value,
    deadline: Instant,
    cancellation: process::Cancellation,
}
impl Measured {
    pub(super) fn publish(mut self, admission: DynResult<()>) -> DynResult<Value> {
        let admission_error = admission.err();
        let guard_error = guard(self.deadline, &self.cancellation).err();
        let errors: Vec<_> = admission_error
            .iter()
            .map(ToString::to_string)
            .chain(guard_error.iter().map(ToString::to_string))
            .collect();
        if !errors.is_empty() {
            // An earlier failure retains its original status and all observations.
            if self.output["status"] == "completed" {
                self.output["status"] = "refused".into();
            }
            self.output["terminal_errors"] = json!(errors);
        }
        write_new(&self.evidence.join("run-receipt.json"), &self.output)?;
        if let Some(error) = admission_error {
            return Err(error);
        }
        if let Some(error) = guard_error {
            return Err(error.into());
        }
        if self.output["status"] != "completed" {
            return Err("local family harness failed/refused/cancelled; final and partial receipts retained".into());
        }
        Ok(self.output)
    }
}
#[cfg(test)]
#[path = "parity_terminal_tests.rs"]
mod terminal_tests;
