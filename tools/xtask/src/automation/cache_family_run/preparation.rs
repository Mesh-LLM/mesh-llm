//! Local observations construct existing typed producer requests; no build/download.
use super::{contract, hash};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Operator {
    schema_version: u64,
    plan: Value,
    correctness: PathBuf,
    stage_server: PathBuf,
    #[serde(default)]
    borrow_resident_hits: bool,
    #[serde(default)]
    cache_decoded_result_hits: bool,
    native_server: Option<PathBuf>,
    artifact_tool: Option<PathBuf>,
    native_build: PathBuf,
    old_source_commit: String,
    new_source_commit: String,
    native_source_commit: String,
    #[serde(default)]
    environment: BTreeMap<String, String>,
    #[serde(default)]
    toolkit_directories: BTreeMap<String, crate::automation::cache_family_profile::Toolkit>,
    execution_seconds: u64,
    cell_seconds: u64,
    request_timeout_ms: u64,
    preparation_seconds: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    operator: Operator,
    use_cases: bool,
}
struct Budget<'a> {
    until: std::time::Instant,
    cancel: &'a Cancellation,
}
impl Budget<'_> {
    fn guard(&self) -> std::io::Result<()> {
        if self.cancel.is_cancelled() || std::time::Instant::now() >= self.until {
            Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "cache preparation interrupted",
            ))
        } else {
            Ok(())
        }
    }
}
fn file(path: &Path, budget: &Budget<'_>) -> DynResult<(PathBuf, String)> {
    budget.guard()?;

    let path = path.canonicalize()?;
    if !std::fs::symlink_metadata(&path)?.is_file() {
        return Err("cache preparation requires canonical regular file".into());
    }
    let pin = crate::product::digest::file_sha256_with_guard(&path, &mut || budget.guard())
        .map_err(|e| e.error)?;
    Ok((path, pin))
}
fn commit(value: &str) -> bool {
    [40, 64].contains(&value.len())
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
impl Operator {
    fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || !(30..=3600).contains(&self.preparation_seconds)
            || ![&self.correctness, &self.stage_server, &self.native_build]
                .iter()
                .all(|p| p.is_absolute())
            || !self
                .native_server
                .iter()
                .chain(self.artifact_tool.iter())
                .all(|p| p.is_absolute())
            || ![
                &self.old_source_commit,
                &self.new_source_commit,
                &self.native_source_commit,
            ]
            .iter()
            .all(|v| commit(v))
        {
            return Err(
                "invalid bounded cache operator tools/provenance/preparation budget".into(),
            );
        }
        crate::automation::cache_family_profile::validate(
            &self.environment,
            &self.toolkit_directories,
        )?;
        contract::Input {
            schema_version: 1,
            plan: self.plan.clone(),
            profiles: BTreeMap::new(),
            execution_seconds: self.execution_seconds,
            cell_seconds: self.cell_seconds,
            request_timeout_ms: self.request_timeout_ms,
        }
        .validate()
    }
}
fn pin_server(plan: &mut Value, field: &str, budget: &Budget<'_>) -> DynResult<Option<Value>> {
    if plan[field].is_null() {
        return Ok(None);
    }
    let path = PathBuf::from(plan[field]["path"].as_str().ok_or("cache server path")?);
    let (path, pin) = file(&path, budget)?;
    if !plan[field]["sha256"].is_null() && plan[field]["sha256"] != pin {
        return Err("cache declared serving tool pin mismatch".into());
    }
    let value = json!({"path":path,"sha256":pin});
    plan[field] = value.clone();
    Ok(Some(value))
}
struct Tools {
    correctness: Value,
    stage: Value,
    native: Option<Value>,
    old: Option<Value>,
    new: Option<Value>,
    artifact: Option<Value>,
    build: PathBuf,
    build_pin: String,
}
fn tools(operator: &mut Operator, budget: &Budget<'_>) -> DynResult<Tools> {
    let (correctness, pin) = file(&operator.correctness, budget)?;
    let (stage, stage_pin) = file(&operator.stage_server, budget)?;
    let native = operator
        .native_server
        .as_ref()
        .map(|p| file(p, budget).map(|(path, sha256)| json!({"path":path,"sha256":sha256})))
        .transpose()?;
    let artifact = operator
        .artifact_tool
        .as_ref()
        .map(|p| file(p, budget).map(|(path, sha256)| json!({"path":path,"sha256":sha256})))
        .transpose()?;
    let old = pin_server(&mut operator.plan, "old_server", budget)?;
    let new = pin_server(&mut operator.plan, "new_server", budget)?;
    let build = operator.native_build.canonicalize()?;
    budget.guard()?;
    let build_pin = crate::product::digest::tree_sha256(&build).map_err(|e| e.error)?;
    budget.guard()?;
    crate::automation::waiting_prefix::native_identity::verify(&build, &build_pin)?;
    crate::automation::cache_family_profile::observe(&mut operator.toolkit_directories)?;
    Ok(Tools {
        correctness: json!({"path":correctness,"sha256":pin}),
        stage: json!({"path":stage,"sha256":stage_pin}),
        native,
        old,
        new,
        artifact,
        build,
        build_pin,
    })
}
fn runtime_model(cell: &Value) -> &Value {
    if cell["model_observation"]["runtime_entrypoint"].is_null() {
        &cell["model_observation"]["canonical"]
    } else {
        &cell["model_observation"]["runtime_entrypoint"]
    }
}
fn artifact(cell: &Value, tools: &Tools, budget: &Budget<'_>) -> DynResult<Value> {
    match cell["model_observation"]["kind"].as_str() {
        Some("single-gguf") => Ok(Value::Null),
        Some(kind @ ("split-gguf-first-shard" | "layer-package-tree")) => {
            let tool = tools
                .artifact
                .as_ref()
                .ok_or("cache shard/package preparation requires native artifact tool")?;
            let mut pins = BTreeMap::new();
            if kind == "split-gguf-first-shard" {
                let model = PathBuf::from(runtime_model(cell).as_str().ok_or("shard primary")?);
                for i in 1..=3 {
                    let name = format!("MiniMax-M2.7-UD-Q2_K_XL-{i:05}-of-00003.gguf");
                    let (_, pin) =
                        file(&model.parent().ok_or("shard parent")?.join(&name), budget)?;
                    pins.insert(name, pin);
                }
            }
            Ok(
                json!({"kind":if kind=="split-gguf-first-shard"{"complete-shards"}else{"layer-package"},"tool":tool["path"],"tool_sha256":tool["sha256"],"shard_pins":pins}),
            )
        }
        _ => Err("cache preparation does not recognize planned artifact".into()),
    }
}
fn profile(
    operator: &Operator,
    cell: &Value,
    tools: &Tools,
    pin: &str,
    budget: &Budget<'_>,
) -> DynResult<contract::Profile> {
    let case = &cell["case"];
    let model = runtime_model(cell);
    let artifact = artifact(cell, tools, budget)?;
    let stage = &tools.stage;
    let correctness = json!({"schema_version":1,"case_key":cell["key"],"model_id":case["model_id"],"correctness":tools.correctness["path"],"correctness_sha256":tools.correctness["sha256"],"stage_server":stage["path"],"stage_server_sha256":stage["sha256"],"model":model,"model_sha256":pin,"artifact":artifact,"native_build":tools.build,"native_build_sha256":tools.build_pin,"ctx_size":case["ctx_size"],"prefix_tokens":case["prefix_tokens"],"cache_hit_repeats":case["cache_hit_repeats"],"runtime_lane_count":operator.plan["runtime_lane_count"].as_u64().unwrap_or(1),"source_port":1,"restore_port":2,"n_gpu_layers":case["n_gpu_layers"],"prompt":null,"topologies":if cell["key"]=="deepseek3"{json!(["package-stage1"])}else{json!(["one-stage"])},"borrow_resident_hits":operator.borrow_resident_hits,"cache_decoded_result_hits":operator.cache_decoded_result_hits,"execution_seconds":operator.cell_seconds,"cell_seconds":operator.cell_seconds.saturating_sub(9).max(1),"settings":operator.environment,"toolkit_directories":operator.toolkit_directories});
    let host = |tool: &Option<Value>, kind: &str, commit: &str| {
        tool.as_ref().map(|tool|json!({"schema_version":1,"host":kind,"binary":tool["path"],"binary_sha256":tool["sha256"],"source_commit":commit,"native_build":tools.build,"native_build_sha256":tools.build_pin,"model":model,"model_sha256":pin,"artifact":artifact,"model_id":case["model_id"],"layer_end":case["layer_end"],"ctx_size":case["ctx_size"],"lane_count":1,"n_gpu_layers":case["n_gpu_layers"],"port":1,"environment":operator.environment,"toolkit_directories":operator.toolkit_directories,"worker":{},"startup_timeout_secs":operator.cell_seconds.saturating_sub(14).clamp(1,900),"execution_timeout_secs":operator.cell_seconds}))
    };
    Ok(contract::Profile {
        correctness,
        native: host(
            &tools.native,
            "native-baseline",
            &operator.native_source_commit,
        ),
        old: host(&tools.old, "skippy-old", &operator.old_source_commit),
        new: host(&tools.new, "skippy-new", &operator.new_source_commit),
    })
}
fn materialize(
    request: &mut Request,
    directory: &Path,
    budget: &Budget<'_>,
) -> DynResult<contract::Input> {
    request.operator.validate()?;
    let tools = tools(&mut request.operator, budget)?;
    if request.use_cases {
        request.operator.plan["use_cases"] = json!(["all"]);
        if request.operator.plan["corpus"].is_null() {
            return Err("use-case preparation requires declared pinned corpus".into());
        }
    } else {
        request.operator.plan["use_cases"] = json!([]);
        request.operator.plan["corpus"] = Value::Null;
    }
    let until = budget.until;
    let cancel = budget.cancel;
    let (plan, clean) = super::child::run(
        "cache-family-plan",
        &request.operator.plan,
        &directory.join("plan"),
        until,
        cancel,
    )?;
    if !clean {
        return Err("cache operator plan refused".into());
    }
    let mut profiles = BTreeMap::new();
    let mut pins = BTreeMap::new();
    for cell in plan["cells"]
        .as_array()
        .ok_or("cache operator plan cells")?
    {
        if cancel.is_cancelled() || std::time::Instant::now() >= until {
            return Err("cache preparation interrupted".into());
        }
        if cell["model_observation"]["status"] == "missing-model" {
            continue;
        }
        let key = cell["key"].as_str().ok_or("case key")?;
        if profiles.contains_key(key) {
            continue;
        }
        let model = PathBuf::from(
            cell["model_observation"]["canonical"]
                .as_str()
                .ok_or("planned model path")?,
        );
        let path = if cell["model_observation"]["kind"] == "layer-package-tree" {
            model.join("manifest.json")
        } else {
            model
        };
        let (_, pin) = file(&path, budget)?;
        if let Some(declared) = request.operator.plan["model_sha256"].get(key)
            && declared != &pin
        {
            return Err("cache supplied model pin mismatch".into());
        }
        profiles.insert(
            key.into(),
            profile(&request.operator, cell, &tools, &pin, budget)?,
        );
        pins.insert(key.to_owned(), pin);
    }
    request.operator.plan["model_sha256"] = json!(pins);
    let result = contract::Input {
        schema_version: 1,
        plan: request.operator.plan.clone(),
        profiles,
        execution_seconds: request.operator.execution_seconds,
        cell_seconds: request.operator.cell_seconds,
        request_timeout_ms: request.operator.request_timeout_ms,
    };
    result.validate()?;
    Ok(result)
}
pub(super) fn worker(path: &Path, output: &Path) -> DynResult<()> {
    let bytes =
        crate::automation::waiting_prefix::adaptive_identity::bounded(path, 2 * 1024 * 1024)?;
    let mut request: Request = serde_json::from_slice(&bytes)?;
    request.operator.validate()?;
    let directory = output.parent().ok_or("cache preparation worker parent")?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let budget = Budget {
        until: std::time::Instant::now()
            + Duration::from_secs(request.operator.preparation_seconds),
        cancel: &cancel,
    };
    let value = materialize(&mut request, directory, &budget);
    let finish = interrupt.finish();
    let value = value?;
    finish?;
    budget.guard()?;
    super::publish(
        output,
        &json!({"request_sha256":hash(&bytes),"input":value,"scope":"observed_local_bytes_and_declared_source_commits_not_build_attestation"}),
    )
}
pub(super) fn run(args: &[String], use_cases: bool) -> DynResult<()> {
    let [a, input, b, output, overrides @ ..] = args else {
        return Err("cache preparation requires ordered input/output".into());
    };
    if a != "--input"
        || b != "--output"
        || !Path::new(input).is_absolute()
        || !Path::new(output).is_absolute()
    {
        return Err("cache preparation requires absolute input/output".into());
    }
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(
        Path::new(input),
        1024 * 1024,
    )?;
    let mut operator: Operator = serde_json::from_slice(&bytes)?;
    override_plan(&mut operator, overrides)?;
    operator.validate()?;
    admit_output(&operator, Path::new(output))?;
    std::fs::create_dir(output)?;
    let request = serde_json::to_vec(&Request {
        operator,
        use_cases,
    })?;
    let path = Path::new(output).join("request.json");
    crate::automation::waiting_prefix::adaptive_identity::fresh(&path, &request)?;
    let receipt = Path::new(output).join("observations.json");
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let parsed: Request = serde_json::from_slice(&request)?;
    let deadline =
        std::time::Instant::now() + Duration::from_secs(parsed.operator.preparation_seconds);
    let result = supervise_worker(
        &path,
        &receipt,
        Path::new(output),
        parsed.operator.preparation_seconds,
        &cancel,
    );
    let finish = interrupt.finish();
    let process = result?;
    finish?;
    if !process.success()
        || process.cleanup.forced
        || !process.cleanup.complete
        || process.cleanup.failure.is_some()
        || process.cleanup.graceful_signal_failed
        || [&process.stdout, &process.stderr]
            .iter()
            .any(|s| !s.line_capture_complete || s.truncated || s.oversized_lines != 0)
        || cancel.is_cancelled()
        || std::time::Instant::now() >= deadline
    {
        return Err("cache preparation child refused; observations retained".into());
    }
    let observed: Value = serde_json::from_slice(
        &crate::automation::waiting_prefix::adaptive_identity::bounded(&receipt, 4 * 1024 * 1024)?,
    )?;
    if observed["request_sha256"] != hash(&request) {
        return Err("cache preparation request correlation mismatch".into());
    }
    let value: contract::Input = serde_json::from_value(observed["input"].clone())?;
    value.validate()?;
    let bytes = serde_json::to_vec_pretty(&value)?;
    publish_eligible(
        &Path::new(output).join("cache-family-input.json"),
        &bytes,
        &mut || {
            Budget {
                until: deadline,
                cancel: &cancel,
            }
            .guard()
            .map_err(Into::into)
        },
        &mut |path, bytes| crate::automation::waiting_prefix::adaptive_identity::fresh(path, bytes),
    )
}

fn override_plan(operator: &mut Operator, args: &[String]) -> DynResult<()> {
    if args.len() > 14 || !args.len().is_multiple_of(2) {
        return Err("cache preparation profile overrides require closed name/value pairs".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    for [name, value] in args.as_chunks::<2>().0 {
        if name == "--use-case-corpus" {
            if !seen.insert("corpus") {
                return Err("duplicate corpus override".into());
            }
            let path = Path::new(value);
            if !path.is_absolute() {
                return Err("corpus override must be absolute".into());
            }
            let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(
                path,
                4 * 1024 * 1024,
            )?;
            operator.plan["corpus"] = json!({"path":path,"sha256":hash(&bytes)});
            continue;
        }

        let field = match name.as_str() {
            "--prefix-tokens" => "prefix_tokens",
            "--runtime-lane-count" => "runtime_lane_count",
            "--llama-parallel" => "llama_parallel",
            "--llama-repeats" => "llama_repeats",
            "--cache-hit-repeats" => "cache_hit_repeats",
            _ => return Err("unknown cache preparation override".into()),
        };
        if !seen.insert(field) {
            return Err("duplicate cache preparation override".into());
        }
        let value: u32 = value.parse()?;
        if value == 0 {
            return Err("cache preparation profile override must be positive".into());
        }
        operator.plan[field] = json!(value);
    }
    Ok(())
}

fn admit_output(operator: &Operator, output: &Path) -> DynResult<()> {
    let parent = output
        .parent()
        .ok_or("cache preparation output parent")?
        .canonicalize()?;
    let name = output
        .file_name()
        .ok_or("cache preparation output filename")?;
    let destination = parent.join(name);
    let cache = operator.plan["cache_root"]
        .as_str()
        .ok_or("cache preparation cache root")?;
    for source in [operator.native_build.as_path(), Path::new(cache)]
        .into_iter()
        .chain(
            operator
                .toolkit_directories
                .values()
                .map(|t| t.path.as_path()),
        )
    {
        let source = source.canonicalize()?;
        if destination.starts_with(&source) || source.starts_with(&destination) {
            return Err(
                "cache preparation output must be disjoint from immutable source trees".into(),
            );
        }
    }
    Ok(())
}
fn publish_eligible(
    path: &Path,
    bytes: &[u8],
    guard: &mut impl FnMut() -> DynResult<()>,
    writer: &mut impl FnMut(&Path, &[u8]) -> DynResult<()>,
) -> DynResult<()> {
    guard()?;
    writer(path, bytes)?;
    if let Err(error) = guard() {
        std::fs::remove_file(path)?;
        return Err(error);
    }
    Ok(())
}

fn supervise_worker(
    path: &Path,
    receipt: &Path,
    output: &Path,
    seconds: u64,
    cancel: &Cancellation,
) -> DynResult<process::ProcessReport> {
    Ok(process::supervise(
        &ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: [
                "automation".into(),
                "cache-family-run".into(),
                "prepare-worker".into(),
                "--input".into(),
                path.to_path_buf().into_os_string(),
                "--output".into(),
                receipt.to_path_buf().into_os_string(),
            ]
            .into_iter()
            .map(Arg::Public)
            .collect(),
            cwd: output.into(),
            environment: ["PATH", "SYSTEMROOT", "WINDIR"]
                .into_iter()
                .filter_map(|k| std::env::var_os(k).map(|v| (k.into(), Arg::Public(v))))
                .collect(),
        },
        &Limits {
            execution: Duration::from_secs(seconds),
            graceful_shutdown: Duration::from_secs(12),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        OutputFiles {
            stdout: Some(Path::new(output).join("prepare.stdout.log")),
            stderr: Some(Path::new(output).join("prepare.stderr.log")),
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_preparation_guard_refuses_cancelled_and_expired_file_observation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tool");
        std::fs::write(&path, b"observed bytes").unwrap();
        let cancellation = Cancellation::default();
        let live = Budget {
            until: std::time::Instant::now() + Duration::from_secs(5),
            cancel: &cancellation,
        };
        assert_eq!(file(&path, &live).unwrap().1, hash(b"observed bytes"));
        let expired = Budget {
            until: std::time::Instant::now(),
            cancel: &cancellation,
        };
        assert!(file(&path, &expired).is_err());
        cancellation.cancel();
        assert!(
            file(&directory.path().join("missing"), &live)
                .unwrap_err()
                .to_string()
                .contains("interrupted")
        );
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    #[test]
    fn cache_preparation_final_publication_revokes_owned_input_on_late_cancel_deadline() {
        for reason in ["cancelled", "deadline"] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("eligible.json");
            let mut calls = 0;
            let result = publish_eligible(
                &path,
                b"candidate",
                &mut || {
                    calls += 1;
                    if calls == 2 {
                        Err(reason.into())
                    } else {
                        Ok(())
                    }
                },
                &mut |p, b| crate::automation::waiting_prefix::adaptive_identity::fresh(p, b),
            );
            assert!(result.is_err());
            assert_eq!(calls, 2);
            assert!(!path.exists());
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("existing.json");
        std::fs::write(&path, b"outside candidate").unwrap();
        assert!(
            publish_eligible(&path, b"replacement", &mut || Ok(()), &mut |p, b| {
                crate::automation::waiting_prefix::adaptive_identity::fresh(p, b)
            })
            .is_err()
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"outside candidate");
        let path = directory.path().join("success.json");
        publish_eligible(&path, b"candidate", &mut || Ok(()), &mut |p, b| {
            crate::automation::waiting_prefix::adaptive_identity::fresh(p, b)
        })
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"candidate");
    }
}
