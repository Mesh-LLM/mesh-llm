//! Fleet fan-out for AgentX benchmarks: one manifest, many machines, everything pinned.
//!
//! `fleet` validates a manifest, pre-flights every kit URL (fresh CDN uploads can take
//! minutes to reach every edge), generates one jy spec per run from the repo-owned
//! worker script, and submits them all. `fleet-status` reports per-job state for a
//! run root; `fleet-harvest` summarizes a completed run. The worker contract and the
//! fleet's hard-won operational rules are documented in
//! `.agents/skills/agentx-fleet-bench/SKILL.md`.
use super::Model;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

pub(super) const FLEET_USAGE: &str =
    "automation agentx {fleet|fleet-status|fleet-harvest} --input PATH [--run-root DIR]";

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FleetManifest {
    pub campaign: String,
    pub product_source_commit: String,
    /// Product repository; defaults to the public mesh-llm repository.
    #[serde(default = "default_repo_url")]
    pub repo_url: String,
    pub mode: String,
    pub corpus: Option<FleetCorpus>,
    pub runs: Vec<FleetRun>,
}

fn default_repo_url() -> String {
    "https://github.com/Mesh-LLM/mesh-llm".to_string()
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FleetCorpus {
    pub repo: String,
    pub rev: String,
    pub sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FleetRun {
    pub model: String,
    pub backend: String,
    #[serde(default)]
    pub probe: BTreeMap<String, u32>,
    /// Model pin overrides; missing fields resolve from the run's model key via
    /// the campaign env contract.
    #[serde(default)]
    pub model_pin: BTreeMap<String, String>,
}

fn positive_hex40(value: &str, field: &str) -> DynResult<()> {
    if value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Ok(());
    }
    Err(format!("{field} requires a full40-hex commit").into())
}

fn validate_path_key(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
}

fn validate(manifest: &FleetManifest) -> DynResult<()> {
    if !validate_path_key(&manifest.campaign) {
        return Err("campaign must be a nonempty filename component".into());
    }
    positive_hex40(&manifest.product_source_commit, "product_source_commit")?;
    if !matches!(manifest.mode.as_str(), "smoke" | "comparison") {
        return Err("mode must be smoke or comparison".into());
    }
    if manifest.runs.is_empty() {
        return Err("fleet manifest needs at least one run".into());
    }
    if manifest.mode == "comparison" && manifest.corpus.is_none() {
        return Err("comparison mode requires corpus pins".into());
    }
    let mut identities = std::collections::BTreeSet::new();
    for run in &manifest.runs {
        if !validate_path_key(&run.model) {
            return Err("model must be a nonempty filename component".into());
        }
        if !identities.insert((&run.model, &run.backend)) {
            return Err(format!("duplicate fleet run {} {}", run.model, run.backend).into());
        }
        if !matches!(run.backend.as_str(), "cuda" | "metal") {
            return Err(format!("run {} backend must be cuda or metal", run.model).into());
        }
    }
    Ok(())
}

fn checked_git(cwd: &Path, args: &[&str]) -> DynResult<()> {
    let output = git_output(Some(cwd), args)?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(())
}

fn git_output(cwd: Option<&Path>, args: &[&str]) -> DynResult<std::process::Output> {
    let mut command = std::process::Command::new("git");
    command.env("GIT_TERMINAL_PROMPT", "0");
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command.args(args);
    Ok(command
        .output()
        .map_err(|e| format!("git {}: {e}", args.first().unwrap_or(&"")))?)
}

/// Pre-flight: the pinned product commit must be resolvable in the product
/// repository before any job is generated or submitted. `git ls-remote`
/// lists only advertised ref tips, so a reachable historical baseline that is
/// no longer a tip would be rejected. Instead, fetch the exact object into a
/// throwaway repository and resolve it there; servers that do not accept
/// SHA-in-want falls back to fetching branch heads, which brings every
/// reachable ancestor with them.
fn commit_exists(repo_url: &str, commit: &str) -> DynResult<bool> {
    let temp = tempfile::tempdir()?;
    let workdir = temp.path();
    checked_git(workdir, &["init", "--bare", "--quiet"])?;
    let exact = git_output(
        Some(workdir),
        &["fetch", "--quiet", "--no-tags", repo_url, commit],
    )?;
    if !exact.status.success() {
        // Most servers refuse SHA-in-want for unpinned objects; fetch branch
        // heads instead, which bring every reachable ancestor along.
        let heads = git_output(
            Some(workdir),
            &[
                "fetch",
                "--quiet",
                "--no-tags",
                repo_url,
                "+refs/heads/*:refs/heads/*",
            ],
        )?;
        if !heads.status.success() {
            return Err(format!(
                "pre-flight fetch from {repo_url} failed: {}",
                String::from_utf8_lossy(&heads.stderr).trim()
            )
            .into());
        }
    }
    let resolved = git_output(
        Some(workdir),
        &["cat-file", "-t", &format!("{commit}^{{commit}}")],
    )?;
    Ok(resolved.status.success())
}

fn worker_script(root: &Path) -> DynResult<String> {
    let path = root.join("skippy/evals/agentx/fleet-worker.sh");
    std::fs::read_to_string(&path)
        .map_err(|_| format!("fleet worker script missing: {}", path.display()))?
        .lines()
        // strip the shebang: the worker runs under bash from the spec command
        .skip(1)
        .collect::<Vec<_>>()
        .join("\n")
        .pipe_ok()
}

trait PipeOk {
    fn pipe_ok(self) -> DynResult<String>;
}

impl PipeOk for String {
    fn pipe_ok(self) -> DynResult<String> {
        Ok(self)
    }
}

fn hex64(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Deterministic jy spec for one run. Model pins arrive fully resolved by the
/// caller (manifest overrides on top of the model catalog contract).
fn run_spec(
    manifest: &FleetManifest,
    run: &FleetRun,
    worker_b64: &str,
    worker_sha: &str,
    model_key: &str,
    pins: &BTreeMap<String, String>,
) -> DynResult<Value> {
    for field in [
        "MODEL_REPO",
        "MODEL_REV",
        "MODEL_FILE",
        "MODEL_SHA",
        "TOK_REPO",
        "TOK_REV",
    ] {
        if !pins.contains_key(field) {
            return Err(format!("run {} missing model pin {field}", run.model).into());
        }
    }
    let mut env = json!({
        "WORKER_B64": worker_b64,
        "WORKER_SHA": worker_sha,
        "MODE": manifest.mode,
        "REPO_URL": manifest.repo_url,
        "BACKEND": run.backend,
        "CAMPAIGN": manifest.campaign,
        "PRODUCT_COMMIT": manifest.product_source_commit,
    });
    let env_obj = env.as_object_mut().unwrap();
    for (key, value) in pins {
        env_obj.insert(key.clone(), json!(value));
    }
    // The strict comparison admission resolves this key against the
    // authoritative catalog, so it must be the canonical catalog key.
    env_obj.insert("MODEL_KEY".into(), json!(model_key));
    env_obj.insert(
        "THREADS".into(),
        json!(run.probe.get("threads").copied().unwrap_or(20).to_string()),
    );
    if let Some(corpus) = manifest.corpus.as_ref() {
        if !hex64(&corpus.sha256) {
            return Err("corpus sha256 must be 64-hex".into());
        }
        env_obj.insert("CORPUS_REPO".into(), json!(corpus.repo));
        env_obj.insert("CORPUS_REV".into(), json!(corpus.rev));
        env_obj.insert("CORPUS_SHA".into(), json!(corpus.sha256));
    }
    let mut probe = json!({
        "PROBE_CONVERSATIONS": run.probe.get("conversations").copied().unwrap_or(4).to_string(),
        "PROBE_TURNS": run.probe.get("turns").copied().unwrap_or(4).to_string(),
        "PROBE_BLOCK_CHARS": run.probe.get("block_chars").copied().unwrap_or(6300).to_string(),
        "PROBE_MAX_TOKENS": run.probe.get("max_tokens").copied().unwrap_or(96).to_string(),
    });
    for (key, value) in probe.as_object_mut().unwrap() {
        env_obj.insert(key.clone(), json!(value));
    }
    // Hardware placement only: the worker's arch and memory guards route the
    // rest. Never target fleet nodes by name.
    let target_os = if run.backend == "cuda" {
        "linux"
    } else {
        "macos"
    };
    Ok(json!({
        "name": format!("{} fleet {} {} ({})", manifest.campaign, run.model, run.backend, manifest.mode),
        "priority": 70,
        "target": {"executor": "lab", "os": target_os, "num_nodes": 1, "spot": false},
        "labels": {"campaign": manifest.campaign, "phase": manifest.mode, "backend": run.backend, "model": run.model},
        "harness": {"name": "custom", "version": format!("agentx-fleet-{}", manifest.mode), "trace": {"kind": "file", "ref": "sha256:".to_string() + &"0".repeat(64)}, "seed": 0, "warmup_requests": 0, "concurrency": [1], "request_count": 1, "duration_s": 86400, "percentiles": [50], "args": {}},
        "runtime": {
            "model_package": {"ref": format!("none:{}", manifest.campaign), "egress": "lab_only"},
            "flags": {"scope": "build"},
            "command": "/bin/bash <<'ENTRY'\nfile=$TMPDIR/worker.sh\nprintf '%s' \"$WORKER_B64\" | base64 -d > \"$file\"\nprintf '%s  %s\\n' \"$WORKER_SHA\" \"$file\" | shasum -a256 -c -\nexec /bin/bash \"$file\"\nENTRY",
            "env": env,
            "volumes": []
        },
        "budget": {"max_duration_s": 86400, "grace_s": 300, "max_usd": 0}
    }))
}

pub(super) fn command(root: &Path, args: &[String]) -> DynResult<()> {
    let (verb, input) = match args.first().map(String::as_str) {
        Some("fleet") => (
            "fleet",
            args.get(1).filter(|f| *f == "--input").and(args.get(2)),
        ),
        Some("fleet-status") => (
            "fleet-status",
            args.get(1).filter(|f| *f == "--run-root").and(args.get(2)),
        ),
        _ => return Err(FLEET_USAGE.into()),
    };
    match (verb, input) {
        ("fleet", Some(path)) => submit(root, Path::new(path)),
        ("fleet-status", Some(run_root)) => status(Path::new(run_root)),
        _ => Err(FLEET_USAGE.into()),
    }
}

fn submit(root: &Path, manifest_path: &Path) -> DynResult<()> {
    let jy = |spec_path: &Path| -> DynResult<String> { jy_submit(spec_path) };
    let preflight =
        |repo_url: &str, commit: &str| -> DynResult<bool> { commit_exists(repo_url, commit) };
    submit_with(root, manifest_path, &jy, &preflight)
}

fn jy_submit(spec_path: &Path) -> DynResult<String> {
    let output = std::process::Command::new("jy")
        .arg("submit")
        .arg(spec_path)
        .output()
        .map_err(|e| format!("jy submit: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "jy submit {} exited with {}: {}",
            spec_path.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string();
    text.lines()
        .find(|l| l.starts_with("job"))
        .and_then(|l| l.split_whitespace().nth(1))
        .map(str::to_string)
        .ok_or_else(|| {
            format!(
                "jy submit printed no job id for {}: {}",
                spec_path.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            )
            .into()
        })
}

/// Submit every validated run spec, persisting each accepted job ID to the
/// campaign ledger atomically as it arrives.
///
/// Every run spec is generated before the first submission so a rejected pin
/// or corpus cannot leave earlier jobs running without a ledger. An existing
/// campaign run-root is rejected unless it carries this exact manifest, in
/// which case only the runs the ledger is still missing are submitted, so a
/// partial failure resumes without duplicating jobs.
fn submit_with(
    root: &Path,
    manifest_path: &Path,
    submit_job: &dyn Fn(&Path) -> DynResult<String>,
    preflight: &dyn Fn(&str, &str) -> DynResult<bool>,
) -> DynResult<()> {
    let manifest: FleetManifest = serde_json::from_slice(
        &crate::automation::receipt_files::bounded(manifest_path, 1024 * 1024)?,
    )?;
    validate(&manifest)?;
    use sha2::{Digest, Sha256};
    let script = worker_script(root)?;
    let worker_sha = hex(&Sha256::digest(script.as_bytes()));
    let worker_b64 = {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(script.as_bytes())
    };
    // Prevalidate every run before the first submission.
    let mut planned = Vec::new();
    for run in &manifest.runs {
        let (model_key, pins) = resolve_pins(root, run, manifest.mode == "comparison")?;
        let spec = run_spec(&manifest, run, &worker_b64, &worker_sha, &model_key, &pins)?;
        planned.push((run, spec));
    }
    let manifest_sha = hex(&Sha256::digest(serde_json::to_vec(&(
        &manifest,
        planned.iter().map(|(_, spec)| spec).collect::<Vec<_>>(),
    ))?));
    let run_root = root.join("target/agentx-fleet").join(&manifest.campaign);
    std::fs::create_dir_all(&run_root)?;
    // Keep this inode in place so independent submitters share one lock.
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(run_root.join("submission.lock"))?;
    lock.try_lock()
        .map_err(|_| "fleet campaign submission is busy")?;
    let ledger_path = run_root.join("fleet.json");
    // Reject or resume an existing campaign before any expensive pre-flight
    // or submission: a re-used campaign name with different inputs must not
    // strand new jobs under an old ledger.
    let mut ledger_runs = match std::fs::read(&ledger_path) {
        Ok(existing) => {
            let ledger: Value = serde_json::from_slice(&existing)
                .map_err(|e| format!("{}: {e}", ledger_path.display()))?;
            let recorded = ledger
                .get("manifest_sha256")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if recorded != manifest_sha {
                return Err(format!(
                    "campaign {} already has a ledger at {} recorded for a different manifest; choose a new campaign name or remove the run root",
                    manifest.campaign,
                    ledger_path.display()
                )
                .into());
            }
            ledger["runs"].as_array().cloned().unwrap_or_default()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    if !preflight(&manifest.repo_url, &manifest.product_source_commit)? {
        return Err(format!(
            "pre-flight failed: commit {} not found in {}",
            &manifest.product_source_commit[..12.min(manifest.product_source_commit.len())],
            manifest.repo_url
        )
        .into());
    }
    std::fs::create_dir_all(&run_root)?;
    let mut newly_submitted = 0usize;
    for (run, spec) in planned {
        let already = ledger_runs.iter().any(|entry| {
            entry["model"].as_str() == Some(run.model.as_str())
                && entry["backend"].as_str() == Some(run.backend.as_str())
        });
        if already {
            continue;
        }
        let path = run_root.join(format!("spec-{}-{}.json", run.model, run.backend));
        std::fs::write(&path, serde_json::to_vec_pretty(&json!({"spec": spec}))?)?;
        let job = submit_job(&path)?;
        ledger_runs.push(
            json!({"model": run.model, "backend": run.backend, "job": job, "spec": path.to_string_lossy()}),
        );
        write_ledger_atomically(&ledger_path, &manifest, &manifest_sha, &ledger_runs)?;
        newly_submitted += 1;
    }
    println!(
        "fleet: {} runs submitted, {} already recorded, ledger at {}",
        newly_submitted,
        ledger_runs.len() - newly_submitted,
        ledger_path.display()
    );
    Ok(())
}

/// Persists the campaign ledger in one rename so an interrupted submission
/// never loses the accepted job IDs that came before it.
fn write_ledger_atomically(
    ledger_path: &Path,
    manifest: &FleetManifest,
    manifest_sha: &str,
    runs: &[Value],
) -> DynResult<()> {
    let ledger = json!({
        "campaign": manifest.campaign,
        "mode": manifest.mode,
        "manifest_sha256": manifest_sha,
        "runs": runs,
    });
    let temp = ledger_path.with_extension("json.tmp");
    std::fs::write(&temp, serde_json::to_vec_pretty(&ledger)?)?;
    std::fs::rename(&temp, ledger_path)?;
    Ok(())
}

fn status(run_root: &Path) -> DynResult<()> {
    let ledger: Value = serde_json::from_slice(&std::fs::read(run_root.join("fleet.json"))?)?;
    for run in ledger["runs"].as_array().unwrap_or(&Vec::new()) {
        let job = run["job"].as_str().unwrap_or_default();
        let output = std::process::Command::new("jy")
            .arg("job")
            .arg(job)
            .output()?;
        let text = String::from_utf8_lossy(&output.stdout).to_string();
        let state = if output.status.success() {
            text.lines()
                .find(|l| l.contains("\"status\""))
                .and_then(|l| l.split('"').nth(3))
                .unwrap_or("unknown")
                .to_string()
        } else {
            format!("jy-error({})", output.status)
        };
        println!(
            "{} {:8} {}",
            run["model"].as_str().unwrap_or("?"),
            run["backend"].as_str().unwrap_or("?"),
            state
        );
    }
    Ok(())
}

/// Resolves a run's model to its canonical catalog key and the pins the
/// worker contract needs. Short manifest aliases map through
/// `skippy/evals/agentx/fleet-models.json`; every repo/revision/file/tokenizer
/// pin comes from the authoritative
/// `skippy/evals/kv-validation/models.json` catalog, which is the same source
/// the comparison admission resolves against. A second pin catalog here would
/// drift from the admitted one, so none is kept.
fn resolve_pins(
    root: &Path,
    run: &FleetRun,
    comparison: bool,
) -> DynResult<(String, BTreeMap<String, String>)> {
    let alias_path = root.join("skippy/evals/agentx/fleet-models.json");
    let aliases: BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&alias_path)?)
        .map_err(|e| format!("{}: {e}", alias_path.display()))?;
    let model_key = aliases
        .get(&run.model)
        .cloned()
        .unwrap_or_else(|| run.model.clone());
    let catalog_path = root.join("skippy/evals/kv-validation/models.json");
    let catalog: Vec<Model> = serde_json::from_slice(&std::fs::read(&catalog_path)?)
        .map_err(|e| format!("{}: {e}", catalog_path.display()))?;
    let model = catalog
        .iter()
        .find(|model| model.key == model_key)
        .ok_or_else(|| {
            format!(
                "run {}: model {model_key} is not in the authoritative catalog {}",
                run.model,
                catalog_path.display()
            )
        })?;
    let file = model
        .files
        .first()
        .ok_or_else(|| format!("catalog model {} pins no weight files", model.key))?;
    let mut pins = BTreeMap::from([
        ("MODEL_REPO".to_string(), model.repo.clone()),
        ("MODEL_REV".to_string(), model.revision.clone()),
        ("MODEL_FILE".to_string(), file.name.clone()),
        ("MODEL_SHA".to_string(), file.sha256.clone()),
        // The tokenizer pin is independent of the GGUF revision; the catalog
        // owns the tokenizer repository and its own revision.
        ("TOK_REPO".to_string(), model.tokenizer.clone()),
        ("TOK_REV".to_string(), model.tokenizer_revision.clone()),
    ]);
    for (key, value) in &run.model_pin {
        let canonical = pins
            .get(key)
            .ok_or_else(|| format!("run {}: unknown model pin {key}", run.model))?;
        if comparison && value != canonical {
            return Err(format!(
                "run {}: comparison pin {key} differs from the authoritative catalog",
                run.model
            )
            .into());
        }
        pins.insert(key.clone(), value.clone());
    }
    Ok((model_key, pins))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests;
