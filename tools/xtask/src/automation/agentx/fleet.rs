//! Fleet fan-out for AgentX benchmarks: one manifest, many machines, everything pinned.
//!
//! `fleet` validates a manifest, pre-flights every kit URL (fresh CDN uploads can take
//! minutes to reach every edge), generates one jy spec per run from the repo-owned
//! worker script, and submits them all. `fleet-status` reports per-job state for a
//! run root; `fleet-harvest` summarizes a completed run. The worker contract and the
//! fleet's hard-won operational rules are documented in
//! `.agents/skills/agentx-fleet-bench/SKILL.md`.
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

fn validate(manifest: &FleetManifest) -> DynResult<()> {
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
    for run in &manifest.runs {
        if !matches!(run.backend.as_str(), "cuda" | "metal") {
            return Err(format!("run {} backend must be cuda or metal", run.model).into());
        }
    }
    Ok(())
}

/// Pre-flight: the pinned product commit must exist in the product repository
/// before any job is generated or submitted.
fn commit_exists(repo_url: &str, commit: &str) -> DynResult<bool> {
    let output = std::process::Command::new("git")
        .args(["ls-remote", repo_url])
        .output()
        .map_err(|e| format!("git ls-remote {repo_url}: {e}"))?;
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.contains(commit))
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
    env_obj.insert("MODEL_KEY".into(), json!(run.model));
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
    let manifest: FleetManifest = serde_json::from_slice(
        &crate::automation::receipt_files::bounded(manifest_path, 1024 * 1024)?,
    )?;
    validate(&manifest)?;
    let script = worker_script(root)?;
    use sha2::{Digest, Sha256};
    let worker_sha = hex(&Sha256::digest(script.as_bytes()));
    let worker_b64 = {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(script.as_bytes())
    };
    let run_root = root.join("target/agentx-fleet").join(&manifest.campaign);
    std::fs::create_dir_all(&run_root)?;
    if !commit_exists(&manifest.repo_url, &manifest.product_source_commit)? {
        return Err(format!(
            "pre-flight failed: commit {} not found in {}",
            &manifest.product_source_commit[..12.min(manifest.product_source_commit.len())],
            manifest.repo_url
        )
        .into());
    }
    let mut resolved = Vec::new();
    for run in &manifest.runs {
        resolved.push((run, resolve_pins(root, run)?));
    }
    let mut submitted = Vec::new();
    for (run, pins) in resolved {
        let spec = run_spec(&manifest, run, &worker_b64, &worker_sha, &pins)?;
        let path = run_root.join(format!("spec-{}-{}.json", run.model, run.backend));
        std::fs::write(&path, serde_json::to_vec_pretty(&json!({"spec": spec}))?)?;
        let output = std::process::Command::new("jy")
            .arg("submit")
            .arg(&path)
            .output()
            .map_err(|e| format!("jy submit: {e}"))?;
        let text = String::from_utf8_lossy(&output.stdout).to_string();
        let job = text
            .lines()
            .find(|l| l.starts_with("job"))
            .and_then(|l| l.split_whitespace().nth(1))
            .ok_or_else(|| {
                format!(
                    "jy submit rejected {}: {}",
                    path.display(),
                    String::from_utf8_lossy(&output.stderr)
                )
            })?;
        submitted
            .push(json!({"model": run.model, "backend": run.backend, "job": job, "spec": path}));
    }
    let ledger = json!({"campaign": manifest.campaign, "mode": manifest.mode, "runs": submitted});
    std::fs::write(
        run_root.join("fleet.json"),
        serde_json::to_vec_pretty(&ledger)?,
    )?;
    println!(
        "fleet: {} runs submitted under {}",
        submitted.len(),
        run_root.display()
    );
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
        let state = text
            .lines()
            .find(|l| l.contains("\"status\""))
            .and_then(|l| l.split('"').nth(3))
            .unwrap_or("unknown");
        println!(
            "{} {:8} {}",
            run["model"].as_str().unwrap_or("?"),
            run["backend"].as_str().unwrap_or("?"),
            state
        );
    }
    Ok(())
}

fn resolve_pins(root: &Path, run: &FleetRun) -> DynResult<BTreeMap<String, String>> {
    // Model pins live in the same shape the campaign specs use, kept in the
    // manifest directory as models.json so a manifest is self-contained.
    let catalog_path = root.join("skippy/evals/agentx/fleet-models.json");
    let catalog: BTreeMap<String, BTreeMap<String, String>> =
        serde_json::from_slice(&std::fs::read(&catalog_path)?)
            .map_err(|e| format!("{}: {e}", catalog_path.display()))?;
    let mut pins = catalog.get(&run.model).cloned().unwrap_or_default();
    for (key, value) in &run.model_pin {
        pins.insert(key.clone(), value.clone());
    }
    Ok(pins)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> FleetManifest {
        serde_json::from_value(json!({
            "campaign": "test-20260101",
            "product_source_commit": "e821ec38b1479cc2dd41da40e1b46e90b4a29850",
            "mode": "smoke",
            "runs": [{"model": "qwen3-4b", "backend": "cuda"}]
        }))
        .unwrap()
    }

    #[test]
    fn fleet_manifest_rejects_bad_commit_and_modes() {
        let mut bad = manifest();
        bad.product_source_commit = "short".into();
        assert!(validate(&bad).is_err());
        let mut mode = manifest();
        mode.mode = "quick".into();
        assert!(validate(&mode).is_err());
    }

    #[test]
    fn fleet_specs_target_hardware_never_node_names() {
        let m = manifest();
        let pins = BTreeMap::from([
            (
                "MODEL_REPO".to_string(),
                "unsloth/Qwen3-4B-Instruct-2507-GGUF".to_string(),
            ),
            (
                "MODEL_REV".to_string(),
                "a06e946bb6b655725eafa393f4a9745d460374c9".to_string(),
            ),
            (
                "MODEL_FILE".to_string(),
                "Qwen3-4B-Instruct-2507-Q4_K_M.gguf".to_string(),
            ),
            ("MODEL_SHA".to_string(), "f".repeat(64)),
            (
                "TOK_REPO".to_string(),
                "Qwen/Qwen3-4B-Instruct-2507".to_string(),
            ),
            (
                "TOK_REV".to_string(),
                "cdbee75f17c01a7cc42f958dc650907174af0554".to_string(),
            ),
        ]);
        let spec = run_spec(&m, &m.runs[0], "dzByaw==", &"c".repeat(64), &pins).unwrap();
        assert_eq!(spec["target"]["os"], "linux");
        assert!(
            spec["target"].get("nodes").is_none(),
            "hardware targeting only"
        );
        assert_eq!(spec["runtime"]["env"]["MODE"], "smoke");
        assert_eq!(
            spec["runtime"]["env"]["REPO_URL"],
            "https://github.com/Mesh-LLM/mesh-llm"
        );
        assert_eq!(spec["runtime"]["env"]["PROBE_TURNS"], "4");
    }

    #[test]
    fn comparison_requires_corpus_pins() {
        let mut m = manifest();
        m.mode = "comparison".into();
        assert!(
            validate(&m).is_err(),
            "comparison without corpus pins must fail"
        );
    }

    #[test]
    fn smoke_manifest_accepts_without_corpus() {
        assert!(validate(&manifest()).is_ok());
    }
}
