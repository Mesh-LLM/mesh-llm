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
    checked_git(workdir, &["init", "--quiet"])?;
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
    for (run, (model_key, pins)) in resolved {
        let spec = run_spec(&manifest, run, &worker_b64, &worker_sha, &model_key, &pins)?;
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

/// Resolves a run's model to its canonical catalog key and the pins the
/// worker contract needs. Short manifest aliases map through
/// `skippy/evals/agentx/fleet-models.json`; every repo/revision/file/tokenizer
/// pin comes from the authoritative
/// `skippy/evals/kv-validation/models.json` catalog, which is the same source
/// the comparison admission resolves against. A second pin catalog here would
/// drift from the admitted one, so none is kept.
fn resolve_pins(root: &Path, run: &FleetRun) -> DynResult<(String, BTreeMap<String, String>)> {
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
        pins.insert(key.clone(), value.clone());
    }
    Ok((model_key, pins))
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
        let spec = run_spec(
            &m,
            &m.runs[0],
            "dzByaw==",
            &"c".repeat(64),
            "qwen3-4b-q4-k-m",
            &pins,
        )
        .unwrap();
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

    /// Writes a self-contained root with an alias map and an authoritative
    /// catalog whose hybrid tokenizer pin is independent of the GGUF revision.
    fn catalog_root(temp: &tempfile::TempDir) -> std::path::PathBuf {
        let agentx = temp.path().join("skippy/evals/agentx");
        let kv = temp.path().join("skippy/evals/kv-validation");
        std::fs::create_dir_all(&agentx).unwrap();
        std::fs::create_dir_all(&kv).unwrap();
        std::fs::write(
            agentx.join("fleet-models.json"),
            r#"{"qwen3.5-4b": "qwen3.5-4b-q4-k-m"}"#,
        )
        .unwrap();
        std::fs::write(
            kv.join("models.json"),
            r#"[
  {
    "key": "qwen3.5-4b-q4-k-m",
    "family": "hybrid-dense",
    "repo": "unsloth/Qwen3.5-4B-GGUF",
    "revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
    "tokenizer": "Qwen/Qwen3.5-4B",
    "files": [
      {
        "name": "Qwen3.5-4B-Q4_K_M.gguf",
        "sha256": "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
      }
    ],
    "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
  }
]"#,
        )
        .unwrap();
        temp.path().to_path_buf()
    }

    /// A two-commit local repository whose branch tip is the second commit,
    /// so the first commit is a reachable non-tip ancestor.
    fn ancestor_fixture() -> (tempfile::TempDir, String, String) {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        checked_git(repo, &["init", "--quiet", "-b", "main"]).unwrap();
        let commit = |message: &str| {
            let output = git_output(
                Some(repo),
                &[
                    "-c",
                    "user.name=fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "--allow-empty",
                    "--quiet",
                    "-m",
                    message,
                ],
            )
            .unwrap();
            assert!(output.status.success(), "fixture commit failed");
        };
        let tip = |repo: &Path| -> String {
            let output = git_output(Some(repo), &["rev-parse", "HEAD"]).unwrap();
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        };
        commit("base");
        let base = tip(repo);
        commit("tip");
        let tip = tip(repo);
        (temp, base, tip)
    }

    #[test]
    fn commit_preflight_accepts_a_non_tip_ancestor_pin() {
        let (temp, base, tip) = ancestor_fixture();
        assert!(commit_exists(temp.path().to_str().unwrap(), &base).unwrap());
        assert!(commit_exists(temp.path().to_str().unwrap(), &tip).unwrap());
    }

    #[test]
    fn commit_preflight_uses_the_exact_object_fetch_when_allowed() {
        let (temp, base, tip) = ancestor_fixture();
        checked_git(
            temp.path(),
            &["config", "uploadpack.allowAnySHA1InWant", "true"],
        )
        .unwrap();
        assert!(commit_exists(temp.path().to_str().unwrap(), &base).unwrap());
        assert!(commit_exists(temp.path().to_str().unwrap(), &tip).unwrap());
    }

    #[test]
    fn commit_preflight_rejects_an_unreachable_object() {
        let (temp, _base, _tip) = ancestor_fixture();
        let missing = "0".repeat(39) + "1";
        assert!(!commit_exists(temp.path().to_str().unwrap(), &missing).unwrap());
    }

    #[test]
    fn commit_preflight_reports_fetch_infrastructure_failures() {
        let error = commit_exists(
            "/nonexistent-repository-path-for-fleet-preflight-test",
            &"a".repeat(40),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("pre-flight fetch"), "{error}");
    }

    #[test]
    fn fleet_alias_resolves_to_the_admitted_catalog_key_and_pins() {
        let temp = tempfile::tempdir().unwrap();
        let root = catalog_root(&temp);
        let run = FleetRun {
            model: "qwen3.5-4b".into(),
            backend: "cuda".into(),
            probe: BTreeMap::new(),
            model_pin: BTreeMap::new(),
        };
        let (model_key, pins) = resolve_pins(&root, &run).unwrap();
        assert_eq!(model_key, "qwen3.5-4b-q4-k-m");
        assert_eq!(pins["MODEL_REPO"], "unsloth/Qwen3.5-4B-GGUF");
        assert_eq!(
            pins["MODEL_REV"],
            "e87f176479d0855a907a41277aca2f8ee7a09523"
        );
        assert_eq!(pins["MODEL_FILE"], "Qwen3.5-4B-Q4_K_M.gguf");
        assert_eq!(
            pins["MODEL_SHA"],
            "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
        );
        // The tokenizer pin is the catalog's own tokenizer revision, not the
        // GGUF revision the alias catalog used to reuse.
        assert_eq!(pins["TOK_REPO"], "Qwen/Qwen3.5-4B");
        assert_eq!(pins["TOK_REV"], "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a");
        assert_ne!(pins["TOK_REV"], pins["MODEL_REV"]);
        // The generated key must pass the existing strict comparison
        // admission that the pinned campaign resolves with.
        crate::automation::agentx::inputs::select_model(&root, &model_key).unwrap();
    }

    #[test]
    fn unknown_fleet_model_names_the_authoritative_catalog() {
        let temp = tempfile::tempdir().unwrap();
        let root = catalog_root(&temp);
        let run = FleetRun {
            model: "qwen3-4b".into(),
            backend: "cuda".into(),
            probe: BTreeMap::new(),
            model_pin: BTreeMap::new(),
        };
        let error = format!("{}", resolve_pins(&root, &run).unwrap_err());
        assert!(error.contains("authoritative catalog"), "{error}");
    }

    #[test]
    fn every_checked_in_fleet_alias_admits_through_the_comparison_catalog() {
        // The fleet aliases must resolve to canonical keys the strict
        // comparison admission accepts, with the catalog's own tokenizer pin.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let alias_path = root.join("skippy/evals/agentx/fleet-models.json");
        let aliases: BTreeMap<String, String> =
            serde_json::from_slice(&std::fs::read(&alias_path).unwrap()).unwrap();
        assert!(!aliases.is_empty(), "the alias map must not be empty");
        for (alias, model_key) in aliases {
            let run = FleetRun {
                model: alias.clone(),
                backend: "cuda".into(),
                probe: BTreeMap::new(),
                model_pin: BTreeMap::new(),
            };
            let (resolved_key, pins) =
                resolve_pins(&root, &run).unwrap_or_else(|error| panic!("alias {alias}: {error}"));
            assert_eq!(resolved_key, model_key, "alias map and resolution agree");
            crate::automation::agentx::inputs::select_model(&root, &resolved_key)
                .unwrap_or_else(|error| panic!("generated model key {resolved_key}: {error}"));
            assert!(
                pins.contains_key("TOK_REV"),
                "the catalog pins the tokenizer revision"
            );
        }
    }
}
