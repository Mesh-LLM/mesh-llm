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
    pub mode: String,
    pub kit: FleetKit,
    pub runs: Vec<FleetRun>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FleetKit {
    pub url_base: String,
    pub cuda_chunks: Option<FleetChunkSet>,
    pub llama_chunks: Option<FleetChunkSet>,
    pub metal_kit: Option<FleetArtifact>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FleetChunkSet {
    pub prefix: String,
    pub count: u32,
    pub sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FleetArtifact {
    pub key: String,
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
    for run in &manifest.runs {
        if !matches!(run.backend.as_str(), "cuda" | "metal") {
            return Err(format!("run {} backend must be cuda or metal", run.model).into());
        }
        if run.backend == "metal" && manifest.kit.metal_kit.is_none() {
            return Err(format!(
                "run {} is metal but the kit has no metal artifact",
                run.model
            )
            .into());
        }
        if run.backend == "cuda" && manifest.kit.cuda_chunks.is_none() {
            return Err(format!("run {} is cuda but the kit has no cuda chunks", run.model).into());
        }
    }
    Ok(())
}

/// Kit URLs a manifest depends on, for pre-flight. Returns (url, label) pairs.
fn kit_urls(manifest: &FleetManifest) -> Vec<(String, String)> {
    let base = manifest.kit.url_base.trim_end_matches('/');
    let mut urls = Vec::new();
    if let Some(cuda) = manifest.kit.cuda_chunks.as_ref() {
        for index in 0..cuda.count {
            urls.push((
                format!("{base}/{}{index:02}", cuda.prefix),
                format!("cuda chunk {index:02}"),
            ));
        }
    }
    if let Some(llama) = manifest.kit.llama_chunks.as_ref() {
        for index in 0..llama.count {
            urls.push((
                format!("{base}/{}{index:02}", llama.prefix),
                format!("llama chunk {index:02}"),
            ));
        }
    }
    if let Some(metal) = manifest.kit.metal_kit.as_ref() {
        urls.push((format!("{base}/{}", metal.key), "metal kit".to_string()));
    }
    urls
}

fn worker_script(root: &Path, mode: &str) -> DynResult<String> {
    let script = match mode {
        "smoke" => "fleet-smoke-worker.sh",
        _ => "fleet-worker.sh",
    };
    let path = root.join("skippy/evals/agentx").join(script);
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
        "KIT_URL_BASE": manifest.kit.url_base.trim_end_matches('/'),
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
    if let Some(cuda) = manifest.kit.cuda_chunks.as_ref() {
        if !hex64(&cuda.sha256) {
            return Err("cuda chunk sha256 must be 64-hex".into());
        }
        env_obj.insert("CUDA_KEY_PREFIX".into(), json!(cuda.prefix));
        env_obj.insert("CUDA_CHUNKS".into(), json!(cuda.count.to_string()));
        env_obj.insert("CUDA_TGZ_SHA".into(), json!(cuda.sha256));
    }
    if let Some(llama) = manifest.kit.llama_chunks.as_ref() {
        env_obj.insert("LLAMA_KEY_PREFIX".into(), json!(llama.prefix));
        env_obj.insert("LLAMA_CHUNKS".into(), json!(llama.count.to_string()));
        env_obj.insert("LLAMA_TGZ_SHA".into(), json!(llama.sha256));
    }
    if let Some(metal) = manifest.kit.metal_kit.as_ref() {
        env_obj.insert("METAL_KIT_KEY".into(), json!(metal.key));
        env_obj.insert("METAL_KIT_SHA".into(), json!(metal.sha256));
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
    let script = worker_script(root, &manifest.mode)?;
    use sha2::{Digest, Sha256};
    let worker_sha = hex(&Sha256::digest(script.as_bytes()));
    let worker_b64 = {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(script.as_bytes())
    };
    let run_root = root.join("target/agentx-fleet").join(&manifest.campaign);
    std::fs::create_dir_all(&run_root)?;
    // Pre-flight every kit URL before generating or submitting anything.
    for (url, label) in kit_urls(&manifest) {
        let status = crate::command::probe_url_range(&url)?;
        if status != 206 {
            return Err(format!("pre-flight failed for {label}: HTTP {status} from {url} (fresh uploads can take minutes to propagate)").into());
        }
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
            "kit": {"url_base": "https://cdn.example/x",
                    "cuda_chunks": {"prefix": "cuda5-", "count": 2, "sha256": "a".repeat(64)},
                    "llama_chunks": {"prefix": "llama-", "count": 1, "sha256": "b".repeat(64)}},
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
        assert_eq!(spec["runtime"]["env"]["CUDA_CHUNKS"], "2");
        assert_eq!(spec["runtime"]["env"]["PROBE_TURNS"], "4");
    }

    #[test]
    fn preflight_urls_cover_every_chunk() {
        let urls = kit_urls(&manifest());
        assert_eq!(urls.len(), 3, "2 cuda chunks + 1 llama chunk");
        assert!(urls[0].0.ends_with("/cuda5-00"));
    }
}
