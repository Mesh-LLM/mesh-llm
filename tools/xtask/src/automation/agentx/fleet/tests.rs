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
    let (model_key, pins) = resolve_pins(&root, &run, true).unwrap();
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
    let error = format!("{}", resolve_pins(&root, &run, true).unwrap_err());
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
        let (resolved_key, pins) = resolve_pins(&root, &run, true)
            .unwrap_or_else(|error| panic!("alias {alias}: {error}"));
        assert_eq!(resolved_key, model_key, "alias map and resolution agree");
        crate::automation::agentx::inputs::select_model(&root, &resolved_key)
            .unwrap_or_else(|error| panic!("generated model key {resolved_key}: {error}"));
        assert!(
            pins.contains_key("TOK_REV"),
            "the catalog pins the tokenizer revision"
        );
    }
}

/// A self-contained submission root: worker script plus a two-model
/// authoritative catalog with aliases.
fn submission_root() -> (tempfile::TempDir, std::path::PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().to_path_buf();
    let agentx = root.join("skippy/evals/agentx");
    let kv = root.join("skippy/evals/kv-validation");
    std::fs::create_dir_all(&agentx).unwrap();
    std::fs::create_dir_all(&kv).unwrap();
    std::fs::write(
        agentx.join("fleet-worker.sh"),
        "#!/usr/bin/env bash\necho fixture worker\n",
    )
    .unwrap();
    std::fs::write(
        agentx.join("fleet-models.json"),
        r#"{"qwen3-4b": "qwen3-4b-q4-k-m", "qwen3.5-4b": "qwen3.5-4b-q4-k-m"}"#,
    )
    .unwrap();
    std::fs::write(
        kv.join("models.json"),
        r#"[
  {"key": "qwen3-4b-q4-k-m", "family": "dense", "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
   "revision": "a06e946bb6b655725eafa393f4a9745d460374c9", "tokenizer": "Qwen/Qwen3-4B-Instruct-2507",
   "files": [{"name": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "sha256": "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"}],
   "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
  {"key": "qwen3.5-4b-q4-k-m", "family": "hybrid-dense", "repo": "unsloth/Qwen3.5-4B-GGUF",
   "revision": "e87f176479d0855a907a41277aca2f8ee7a09523", "tokenizer": "Qwen/Qwen3.5-4B",
   "files": [{"name": "Qwen3.5-4B-Q4_K_M.gguf", "sha256": "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"}],
   "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}
]"#,
    )
    .unwrap();
    (temp, root)
}

fn two_run_manifest() -> FleetManifest {
    serde_json::from_value(json!({
        "campaign": "test-20260101",
        "product_source_commit": "e821ec38b1479cc2dd41da40e1b46e90b4a29850",
        "mode": "smoke",
        "runs": [
            {"model": "qwen3-4b", "backend": "cuda"},
            {"model": "qwen3.5-4b", "backend": "cuda"}
        ]
    }))
    .unwrap()
}

fn write_manifest(manifest: &FleetManifest) -> (tempfile::NamedTempFile, std::path::PathBuf) {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), serde_json::to_vec_pretty(manifest).unwrap()).unwrap();
    let path = file.path().to_path_buf();
    (file, path)
}

fn ledger_path(root: &Path, manifest: &FleetManifest) -> std::path::PathBuf {
    root.join("target/agentx-fleet")
        .join(&manifest.campaign)
        .join("fleet.json")
}

fn ledger_runs(root: &Path, manifest: &FleetManifest) -> Vec<Value> {
    let ledger: Value =
        serde_json::from_slice(&std::fs::read(ledger_path(root, manifest)).unwrap()).unwrap();
    ledger["runs"].as_array().cloned().unwrap()
}

#[test]
fn prevalidation_fails_before_any_submission() {
    let (_temp, root) = submission_root();
    let mut bad = two_run_manifest();
    bad.runs.push(
        serde_json::from_value(json!({"model": "unknown-model", "backend": "cuda"})).unwrap(),
    );
    let (_file, manifest_path) = write_manifest(&bad);
    let calls = std::cell::Cell::new(0usize);
    let submit = |_spec: &Path| -> DynResult<String> {
        calls.set(calls.get() + 1);
        Ok("job-should-never-happen".into())
    };
    let preflight = |_repo: &str, _commit: &str| -> DynResult<bool> { Ok(true) };
    let error = submit_with(&root, &manifest_path, &submit, &preflight).unwrap_err();
    assert_eq!(
        calls.get(),
        0,
        "no job may launch when a later run prevalidation fails"
    );
    assert!(
        format!("{error}").contains("unknown-model")
            || format!("{error}").contains("authoritative catalog"),
        "{error}"
    );
}

#[test]
fn partial_failure_keeps_the_accepted_job_ids_in_the_ledger() {
    let (_temp, root) = submission_root();
    let manifest = two_run_manifest();
    let (_file, manifest_path) = write_manifest(&manifest);
    let submit = |spec: &Path| -> DynResult<String> {
        if spec.to_string_lossy().contains("spec-qwen3-4b-cuda.json") {
            Ok("job-first-1".into())
        } else {
            Err("jy submit rejected: quota exhausted".into())
        }
    };
    let preflight = |_repo: &str, _commit: &str| -> DynResult<bool> { Ok(true) };
    let error = submit_with(&root, &manifest_path, &submit, &preflight).unwrap_err();
    assert!(format!("{error}").contains("quota exhausted"), "{error}");
    let runs = ledger_runs(&root, &manifest);
    assert_eq!(
        runs.len(),
        1,
        "the accepted job must already be in the ledger"
    );
    assert_eq!(runs[0]["job"], "job-first-1");
    assert_eq!(runs[0]["model"], "qwen3-4b");
}

#[test]
fn ledger_is_persisted_after_each_acceptance() {
    let (_temp, root) = submission_root();
    let manifest = two_run_manifest();
    let (_file, manifest_path) = write_manifest(&manifest);
    let seen_mid_flight = std::cell::RefCell::new(None);
    let submit = |spec: &Path| -> DynResult<String> {
        if spec.to_string_lossy().contains("spec-qwen3.5-4b-cuda.json") {
            let runs = ledger_runs(&root, &manifest);
            *seen_mid_flight.borrow_mut() = Some(runs.len());
        }
        Ok("job-accepted".into())
    };
    let preflight = |_repo: &str, _commit: &str| -> DynResult<bool> { Ok(true) };
    submit_with(&root, &manifest_path, &submit, &preflight).unwrap();
    assert_eq!(
        seen_mid_flight.borrow().as_ref(),
        Some(&1usize),
        "the first accepted ID must be on disk before the second submission starts"
    );
    assert_eq!(ledger_runs(&root, &manifest).len(), 2);
}

#[test]
fn rerunning_the_same_manifest_resumes_without_duplicating_jobs() {
    let (_temp, root) = submission_root();
    let manifest = two_run_manifest();
    let (_file, manifest_path) = write_manifest(&manifest);
    let calls = std::cell::Cell::new(0usize);
    {
        let calls = &calls;
        let submit = |_spec: &Path| -> DynResult<String> {
            let n = calls.get() + 1;
            calls.set(n);
            Ok(format!("job-{n}"))
        };
        let preflight = |_repo: &str, _commit: &str| -> DynResult<bool> { Ok(true) };
        submit_with(&root, &manifest_path, &submit, &preflight).unwrap();
    }
    assert_eq!(calls.get(), 2);
    {
        let submit = |_spec: &Path| -> DynResult<String> {
            calls.set(calls.get() + 1);
            Ok("job-duplicate".into())
        };
        let preflight = |_repo: &str, _commit: &str| -> DynResult<bool> { Ok(true) };
        submit_with(&root, &manifest_path, &submit, &preflight).unwrap();
    }
    assert_eq!(calls.get(), 2, "the resume must not resubmit recorded runs");
    assert_eq!(ledger_runs(&root, &manifest).len(), 2);
}

#[test]
fn reused_campaign_with_a_different_manifest_is_rejected() {
    let (_temp, root) = submission_root();
    let manifest = two_run_manifest();
    let (_file, manifest_path) = write_manifest(&manifest);
    let calls = std::cell::Cell::new(0usize);
    let submit = |_spec: &Path| -> DynResult<String> {
        calls.set(calls.get() + 1);
        Ok("job-1".into())
    };
    let preflight = |_repo: &str, _commit: &str| -> DynResult<bool> { Ok(true) };
    submit_with(&root, &manifest_path, &submit, &preflight).unwrap();
    assert_eq!(calls.get(), 2);
    let mut divergent = two_run_manifest();
    divergent.mode = "comparison".into();
    divergent.corpus = Some(FleetCorpus {
        repo: "semianalysisai/cc-traces-weka-062126-256k".into(),
        rev: "8fecd2fc56694469f758f0afbbb6335ad3043740".into(),
        sha256: "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef".into(),
    });
    let (_file2, divergent_path) = write_manifest(&divergent);
    let error = submit_with(&root, &divergent_path, &submit, &preflight).unwrap_err();
    assert_eq!(calls.get(), 2, "a divergent manifest must not launch jobs");
    assert!(format!("{error}").contains("different manifest"), "{error}");
    assert_eq!(
        ledger_runs(&root, &manifest).len(),
        2,
        "the old ledger survives"
    );
}

/// Drives the real worker smoke wiring through bash: stub servers at the
/// exact paths the wiring launches, a stub probe whose exit status is the
/// scenario, and the real run_engine plus acceptance block extracted from
/// `fleet-worker.sh` so the test fails when the wiring drifts.
#[cfg(unix)]
fn smoke_wiring_exit(probe_exit: u8) -> (std::process::ExitStatus, String, tempfile::TempDir) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let script = std::fs::read_to_string(root.join("skippy/evals/agentx/fleet-worker.sh")).unwrap();
    let start = script
        .find("  PORT=65100\n")
        .expect("worker smoke section marker");
    let run_engine_start = script[start..]
        .find("  run_engine () {\n")
        .expect("run_engine marker")
        + start;
    let run_engine_end = script[run_engine_start..]
        .find("    return 0\n  }\n")
        .expect("run_engine end marker")
        + run_engine_start
        + "    return 0\n  }\n".len();
    let acceptance_start = script
        .find("  SMOKE_STATUS=0\n")
        .expect("acceptance marker");
    let acceptance_end_marker = "    echo SMOKE_FAILED\n    exit 1\n  fi\n";
    let acceptance_end = script[acceptance_start..]
        .find(acceptance_end_marker)
        .expect("acceptance end marker")
        + acceptance_start
        + acceptance_end_marker.len();
    let wiring = format!(
        "{}\n{}\necho SMOKE_ACCEPTED\n",
        &script[run_engine_start..run_engine_end],
        &script[acceptance_start..acceptance_end],
    );
    let temp = tempfile::tempdir().unwrap();
    let work = temp.path().join("work");
    let repo = work.join("repo");
    std::fs::create_dir_all(repo.join(".deps/agentx-llama-build/bin")).unwrap();
    std::fs::create_dir_all(repo.join("target/release")).unwrap();
    let artifacts = temp.path().join("artifacts");
    std::fs::create_dir_all(&artifacts).unwrap();
    // Stub servers that stay alive where the wiring expects llama-server
    // and skippy to run, so readiness passes and the probe is reached.
    for (path, port_arg) in [
        (".deps/agentx-llama-build/bin/llama-server", "unused"),
        ("target/release/skippy", "unused"),
    ] {
        let server = format!("#!/usr/bin/env bash\nwhile :; do sleep 60; done\n# {port_arg}\n");
        let server_path = work.join("repo").join(path);
        std::fs::write(&server_path, server).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&server_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let shims = temp.path().join("shims");
    std::fs::create_dir_all(&shims).unwrap();
    let cargo = format!("#!/usr/bin/env bash\nexit {probe_exit}\n");
    std::fs::write(shims.join("cargo"), cargo).unwrap();
    // Readiness requires two consecutive successful /v1/models answers.
    std::fs::write(shims.join("curl"), "#!/usr/bin/env bash\nexit 0\n").unwrap();
    for shim in ["cargo", "curl"] {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(shims.join(shim), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let driver = format!(
        "#!/usr/bin/env bash\n\
         set -Eeuo pipefail\n\
         export PATH={}:$PATH\n\
         W={}\n\
         export JIANYANG_ARTIFACT_DIR={}\n\
         mkdir -p \"$W/repo\"\n\
         MODEL=dummy-model MODEL_KEY=stub BACKEND=cuda THREADS=4 RUNTIME_DIR=$W/rt\n\
         PROBE_CONVERSATIONS=4 PROBE_TURNS=4 PROBE_BLOCK_CHARS=6300 PROBE_MAX_TOKENS=96\n\
         PORT=65100\n\
         NGL=99\n\
         CTX=$((131072*PROBE_CONVERSATIONS))\n\
         {wiring}\n",
        shims.display(),
        work.display(),
        artifacts.display(),
    );
    let driver_path = temp.path().join("driver.sh");
    std::fs::write(&driver_path, driver).unwrap();
    let output = std::process::Command::new("bash")
        .arg(&driver_path)
        .output()
        .expect("bash driver");
    (
        output.status,
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        temp,
    )
}

#[cfg(unix)]
#[test]
fn smoke_wiring_accepts_when_the_probe_passes() {
    let (status, output, _artifacts) = smoke_wiring_exit(0);
    assert!(status.success(), "wiring output: {output}");
    assert_eq!(output.matches("RETENTION_OK").count(), 2, "{output}");
    assert!(output.contains("SMOKE_ACCEPTED"), "{output}");
    assert!(!output.contains("SMOKE_FAILED"), "{output}");
}

#[cfg(unix)]
#[test]
fn smoke_wiring_fails_and_keeps_logs_when_the_probe_fails() {
    let (status, output, artifacts) = smoke_wiring_exit(3);
    assert!(
        !status.success(),
        "a failed probe must fail the smoke; output: {output}"
    );
    assert!(output.contains("SMOKE_FAILED"), "{output}");
    assert!(output.contains("SMOKE_FAILED rc=3"), "{output}");
    // Logs are preserved before the nonzero exit.
    for engine in ["llama", "skippy"] {
        assert!(
            artifacts
                .path()
                .join("artifacts")
                .join(format!("serve-{engine}.log"))
                .is_file(),
            "serve log for {engine} must be preserved in artifacts"
        );
    }
}

#[test]
fn concurrent_submission_fails_before_launching_another_job() {
    let (_temp, root) = submission_root();
    let (_file, path) = write_manifest(&two_run_manifest());
    let submit = |_spec: &Path| -> DynResult<String> {
        let nested = |_spec: &Path| -> DynResult<String> { panic!("duplicate submission") };
        let error = submit_with(&root, &path, &nested, &|_, _| Ok(true)).unwrap_err();
        assert!(error.to_string().contains("busy"), "{error}");
        Ok("job-fixture".into())
    };
    submit_with(&root, &path, &submit, &|_, _| Ok(true)).unwrap();
}

#[test]
fn changed_worker_cannot_resume_an_existing_campaign() {
    let (_temp, root) = submission_root();
    let (_file, path) = write_manifest(&two_run_manifest());
    submit_with(&root, &path, &|_| Ok("job-fixture".into()), &|_, _| {
        Ok(true)
    })
    .unwrap();
    std::fs::write(
        root.join("skippy/evals/agentx/fleet-worker.sh"),
        "#!/bin/bash\necho changed\n",
    )
    .unwrap();
    let error = submit_with(
        &root,
        &path,
        &|_| panic!("changed worker submitted"),
        &|_, _| Ok(true),
    )
    .unwrap_err();
    assert!(error.to_string().contains("different manifest"), "{error}");
}

#[test]
fn pin_overrides_cannot_replace_worker_environment_or_comparison_identity() {
    let temp = tempfile::tempdir().unwrap();
    let root = catalog_root(&temp);
    let mut run = FleetRun {
        model: "qwen3.5-4b".into(),
        backend: "cuda".into(),
        probe: BTreeMap::new(),
        model_pin: BTreeMap::from([("MODE".into(), "smoke".into())]),
    };
    assert!(
        resolve_pins(&root, &run, false)
            .unwrap_err()
            .to_string()
            .contains("unknown model pin")
    );
    run.model_pin = BTreeMap::from([("TOK_REV".into(), "different".into())]);
    assert!(
        resolve_pins(&root, &run, true)
            .unwrap_err()
            .to_string()
            .contains("differs from the authoritative catalog")
    );
    assert_eq!(
        resolve_pins(&root, &run, false).unwrap().1["TOK_REV"],
        "different"
    );
}

#[test]
fn fleet_rejects_unsafe_paths_and_duplicate_ledger_identities() {
    for key in [
        "",
        ".",
        "..",
        "../outside",
        "/absolute",
        "nested/name",
        "back\\slash",
    ] {
        let mut m = manifest();
        m.campaign = key.into();
        assert!(validate(&m).is_err(), "unsafe campaign {key}");
        m = manifest();
        m.runs[0].model = key.into();
        assert!(validate(&m).is_err(), "unsafe model {key}");
    }
    let mut m = manifest();
    m.runs
        .push(serde_json::from_value(serde_json::to_value(&m.runs[0]).unwrap()).unwrap());
    assert!(
        validate(&m)
            .unwrap_err()
            .to_string()
            .contains("duplicate fleet run")
    );
}
