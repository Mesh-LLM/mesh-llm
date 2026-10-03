//! Actual optional downloader shell and typed manual-admission owner, with finite HF seams.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Output,
};
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    manifest: PathBuf,
    registry: PathBuf,
    snapshot: PathBuf,
    hf: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        fs::create_dir_all(root.join("scripts/lib")).unwrap();
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for name in ["download-skippy-parity-candidates.sh", "lib/automation.sh"] {
            fs::copy(
                repo.join("scripts").join(name),
                root.join("scripts").join(name),
            )
            .unwrap();
        }
        let snapshot = root.join("snapshot/nested");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("model.gguf"), b"model-bytes").unwrap();
        fs::write(snapshot.join("projector.gguf"), b"projector-bytes").unwrap();
        let manifest = root.join("candidates.json");
        let registry = root.join("models.json");
        let hf = root.join("hf");
        fs::write(&hf,"#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$TRACE\"\nprintf 'path=%s\\n' \"$SNAPSHOT\"\nexit \"${HF_STATUS:-0}\"\n").unwrap();
        fs::set_permissions(&hf, fs::Permissions::from_mode(0o755)).unwrap();
        let fixture = Self {
            _temp: temp,
            root,
            manifest,
            registry,
            snapshot,
            hf,
        };
        fixture.save(&fixture.candidates(), &json!({"artifacts":[]}));
        fixture
    }
    fn candidates(&self) -> Json {
        let record = |name: &str| {
            let bytes = fs::read(self.snapshot.join(name)).unwrap();
            json!({"size_bytes":bytes.len(),"blob_id":hex::encode(Sha256::digest(bytes))})
        };
        json!({"support_priority":{"p0":{"families":["vision-family"],"llama_models":[]}},"candidates":[{"llama_model":"vision","family":"vision-family","status":"candidate_multimodal","repo":"owner/model","revision":"0123456789abcdef0123456789abcdef01234567","include":["nested/model.gguf","nested/projector.gguf"],"file_integrity":{"nested/model.gguf":record("model.gguf"),"nested/projector.gguf":record("projector.gguf")}}]})
    }
    fn save(&self, candidates: &Json, registry: &Json) {
        fs::write(&self.manifest, serde_json::to_vec(candidates).unwrap()).unwrap();
        fs::write(&self.registry, serde_json::to_vec(registry).unwrap()).unwrap();
    }
    fn run(&self, args: &[&str], status: u8) -> Output {
        let mut arguments = vec![Value::Public(
            self.root
                .join("scripts/download-skippy-parity-candidates.sh")
                .into(),
        )];
        arguments.extend(args.iter().map(|arg| Value::Public((*arg).into())));
        arguments.extend([
            Value::Public("--timeout-secs".into()),
            Value::Public("2".into()),
        ]);
        self.execute("/bin/bash".into(), arguments, status)
    }
    fn typed(&self, cadence: &[&str]) -> Output {
        let mut arguments = ["models", "parity-download", "--dry-run"]
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect::<Vec<_>>();
        for (option, path) in [
            ("--manifest", &self.manifest),
            ("--model-manifest", &self.registry),
            ("--hf-command", &self.hf),
        ] {
            arguments.extend([
                Value::Public(option.into()),
                Value::Public(path.clone().into()),
            ]);
        }
        arguments.extend(cadence.iter().map(|arg| Value::Public((*arg).into())));
        self.execute(env!("CARGO_BIN_EXE_xtask").into(), arguments, 0)
    }
    fn execute(&self, executable: PathBuf, arguments: Vec<Value>, status: u8) -> Output {
        let environment = std::collections::BTreeMap::from([
            (
                "PATH".into(),
                Value::Public(
                    format!("{}:/usr/bin:/bin:/opt/homebrew/bin", self.root.display()).into(),
                ),
            ),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
            (
                "SKIPPY_PARITY_MANIFEST".into(),
                Value::Public(self.manifest.clone().into()),
            ),
            (
                "SKIPPY_PARITY_MODEL_MANIFEST".into(),
                Value::Public(self.registry.clone().into()),
            ),
            (
                "TRACE".into(),
                Value::Public(self.root.join("trace").into()),
            ),
            (
                "SNAPSHOT".into(),
                Value::Public(self.snapshot.parent().unwrap().into()),
            ),
            ("HF_STATUS".into(), Value::Public(status.to_string().into())),
        ]);
        let report = process::supervise(
            &ProcessSpec {
                executable,
                cwd: self.root.clone(),
                arguments,
                environment,
            },
            &Limits {
                execution: std::time::Duration::from_secs(10),
                graceful_shutdown: std::time::Duration::from_secs(3),
                forced_shutdown: std::time::Duration::from_secs(2),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert!(
            report.cleanup.complete && !report.stdout.truncated && !report.stderr.truncated,
            "{report:?}"
        );
        Output {
            status: report.status.expect("finite fixture exited"),
            stdout: report.stdout.bytes_retained,
            stderr: report.stderr.bytes_retained,
        }
    }
    fn trace(&self) -> Vec<String> {
        fs::read_to_string(self.root.join("trace"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }
    fn registered(&self, cadences: Json) -> Json {
        let row = self.candidates()["candidates"][0].clone();
        let revision = row["revision"].as_str().unwrap();
        let names = row["include"].as_array().unwrap();
        json!({"manifest_kind":"test-model-artifacts","artifacts":[{"id":"vision-artifact","repo":"registered/model","revision":revision,"selector":"Q4_K_M","model_ref":"registered/model:Q4_K_M","cadences":cadences,"files":names,"file_integrity":row["file_integrity"],"urls":names.iter().map(|n|format!("https://huggingface.co/registered/model/resolve/{revision}/{}",n.as_str().unwrap())).collect::<Vec<_>>()}]})
    }
}
#[test]
fn actual_shell_pins_and_verifies_both_nested_model_and_projector_files() {
    let fixture = Fixture::new();
    let output = fixture.run(&["--priority", "p0"], 0);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let args = fixture.trace();
    assert_eq!(&args[..2], ["download", "owner/model"]);
    assert!(args.iter().any(|a| a == "nested/model.gguf"));
    assert!(args.iter().any(|a| a == "nested/projector.gguf"));
    assert!(
        args.windows(2)
            .any(|pair| pair == ["--revision", "0123456789abcdef0123456789abcdef01234567"])
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("verified immutable test artifact:"));
    assert!(stdout.contains("model.gguf"));
    assert!(stdout.contains("projector.gguf"));
    assert!(stdout.contains("download complete: 1 targets, 0 skipped"));
    // HF can return one nested file instead of its snapshot directory. A root
    // file sharing that basename must not steal the selected nested suffix.
    let root_model = fixture.snapshot.parent().unwrap().join("model.gguf");
    let bytes = b"different-root-model";
    fs::write(&root_model, bytes).unwrap();
    let mut candidates = fixture.candidates();
    candidates["candidates"][0]["include"]
        .as_array_mut()
        .unwrap()
        .insert(0, json!("model.gguf"));
    candidates["candidates"][0]["file_integrity"]["model.gguf"] =
        json!({"size_bytes":bytes.len(),"blob_id":hex::encode(Sha256::digest(bytes))});
    fixture.save(&candidates, &json!({"artifacts":[]}));
    fs::write(
        &fixture.hf,
        "#!/bin/sh\nprintf 'path=%s/nested/model.gguf\\n' \"$SNAPSHOT\"\n",
    )
    .unwrap();
    let nested_file = fixture.run(&[], 0);
    assert!(
        nested_file.status.success(),
        "{}",
        String::from_utf8_lossy(&nested_file.stderr)
    );
    let verified = String::from_utf8_lossy(&nested_file.stdout);
    assert_eq!(
        verified
            .matches("verified immutable test artifact:")
            .count(),
        3
    );
    assert!(verified.contains(&format!(
        "verified immutable test artifact: {}",
        root_model.display()
    )));
}
#[test]
fn actual_shell_registered_manual_admission_precedes_any_download_and_overrides_row_targets() {
    let fixture = Fixture::new();
    fs::remove_file(&fixture.manifest).unwrap();
    let help = fixture.typed(&["--help"]);
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("--cadence manual"));
    assert!(!fixture.root.join("trace").exists());
    fixture.save(&fixture.candidates(), &json!({"artifacts":[]}));
    for cadence in [&[][..], &["--cadence", "pull-request"][..]] {
        let refused = fixture.typed(cadence);
        assert_eq!(refused.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&refused.stderr).contains("cadence"));
        assert!(!fixture.root.join("trace").exists());
    }
    assert!(fixture.typed(&["--cadence", "manual"]).status.success());
    assert!(!fixture.root.join("trace").exists());
    for manual in [false, true] {
        let fixture = Fixture::new();
        let mut row = fixture.candidates();
        row["candidates"][0]["artifact_id"] = json!("vision-artifact");
        row["candidates"][0]["repo"] = json!("ignored/stale");
        fixture.save(
            &row,
            &fixture.registered(if manual {
                json!(["manual"])
            } else {
                json!(["pull-request"])
            }),
        );
        let output = fixture.run(&[], 0);
        assert_eq!(output.status.success(), manual);
        if manual {
            assert_eq!(fixture.trace()[1], "registered/model");
        } else {
            assert!(!fixture.root.join("trace").exists());
        }
    }
}
#[test]
fn actual_shell_bad_model_or_projector_integrity_never_publishes_a_verified_row() {
    for (name, same_size) in [
        ("model.gguf", false),
        ("model.gguf", true),
        ("projector.gguf", false),
        ("projector.gguf", true),
    ] {
        let fixture = Fixture::new();
        let projector = fixture.snapshot.join(name);
        let mut bytes = fs::read(&projector).unwrap();
        if same_size {
            bytes[0] ^= 1;
        } else {
            bytes.push(0);
        }
        fs::write(projector, bytes).unwrap();
        let output = fixture.run(&[], 0);
        assert!(!output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!stdout.contains("verified immutable test artifact:"));
        assert!(!stdout.contains("download complete:"));
        assert!(fixture.root.join("trace").exists());
    }
}
#[test]
fn actual_shell_refuses_bad_pin_or_missing_registered_artifact_before_process() {
    for field in ["revision", "artifact_id", "include"] {
        let fixture = Fixture::new();
        let mut data = fixture.candidates();
        data["candidates"][0][field] = match field {
            "revision" => json!("main"),
            "artifact_id" => json!("absent"),
            _ => json!(["../outside.gguf"]),
        };
        fixture.save(&data, &json!({"artifacts":[]}));
        let output = fixture.run(&[], 0);
        assert!(!output.status.success());
        assert!(!fixture.root.join("trace").exists());
    }
}
#[test]
fn actual_shell_filters_priorities_and_status_and_dry_run_never_downloads() {
    let fixture = Fixture::new();
    for args in [
        &["--status", "certified"][..],
        &["--priority", "p1"][..],
        &["--dry-run"][..],
    ] {
        let output = fixture.run(args, 0);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!fixture.root.join("trace").exists());
    }
}
#[test]
fn actual_shell_retains_best_effort_failure_without_immutable_success_claim() {
    let fixture = Fixture::new();
    let output = fixture.run(&[], 7);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("1 targets, 1 skipped"));
    assert!(!stdout.contains("verified immutable test artifact:"));
}

#[test]
fn actual_shell_preserves_unpinned_pattern_discovery_and_family_priority_override() {
    let fixture = Fixture::new();
    let data = json!({"support_priority":{"p0":{"llama_models":["vision"]},"p1":{"families":["vision-family"]}},"candidates":[{"llama_model":"vision","family":"vision-family","status":"needs_candidate","repo":"owner/discovery","include":["Q4/*.gguf","projector*.gguf"]},{"llama_model":"missing","family":"vision-family","status":"needs_candidate"}]});
    fixture.save(&data, &json!({"artifacts":[]}));
    let filtered = fixture.run(&["--priority", "p0"], 0);
    assert!(filtered.status.success());
    assert!(!fixture.root.join("trace").exists());
    let selected = fixture.run(&["--priority", "p1"], 0);
    assert!(selected.status.success());
    let args = fixture.trace();
    assert!(
        args.windows(2)
            .any(|pair| pair == ["--include", "Q4/*.gguf"])
    );
    assert!(
        args.windows(2)
            .any(|pair| pair == ["--include", "projector*.gguf"])
    );
    assert!(!args.iter().any(|arg| arg == "--revision"));
    let stdout = String::from_utf8_lossy(&selected.stdout);
    assert!(stdout.contains("missing target:"));
    assert!(!stdout.contains("verified immutable test artifact:"));
}
#[test]
fn actual_shell_admits_all_selected_rows_before_any_external_download() {
    let fixture = Fixture::new();
    let mut data = fixture.candidates();
    let mut invalid = data["candidates"][0].clone();
    invalid["artifact_id"] = json!("absent");
    data["candidates"].as_array_mut().unwrap().push(invalid);
    fixture.save(&data, &json!({"artifacts":[]}));
    let refused = fixture.run(&[], 0);
    assert!(!refused.status.success());
    assert!(!fixture.root.join("trace").exists());
}
#[test]
fn actual_shell_deadline_refuses_completion_and_cleans_owned_download_descendant() {
    let fixture = Fixture::new();
    fs::write(
        &fixture.hf,
        "#!/bin/sh\ntrap '' TERM\n/bin/sleep 30 &\nprintf '%s\\n' \"$!\" > \"$TRACE\"\nwait\n",
    )
    .unwrap();
    let failed = fixture.run(&[], 0);
    assert!(!failed.status.success());
    assert!(!String::from_utf8_lossy(&failed.stdout).contains("download complete:"));
    let pid = fs::read_to_string(fixture.root.join("trace"))
        .unwrap()
        .trim()
        .parse::<i32>()
        .unwrap();
    assert!(pid > 1);
    let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        if unsafe { libc::kill(pid, 0) } == -1
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
        {
            break;
        }
        assert!(
            std::time::Instant::now() < until,
            "owned download descendant remains"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn actual_registered_exact_file_admission_refuses_options_patterns_controls_and_duplicates_before_hf()
 {
    for names in [
        vec!["--revision"],
        vec!["nested/*.gguf"],
        vec!["nested/model?.gguf"],
        vec!["nested/[ab].gguf"],
        vec!["nested/model\t.gguf"],
        vec!["nested/model\u{7f}.gguf"],
        vec!["nested/model.gguf", "nested/model.gguf"],
    ] {
        let fixture = Fixture::new();
        let mut candidates = fixture.candidates();
        let mut invalid = candidates["candidates"][0].clone();
        invalid["artifact_id"] = json!("vision-artifact");
        candidates["candidates"]
            .as_array_mut()
            .unwrap()
            .push(invalid);
        let mut registry = fixture.registered(json!(["manual"]));
        let row = &mut registry["artifacts"][0];
        let original = row["file_integrity"]["nested/model.gguf"].clone();
        row["files"] = json!(names);
        row["file_integrity"] = Json::Object(
            names
                .iter()
                .map(|name| (name.to_string(), original.clone()))
                .collect(),
        );
        row["urls"] = json!(names.iter().map(|name| format!("https://huggingface.co/registered/model/resolve/0123456789abcdef0123456789abcdef01234567/{name}")).collect::<Vec<_>>());
        fixture.save(&candidates, &registry);
        let output = fixture.run(&[], 0);
        assert!(!output.status.success(), "{names:?}");
        assert!(!fixture.root.join("trace").exists());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!stdout.contains("verified immutable test artifact:"));
        assert!(!stdout.contains("parity download complete:"));
    }
}

#[test]
fn actual_shell_verification_accepts_regular_hf_snapshot_symlinks_for_model_and_projector() {
    let fixture = Fixture::new();
    for name in ["model.gguf", "projector.gguf"] {
        let original = fixture.snapshot.join(name);
        let blob = fixture.root.join(format!("blob-{name}"));
        fs::rename(&original, &blob).unwrap();
        std::os::unix::fs::symlink(blob, original).unwrap();
    }
    let output = fixture.run(&[], 0);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.matches("verified immutable test artifact:").count(),
        2
    );
    assert!(stdout.contains("parity download complete: 1 targets, 0 skipped"));
}
