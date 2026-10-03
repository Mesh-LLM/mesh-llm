use super::*;
use crate::process::{
    self as supervisor, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness,
};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};
fn isolated(name: &str) -> bool {
    if std::env::var_os("VERIFY_INSPECTION_FIXTURE_CHILD").is_some() {
        return false;
    }
    let target = format!("{}::{name}", module_path!().split_once("::").unwrap().1);
    let output = supervisor::supervise_raw(
        &ProcessSpec {
            executable: std::env::current_exe().unwrap(),
            cwd: std::env::current_dir().unwrap(),
            arguments: ["--exact", &target, "--nocapture", "--test-threads=1"]
                .into_iter()
                .map(|a| crate::process::Value::Public(a.into()))
                .collect(),
            environment: BTreeMap::from([(
                "VERIFY_INSPECTION_FIXTURE_CHILD".into(),
                crate::process::Value::Public("1".into()),
            )]),
        },
        &Limits {
            execution: Duration::from_secs(90),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(output.process.success(), "{:?}", output.process);
    assert!(
        String::from_utf8_lossy(output.stdout.unwrap().as_bytes()).contains("1 passed; 0 failed")
    );
    true
}
fn commit(root: &Path) -> String {
    process::text(root, &["add", "-A"]).unwrap();
    process::text(
        root,
        &[
            "-c",
            "user.name=Verification Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    )
    .unwrap();
    process::text(root, &["rev-parse", "HEAD"]).unwrap()
}
fn policy_source(root: &Path) {
    for dir in [
        "third_party/llama.cpp/patches/model_support",
        "third_party/llama.cpp/patches/generated",
        "ci/llama-canary",
        "docs/skippy",
        "crates/skippy-ffi/src",
        "crates/mesh-llm-host-runtime/src/inference/skippy",
    ] {
        fs::create_dir_all(root.join(dir)).unwrap();
    }
    fs::write(root.join(".gitignore"), ".deps/\ntarget/\n").unwrap();
    fs::write(
        root.join("third_party/llama.cpp/patches/0001-fixture.patch"),
        "fixture",
    )
    .unwrap();
    for (lane, name) in [
        ("model_support", "0001-model.patch"),
        ("generated", "0001-family-fixture.patch"),
    ] {
        fs::write(
            root.join(format!("third_party/llama.cpp/patches/{lane}/series")),
            format!("{name}\n"),
        )
        .unwrap();
        fs::write(
            root.join(format!("third_party/llama.cpp/patches/{lane}/{name}")),
            "fixture",
        )
        .unwrap();
    }
    fs::write(
        root.join("third_party/llama.cpp/upstream.txt"),
        format!("{}\n", "a".repeat(40)),
    )
    .unwrap();
    fs::write(root.join("crates/skippy-ffi/src/lib.rs"), "pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_PATCH: u32 = 3;\n").unwrap();
    let family = json!({"policy":{"profiles":{"full":{"status":"certified","required_lanes":["single-step","chain","state-handoff"]}}},"models":[{"family":"fixture","class":"causal_generation","architecture":"fixture","profile":"full","resources":{"estimated_model_bytes":1}}]});
    fs::write(
        root.join("ci/llama-canary/family-certified.json"),
        serde_json::to_vec(&family).unwrap(),
    )
    .unwrap();
    fs::write(root.join("docs/skippy/llama-parity-candidates.json"), serde_json::to_vec(&json!({"candidates":[{"llama_model":"fixture","family":"fixture","status":"needs_candidate"}]})).unwrap()).unwrap();
}
fn prepared(root: &Path) {
    let native = root.join(".deps/llama.cpp");
    fs::create_dir_all(native.join("src/models")).unwrap();
    fs::write(
        native.join("src/models/fixture.cpp"),
        "begin_block(layer); end_block(layer);\n",
    )
    .unwrap();
    process::text(&native, &["init", "--quiet"]).unwrap();
    let head = commit(&native);
    for (name, value) in [
        (".mesh-llm-upstream-sha", "a".repeat(40)),
        (".mesh-llm-patched-sha", head),
        (".mesh-llm-prepare-schema", "5".into()),
        (
            ".mesh-llm-patch-digest",
            source::patch_digest(&root.join("third_party/llama.cpp/patches"))
                .unwrap()
                .as_str()
                .to_owned(),
        ),
    ] {
        fs::write(native.join(name), format!("{value}\n")).unwrap();
    }
}
struct Fixture {
    _temp: tempfile::TempDir,
    authority: VerificationSource,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let controller = temp.path().join("controller");
        policy_source(&controller);
        process::text(&controller, &["init", "--quiet"]).unwrap();
        let base = commit(&controller);
        let path = controller.join("ci/llama-canary/family-certified.json");
        let mut family: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        family["models"][0]["resources"]["estimated_model_bytes"] = json!(2);
        fs::write(&path, serde_json::to_vec(&family).unwrap()).unwrap();
        super::super::split_roster::admit(&controller, false).unwrap();
        let candidate = commit(&controller);
        let tree = process::text(&controller, &["rev-parse", "HEAD^{tree}"]).unwrap();
        let root = temp.path().join("candidate");
        process::git(
            &controller,
            &[
                "worktree".into(),
                "add".into(),
                "--quiet".into(),
                "--detach".into(),
                root.clone().into(),
                candidate.clone().into(),
            ],
            None,
        )
        .unwrap();
        process::text(&controller, &["checkout", "--quiet", "--detach", &base]).unwrap();
        fs::write(controller.join("trusted-note.txt"), "controller advanced\n").unwrap();
        let revision = commit(&controller);
        prepared(&root);
        Self {
            _temp: temp,
            authority: VerificationSource {
                controller: FrozenVerifier {
                    root: controller.canonicalize().unwrap(),
                    revision,
                    executable_sha256: frozen_verifier::executable_digest(
                        &std::env::current_exe().unwrap(),
                    )
                    .unwrap(),
                },
                root: root.canonicalize().unwrap(),
                base,
                candidate,
                tree,
            },
        }
    }
    fn document(&self) -> Value {
        json!({"authority":{"controller":{"root":self.authority.controller.root,"revision":self.authority.controller.revision,"executable_sha256":self.authority.controller.executable_sha256},
            "root":self.authority.root,"base":self.authority.base,"candidate":self.authority.candidate,"tree":self.authority.tree}})
    }
    fn inspect(&self, verb: &str) -> DynResult<Value> {
        inspect(&serde_json::to_vec(&self.document())?, verb)
    }
    fn snapshot(&mut self) {
        let root = &self.authority.root;
        process::text(root, &["add", "-A"]).unwrap();
        self.authority.tree = process::text(root, &["write-tree"]).unwrap();
        self.authority.candidate = process::text(
            root,
            &[
                "-c",
                "user.name=Verification Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit-tree",
                &self.authority.tree,
                "-p",
                &self.authority.base,
                "-m",
                "exact candidate fixture",
            ],
        )
        .unwrap();
        process::text(
            root,
            &["reset", "--hard", "--quiet", &self.authority.candidate],
        )
        .unwrap();
    }
}
#[test]
fn independent_inspections_reuse_base_blob_policy_candidate_parity_and_read_only_roster() {
    if isolated(
        "independent_inspections_reuse_base_blob_policy_candidate_parity_and_read_only_roster",
    ) {
        return;
    }
    process::operation(|| {
        let f = Fixture::new();
        assert_ne!(f.authority.controller.revision, f.authority.base);
        assert_eq!(
            f.inspect("verification-manifest-policy")?["status"],
            "agent_manifest_policy_admitted"
        );
        assert_eq!(
            f.inspect("verification-parity-inventory")?["model_sources"],
            1
        );
        let path = f
            .authority
            .root
            .join("crates/mesh-llm-host-runtime/src/inference/skippy/split-certified.json");
        let bytes = fs::read(&path)?;
        let result = f.inspect("verification-split-roster-check")?;
        assert_eq!(result["check"], true);
        assert_eq!(fs::read(path)?, bytes);
        f.authority.validate()?;
        Ok(())
    })
    .unwrap();
}
#[test]
fn content_refusals_remain_failures_for_clean_exact_snapshots() {
    if isolated("content_refusals_remain_failures_for_clean_exact_snapshots") {
        return;
    }
    process::operation(|| {
        let mut f = Fixture::new();
        let path = f
            .authority
            .root
            .join("ci/llama-canary/family-certified.json");
        let original = fs::read(&path)?;
        let mut family: Value = serde_json::from_slice(&original)?;
        family["models"][0]["family"] = json!("changed artifact authority");
        fs::write(&path, serde_json::to_vec(&family)?)?;
        f.snapshot();
        f.authority.validate()?;
        assert!(f.inspect("verification-manifest-policy").is_err());
        fs::write(&path, original)?;
        let parity = f
            .authority
            .root
            .join("docs/skippy/llama-parity-candidates.json");
        fs::write(&parity, b"{\"candidates\":[]}")?;
        f.snapshot();
        f.authority.validate()?;
        assert!(f.inspect("verification-parity-inventory").is_err());
        Ok(())
    })
    .unwrap();
}
#[test]
fn stale_roster_is_refused_without_repair_and_write_mode_or_workflow_schema_is_not_accepted() {
    if isolated(
        "stale_roster_is_refused_without_repair_and_write_mode_or_workflow_schema_is_not_accepted",
    ) {
        return;
    }
    process::operation(|| {
        let mut f = Fixture::new();
        let path = f
            .authority
            .root
            .join("crates/mesh-llm-host-runtime/src/inference/skippy/split-certified.json");
        fs::write(&path, b"stale roster")?;
        f.snapshot();
        f.authority.validate()?;
        assert!(
            f.inspect("verification-split-roster-check")
                .unwrap_err()
                .to_string()
                .contains("stale")
        );
        assert_eq!(fs::read(&path)?, b"stale roster");
        for key in ["check", "context", "run_id"] {
            let mut document = f.document();
            document[key] = json!(false);
            assert!(
                inspect(
                    &serde_json::to_vec(&document)?,
                    "verification-split-roster-check"
                )
                .is_err()
            );
        }
        Ok(())
    })
    .unwrap();
}
#[test]
fn independent_inspection_checks_prepared_recipe_and_rejects_dirty_source_or_cancelled_scope() {
    if isolated(
        "independent_inspection_checks_prepared_recipe_and_rejects_dirty_source_or_cancelled_scope",
    ) {
        return;
    }
    let f = Fixture::new();
    process::operation(|| {
        fs::write(
            f.authority
                .root
                .join(".deps/llama.cpp/.mesh-llm-patched-sha"),
            format!("{}\n", "0".repeat(40)),
        )?;
        assert!(f.inspect("verification-parity-inventory").is_err());
        fs::write(f.authority.root.join("untracked-source.rs"), "dirty source")?;
        assert!(f.inspect("verification-split-roster-check").is_err());
        fs::remove_file(f.authority.root.join("untracked-source.rs"))?;
        Ok(())
    })
    .unwrap();
    #[cfg(unix)]
    assert!(
        process::operation(|| {
            unsafe {
                libc::raise(libc::SIGTERM);
            }
            assert!(f.inspect("verification-manifest-policy").is_err());
            Ok(())
        })
        .is_err()
    );
}
