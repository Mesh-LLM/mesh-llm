use super::*;
use crate::{
    automation::canary_receipts::Digest,
    process::{
        self as supervisor, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions,
        Readiness,
    },
};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

fn isolated(name: &str) -> bool {
    if std::env::var_os("VERIFY_SOURCE_FIXTURE_CHILD").is_some() {
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
                "VERIFY_SOURCE_FIXTURE_CHILD".into(),
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
struct Fixture {
    _temp: tempfile::TempDir,
    authority: VerificationSource,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let controller = temp.path().join("controller");
        fs::create_dir_all(controller.join("src")).unwrap();
        fs::create_dir_all(controller.join(".github/workflows")).unwrap();
        fs::write(controller.join(".gitignore"), "target/\n.deps/\n").unwrap();
        fs::write(controller.join("src/lib.rs"), "base\n").unwrap();
        fs::write(controller.join(".github/workflows/guard.yml"), "trusted\n").unwrap();
        process::text(&controller, &["init", "--quiet"]).unwrap();
        let base = commit(&controller);
        fs::write(controller.join("src/lib.rs"), "candidate\n").unwrap();
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
        fs::write(
            controller.join("controller-note.txt"),
            "trusted controller advanced independently\n",
        )
        .unwrap();
        let revision = commit(&controller);
        let executable_sha256 =
            frozen_verifier::executable_digest(&std::env::current_exe().unwrap()).unwrap();
        Self {
            _temp: temp,
            authority: VerificationSource {
                controller: FrozenVerifier {
                    root: controller.canonicalize().unwrap(),
                    revision,
                    executable_sha256,
                },
                root: root.canonicalize().unwrap(),
                base,
                candidate,
                tree,
            },
        }
    }
    fn json(&self) -> Value {
        json!({"authority":{"controller":{"root":self.authority.controller.root,"revision":self.authority.controller.revision,
            "executable_sha256":self.authority.controller.executable_sha256},"root":self.authority.root,"base":self.authority.base,
            "candidate":self.authority.candidate,"tree":self.authority.tree}})
    }
}
#[test]
fn exact_direct_child_is_independent_of_advanced_trusted_controller_and_ignored_build_outputs() {
    if isolated(
        "exact_direct_child_is_independent_of_advanced_trusted_controller_and_ignored_build_outputs",
    ) {
        return;
    }
    let fixture = Fixture::new();
    assert_ne!(
        fixture.authority.base,
        fixture.authority.controller.revision
    );
    fs::create_dir(fixture.authority.root.join("target")).unwrap();
    fs::write(fixture.authority.root.join("target/build.log"), "output\n").unwrap();
    let value =
        process::operation(|| execute(&serde_json::to_vec(&fixture.json()).unwrap())).unwrap();
    assert_eq!(value["status"], "verification_source_admitted");
    assert_eq!(value["candidate"], fixture.authority.candidate);
    assert_eq!(value["tree"], fixture.authority.tree);
    assert!(value.get("run_id").is_none());
    assert!(value.get("certified").is_none());
}
#[test]
fn wrong_head_base_tree_controller_revision_digest_and_shared_root_are_rejected() {
    if isolated("wrong_head_base_tree_controller_revision_digest_and_shared_root_are_rejected") {
        return;
    }
    let fixture = Fixture::new();
    for field in [
        "base",
        "candidate",
        "tree",
        "controller-revision",
        "controller-digest",
        "shared-root",
        "subdirectory",
    ] {
        let mut authority = fixture.authority.clone();
        match field {
            "base" => authority.base = authority.controller.revision.clone(),
            "candidate" => authority.candidate = authority.base.clone(),
            "tree" => authority.tree = "0".repeat(40),
            "controller-revision" => authority.controller.revision = authority.base.clone(),
            "controller-digest" => {
                authority.controller.executable_sha256 = Digest::try_from("f".repeat(64)).unwrap()
            }
            "shared-root" => authority.root = authority.controller.root.clone(),
            "subdirectory" => authority.root = authority.root.join("src"),
            _ => unreachable!(),
        }
        assert!(
            process::operation(|| authority.validate()).is_err(),
            "admitted {field}"
        );
    }
}
#[test]
fn tracked_staged_and_untracked_candidate_changes_and_controller_changes_are_rejected() {
    if isolated(
        "tracked_staged_and_untracked_candidate_changes_and_controller_changes_are_rejected",
    ) {
        return;
    }
    let fixture = Fixture::new();
    let root = &fixture.authority.root;
    fs::write(root.join("src/lib.rs"), "mutated\n").unwrap();
    assert!(process::operation(|| fixture.authority.validate()).is_err());
    process::text(root, &["add", "src/lib.rs"]).unwrap();
    assert!(process::operation(|| fixture.authority.validate()).is_err());
    process::text(root, &["reset", "--hard", "--quiet", "HEAD"]).unwrap();
    fs::write(root.join("untracked-source.rs"), "mutated\n").unwrap();
    assert!(process::operation(|| fixture.authority.validate()).is_err());
    fs::remove_file(root.join("untracked-source.rs")).unwrap();
    fs::write(
        fixture.authority.controller.root.join("src/lib.rs"),
        "mutable verifier\n",
    )
    .unwrap();
    assert!(process::operation(|| fixture.authority.validate()).is_err());
}
#[test]
fn multi_commit_merge_and_protected_orchestration_candidates_are_rejected() {
    if isolated("multi_commit_merge_and_protected_orchestration_candidates_are_rejected") {
        return;
    }
    let fixture = Fixture::new();
    let mut authority = fixture.authority.clone();
    fs::write(authority.root.join("src/lib.rs"), "second coding commit\n").unwrap();
    authority.candidate = commit(&authority.root);
    authority.tree = process::text(&authority.root, &["rev-parse", "HEAD^{tree}"]).unwrap();
    assert!(process::operation(|| authority.validate()).is_err());
    let merge = process::text(
        &authority.root,
        &[
            "-c",
            "user.name=Verification Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit-tree",
            &authority.tree,
            "-p",
            &authority.candidate,
            "-p",
            &authority.base,
            "-m",
            "merge fixture",
        ],
    )
    .unwrap();
    process::text(
        &authority.root,
        &["checkout", "--quiet", "--detach", &merge],
    )
    .unwrap();
    authority.candidate = merge;
    assert!(process::operation(|| authority.validate()).is_err());
    process::text(
        &authority.root,
        &["checkout", "--quiet", "--detach", &authority.base],
    )
    .unwrap();
    fs::write(
        authority.root.join(".github/workflows/guard.yml"),
        "changed gate\n",
    )
    .unwrap();
    authority.candidate = commit(&authority.root);
    authority.tree = process::text(&authority.root, &["rev-parse", "HEAD^{tree}"]).unwrap();
    assert!(process::operation(|| authority.validate()).is_err());
}
#[test]
fn supplied_workflow_authority_and_unknown_candidate_fields_are_refused() {
    if isolated("supplied_workflow_authority_and_unknown_candidate_fields_are_refused") {
        return;
    }
    let fixture = Fixture::new();
    let mut value = fixture.json();
    value["authority"]["run_id"] = json!("123");
    assert!(execute(&serde_json::to_vec(&value).unwrap()).is_err());
    value = fixture.json();
    value["authority"]["controller"]["selected_source"] = json!(fixture.authority.candidate);
    assert!(execute(&serde_json::to_vec(&value).unwrap()).is_err());
}

#[test]
fn hidden_or_skipped_tracked_candidate_and_controller_entries_cannot_admit_source() {
    if isolated("hidden_or_skipped_tracked_candidate_and_controller_entries_cannot_admit_source") {
        return;
    }
    for candidate in [true, false] {
        for flag in ["--assume-unchanged", "--skip-worktree"] {
            let fixture = Fixture::new();
            let root = if candidate {
                &fixture.authority.root
            } else {
                &fixture.authority.controller.root
            };
            let path = root.join("src/lib.rs");
            let original = fs::read(&path).unwrap();
            process::text(root, &["update-index", flag, "--", "src/lib.rs"]).unwrap();
            fs::write(&path, "uncommitted hidden source\n").unwrap();
            // These ordinary checks alone are the exact original admission gap.
            process::text(root, &["diff", "--cached", "--exit-code", "HEAD", "--"]).unwrap();
            process::text(root, &["diff", "--exit-code", "HEAD", "--"]).unwrap();
            assert!(
                process::operation(|| fixture.authority.validate()).is_err(),
                "{flag} candidate={candidate}"
            );
            fs::write(path, original).unwrap();
            let reset = if flag == "--assume-unchanged" {
                "--no-assume-unchanged"
            } else {
                "--no-skip-worktree"
            };
            process::text(root, &["update-index", reset, "--", "src/lib.rs"]).unwrap();
            assert!(
                process::operation(|| fixture.authority.validate()).is_ok(),
                "clean source restored {flag}"
            );
        }
    }
}
