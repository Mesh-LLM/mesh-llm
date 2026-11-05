//! Authored filesystem/Git fixtures. Never execute packaged native payloads.
use super::{
    archive, archive_extract, archive_write, candidate_view, combine, packing, process,
    producer_receipt::{self, Context},
    restore_transaction::Publication,
    restoring, source, tests, workload,
};

#[test]
fn combined_native_package_keeps_one_candidate_identity() {
    if isolated("combined_native_package_keeps_one_candidate_identity") {
        return;
    }
    for (mac_attempt, linux_attempt, current_attempt) in
        [(1, 1, 1), (1, 2, 2), (2, 1, 2), (1, 3, 2), (3, 1, 2)]
    {
        let mut fixture = Fixture::new();
        fixture.context.run_attempt = mac_attempt.to_string();
        let (package, _) = fixture.pack();
        let mac = native_producer(&fixture, &package, "macos-arm64-metal", mac_attempt);
        let linux = native_producer(&fixture, &package, "linux-arm64-cuda", linux_attempt);
        let output = fixture.directory.path().join("combined-package");
        let mut context = fixture.context.clone();
        context.run_attempt = current_attempt.to_string();
        let result = combine::run(&combine::Input {
            context: context.clone(),
            mac_package: mac.clone(),
            mac_identity_sha256: Digest::of_file(&mac.join("identity.json")).unwrap(),
            linux_package: linux.clone(),
            linux_identity_sha256: Digest::of_file(&linux.join("identity.json")).unwrap(),
            output: output.clone(),
        });
        if linux_attempt > current_attempt || mac_attempt > current_attempt {
            assert!(result.is_err());
            assert!(!output.exists());
            continue;
        }
        let result = result.unwrap();
        assert_eq!(result["candidate"], fixture.base);
        let combined: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("identity.json")).unwrap()).unwrap();
        assert_eq!(combined["run_attempt"], mac_attempt.to_string());
        assert!(output.join("linux-binaries.tar").is_file());
        assert!(output.join("linux-workload-oracles.tar").is_file());
        let consumer = fixture.consumer();
        restoring::restore(&restoring::Input {
            context,
            root: consumer.clone(),
            package: output,
            identity_sha256: Digest::try_from(
                result["identity_sha256"].as_str().unwrap().to_owned(),
            )
            .unwrap(),
        })
        .unwrap();
        assert_eq!(
            fs::read(consumer.join("target/debug/skippy")).unwrap(),
            tests::native_executable_bytes()
        );
        assert_eq!(
            fs::read(consumer.join(".deps/canary-workload-oracles/cargo/debug/skippy")).unwrap(),
            tests::native_executable_bytes()
        );
    }
}

fn native_producer(fixture: &Fixture, package: &Path, platform: &str, attempt: u32) -> PathBuf {
    let output = fixture.directory.path().join(platform);
    fs::create_dir(&output).unwrap();
    for entry in fs::read_dir(package).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), output.join(entry.file_name())).unwrap();
    }
    let binary = if platform == "linux-arm64-cuda" {
        tests::elf_bytes()
    } else {
        tests::executable_bytes("/usr/lib/libSystem.B.dylib")
    };
    fs::remove_file(output.join("binaries.tar")).unwrap();
    archive_write::write(
        &output.join("binaries.tar"),
        archive::BINARIES
            .into_iter()
            .map(|name| archive_write::Input {
                name: name.into(),
                content: archive_write::Content::Bytes(binary.clone()),
                digest: Digest::of_bytes(&binary),
                executable: true,
            })
            .collect(),
    )
    .unwrap();
    let mut workloads =
        tests::workload_tar_for(&fixture.base, &fixture.native, 200, &binary, |_| {});
    workloads.extend_from_slice(&[0; 1024]);
    fs::write(output.join("workload-oracles.tar"), workloads).unwrap();
    let linux_bundle = output.join("llama-source.bundle");
    fs::remove_file(&linux_bundle).unwrap();
    process::git(
        &fixture.root.join(".deps/llama.cpp"),
        &[
            "bundle".into(),
            "create".into(),
            linux_bundle.clone().into(),
            "HEAD".into(),
        ],
        None,
    )
    .unwrap();
    let mut identity: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("identity.json")).unwrap()).unwrap();
    identity["platform"] = json!(platform);
    identity["run_attempt"] = json!(attempt.to_string());
    identity["binaries_sha256"] = json!(Digest::of_file(&output.join("binaries.tar")).unwrap());
    identity["workload_oracles_sha256"] =
        json!(Digest::of_file(&output.join("workload-oracles.tar")).unwrap());
    identity["llama_bundle_sha256"] = json!(Digest::of_file(&linux_bundle).unwrap());
    fs::write(
        output.join("identity.json"),
        serde_json::to_vec_pretty(&identity).unwrap(),
    )
    .unwrap();
    output
}
use crate::automation::canary_receipts::Digest;
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

// Separate harness processes prevent process-global signal scope overlap in a full suite.
fn isolated(name: &str) -> bool {
    if std::env::var_os("CANARY_PACKAGE_FIXTURE_CHILD").is_some() {
        return false;
    }
    let module = module_path!()
        .split_once("::")
        .map_or(module_path!(), |(_, path)| path);
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("{module}::{name}"),
            "--nocapture",
            "--test-threads=1",
        ])
        .env("CANARY_PACKAGE_FIXTURE_CHILD", "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "fixture child failed: {output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
        "fixture did not execute: {output:?}"
    );
    true
}

fn commit(root: &Path, message: &str) -> String {
    process::text(root, &["add", "-A"]).unwrap();
    process::text(
        root,
        &[
            "-c",
            "user.name=Package Fixture",
            "-c",
            "user.email=package-fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            message,
        ],
    )
    .unwrap();
    process::text(root, &["rev-parse", "HEAD"]).unwrap()
}

struct Fixture {
    directory: tempfile::TempDir,
    root: PathBuf,
    context: Context,
    base: String,
    native: String,
    closure: PathBuf,
    admitted: PathBuf,
    test_build: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("controller");
        fs::create_dir(&root).unwrap();
        process::text(&root, &["init", "--quiet"]).unwrap();
        fs::write(root.join(".gitignore"), ".deps/\ntarget/\ncanary-source/\n").unwrap();
        let patches = root.join("skippy/llama_cpp/patches");
        fs::create_dir_all(&patches).unwrap();
        fs::write(patches.join("0001-core.patch"), b"authored recipe fixture").unwrap();
        let upstream = "a".repeat(40);
        fs::write(root.join("skippy/llama_cpp/upstream.txt"), &upstream).unwrap();
        fs::create_dir_all(root.join("ci/llama-canary")).unwrap();
        fs::write(
            root.join("ci/llama-canary/family-certified.json"),
            b"final fixture manifest",
        )
        .unwrap();
        let base = commit(&root, "authored controller source");
        let native_root = root.join(".deps/llama.cpp");
        fs::create_dir_all(&native_root).unwrap();
        process::text(&native_root, &["init", "--quiet"]).unwrap();
        fs::write(
            native_root.join("fixture.cpp"),
            b"int fixture() { return 7; }\n",
        )
        .unwrap();
        let native = commit(&native_root, "authored native tree");
        for (name, value) in BTreeMap::from([
            (".mesh-llm-upstream-sha", upstream),
            (".mesh-llm-patched-sha", native.clone()),
            (
                ".mesh-llm-patch-digest",
                source::patch_digest(&patches).unwrap().as_str().to_owned(),
            ),
            (".mesh-llm-prepare-schema", "5".into()),
        ]) {
            fs::write(native_root.join(name), format!("{value}\n")).unwrap();
        }
        let archive_path = directory.path().join("authored-workloads.tar");
        let mut bytes = tests::workload_tar(&base, &native, 200, |_| {});
        bytes.extend_from_slice(&[0; 1024]);
        fs::write(&archive_path, bytes).unwrap();
        let closure = directory.path().join("closure");
        archive_extract::extract(&archive_path, &closure).unwrap();
        archive_extract::normalize_workload(&closure).unwrap();
        fs::create_dir_all(root.join("target/debug/deps")).unwrap();
        let binary = fs::read(closure.join("cargo/debug/skippy")).unwrap();
        for name in archive::BINARIES {
            let path = if name == "skippy-mm-test" {
                root.join("target/debug/deps/skippy_serving-fixture")
            } else {
                root.join("target/debug").join(name)
            };
            fs::write(&path, &binary).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
            }
        }
        let test_build = directory.path().join("test-build.jsonl");
        fs::write(&test_build,serde_json::to_vec(&json!({"reason":"compiler-artifact","target":{"name":"skippy_serving"},"profile":{"test":true},"executable":root.join("target/debug/deps/skippy_serving-fixture")})).unwrap()).unwrap();
        let admitted = directory.path().join("admitted");
        fs::create_dir(&admitted).unwrap();
        let plan=serde_json::to_vec(&json!({"selected_models":[{"family":"fixture","class":"causal_generation","certification_lanes":[],"artifact":{"files":["weights.gguf"],"file_integrity":{"weights.gguf":{"size_bytes":1}}},"resources":{"estimated_model_bytes":1}}],"github_matrix":{"include":[{"id":"shard-0","shard_index":0,"families":"fixture","estimated_work_bytes":1}]},"shards":[{"shard_index":0,"families":["fixture"]}],"required_certification_lanes":["single-step","chain","state-handoff"]})).unwrap();
        fs::write(admitted.join("plan.json"), &plan).unwrap();
        fs::write(admitted.join("source-plan.json"),serde_json::to_vec(&json!({"schema":1,"controller_revision":base,"selected_revision":base,"manifest_sha256":Digest::of_file(&root.join("ci/llama-canary/family-certified.json")).unwrap(),"plan_sha256":Digest::of_bytes(&plan),"cache_admission":"gguf_metadata","gguf_admission":"metadata_admitted"})).unwrap()).unwrap();
        let context = Context {
            controller_root: root.clone(),
            controller_revision: base.clone(),
            selected_source: String::new(),
            run_id: "100".into(),
            run_attempt: "1".into(),
        };
        Self {
            directory,
            root,
            context,
            base,
            native,
            closure,
            admitted,
            test_build,
        }
    }
    fn pack(&self) -> (PathBuf, Digest) {
        self.pack_candidate(&self.base, None)
    }
    fn pack_candidate(&self, candidate: &str, bundle: Option<PathBuf>) -> (PathBuf, Digest) {
        let receipt = self.directory.path().join("producer-receipt.json");
        let digest = producer_receipt::write(&producer_receipt::Input {
            context: self.context.clone(),
            root: self.root.clone(),
            closure: self.closure.clone(),
            output: receipt.clone(),
        })
        .unwrap();
        let summary = self.directory.path().join("summary.md");
        fs::write(&summary, b"authored summary\n").unwrap();
        let output = self.directory.path().join("package");
        let result = packing::pack(&packing::Input {
            context: self.context.clone(),
            root: self.root.clone(),
            output: output.clone(),
            candidate: candidate.to_owned(),
            base: self.base.clone(),
            branch: "llama-canary/repair-fixture".into(),
            pass_id: "repair-1".into(),
            mode: if bundle.is_some() {
                "repair-build"
            } else {
                "pinned-build"
            }
            .into(),
            test_build: self.test_build.clone(),
            bundle,
            summary,
            workload_oracles: self.closure.clone(),
            admitted_plan: self.admitted.clone(),
            admitted_identity_sha256: Digest::of_file(&self.admitted.join("source-plan.json"))
                .unwrap(),
            producer_receipt: receipt,
            producer_receipt_sha256: digest,
        })
        .unwrap();
        (
            output,
            Digest::try_from(result["identity_sha256"].as_str().unwrap().to_owned()).unwrap(),
        )
    }
    fn consumer(&self) -> PathBuf {
        let target = self.context.controller_root.join("canary-source");
        process::git(
            &self.root,
            &[
                "clone".into(),
                "--quiet".into(),
                "--no-hardlinks".into(),
                self.root.clone().into(),
                target.clone().into(),
            ],
            None,
        )
        .unwrap();
        target
    }
}

#[test]
fn full_pack_restore_consumes_immutable_plan_and_real_closure_bytes_without_execution() {
    if isolated(
        "full_pack_restore_consumes_immutable_plan_and_real_closure_bytes_without_execution",
    ) {
        return;
    }
    let fixture = Fixture::new();
    let (package, digest) = fixture.pack();
    let root = fixture.consumer();
    let plan = fs::read(package.join("plan.json")).unwrap();
    let result = restoring::restore(&restoring::Input {
        context: fixture.context.clone(),
        root: root.clone(),
        package: package.clone(),
        identity_sha256: digest,
    })
    .unwrap();
    assert_eq!(result["candidate"], fixture.base);
    assert_eq!(fs::read(package.join("plan.json")).unwrap(), plan);
    assert_eq!(
        process::text(&fixture.root, &["rev-parse", "HEAD"]).unwrap(),
        fixture.base
    );
    assert_eq!(source::prepared(&root).unwrap().head, fixture.native);
    assert_eq!(
        fs::read(root.join("target/debug/skippy")).unwrap(),
        fs::read(fixture.root.join("target/debug/skippy")).unwrap()
    );
    assert!(
        workload::verify_producer(
            &root,
            &root.join(".deps/canary-workload-oracles"),
            &fixture.directory.path().join("verified.diff"),
            &fixture.native
        )
        .is_ok()
    );
    // An already populated consumer is not a fresh restore role.
    let identity = Digest::of_file(&package.join("identity.json")).unwrap();
    assert!(
        restoring::restore(&restoring::Input {
            context: fixture.context.clone(),
            root,
            package,
            identity_sha256: identity
        })
        .is_err()
    );
}

#[test]
fn corrupt_package_and_foreign_receipt_leave_fresh_consumer_untouched() {
    if isolated("corrupt_package_and_foreign_receipt_leave_fresh_consumer_untouched") {
        return;
    }
    let fixture = Fixture::new();
    let (package, digest) = fixture.pack();
    let root = fixture.consumer();
    let path = package.join("binaries.tar");
    let mut bytes = fs::read(&path).unwrap();
    bytes[512] ^= 1;
    fs::write(path, bytes).unwrap();
    assert!(
        restoring::restore(&restoring::Input {
            context: fixture.context.clone(),
            root: root.clone(),
            package,
            identity_sha256: digest
        })
        .is_err()
    );
    assert_eq!(
        process::text(&root, &["rev-parse", "HEAD"]).unwrap(),
        fixture.base
    );
    assert!(!root.join(".deps").exists());
    assert!(!root.join("target").exists());
    let receipt = fixture.directory.path().join("foreign.json");
    fs::write(&receipt, b"foreign receipt").unwrap();
    assert!(
        producer_receipt::consume(
            &receipt,
            &Digest::of_bytes(b"frozen receipt"),
            &fixture.context,
            &fixture.root,
            &fixture.closure,
            &fixture.base
        )
        .is_err()
    );
}

#[test]
fn atomic_publication_rolls_back_on_post_swap_rejection_and_cancellation() {
    if isolated("atomic_publication_rolls_back_on_post_swap_rejection_and_cancellation") {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("consumer");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("identity"), b"old").unwrap();
    for cancel in [false, true] {
        let result = process::operation(|| {
            let stage = candidate_view::Owned::new(directory.path(), "transaction-fixture")?;
            fs::create_dir(stage.path.join("source"))?;
            fs::write(stage.path.join("source/identity"), b"new")?;
            let mut transaction = Publication::new(&root, stage)?;
            transaction.publish()?;
            assert_eq!(fs::read(root.join("identity"))?, b"new");
            if cancel {
                process::cancellation().cancel();
                transaction.commit()
            } else {
                transaction.rollback()?;
                Err("authored post-publication rejection".into())
            }
        });
        assert!(result.is_err());
        assert_eq!(fs::read(root.join("identity")).unwrap(), b"old");
    }
}

#[test]
fn controller_receipt_survives_staging_but_rejects_replaced_source_or_producer() {
    if isolated("controller_receipt_survives_staging_but_rejects_replaced_source_or_producer") {
        return;
    }
    let fixture = Fixture::new();
    fs::write(
        fixture.root.join("new-source.rs"),
        b"pub fn fixture() -> u8 { 7 }\n",
    )
    .unwrap();
    let source =
        workload::source_identity(&fixture.root, &fixture.directory.path().join("dirty.diff"))
            .unwrap();
    let producer = fixture.closure.join("producer.json");
    let mut document: serde_json::Value =
        serde_json::from_slice(&fs::read(&producer).unwrap()).unwrap();
    document["source"] = serde_json::to_value(source).unwrap();
    fs::write(&producer, serde_json::to_vec(&document).unwrap()).unwrap();
    let receipt = fixture.directory.path().join("dirty-receipt.json");
    let digest = producer_receipt::write(&producer_receipt::Input {
        context: fixture.context.clone(),
        root: fixture.root.clone(),
        closure: fixture.closure.clone(),
        output: receipt.clone(),
    })
    .unwrap();
    process::text(&fixture.root, &["add", "-A"]).unwrap();
    let tree = process::text(&fixture.root, &["write-tree"]).unwrap();
    let candidate = process::text(
        &fixture.root,
        &[
            "-c",
            "user.name=Package Fixture",
            "-c",
            "user.email=package-fixture@example.invalid",
            "commit-tree",
            &tree,
            "-p",
            &fixture.base,
            "-m",
            "authored candidate snapshot",
        ],
    )
    .unwrap();
    let sealed = producer_receipt::consume(
        &receipt,
        &digest,
        &fixture.context,
        &fixture.root,
        &fixture.closure,
        &candidate,
    )
    .unwrap();
    let sealed: serde_json::Value = serde_json::from_slice(&sealed).unwrap();
    assert_eq!(sealed["source"]["head"], candidate);
    assert_eq!(
        sealed["source"]["worktree_sha256"],
        Digest::of_bytes(b"").as_str()
    );
    fs::write(
        fixture.root.join("new-source.rs"),
        b"pub fn fixture() -> u8 { 9 }\n",
    )
    .unwrap();
    assert!(
        producer_receipt::consume(
            &receipt,
            &digest,
            &fixture.context,
            &fixture.root,
            &fixture.closure,
            &candidate
        )
        .is_err()
    );
    fs::write(
        fixture.root.join("new-source.rs"),
        b"pub fn fixture() -> u8 { 7 }\n",
    )
    .unwrap();
    fs::write(producer, b"replaced producer").unwrap();
    assert!(
        producer_receipt::consume(
            &receipt,
            &digest,
            &fixture.context,
            &fixture.root,
            &fixture.closure,
            &candidate
        )
        .is_err()
    );
}

fn changed_package(fixture: &Fixture) -> (PathBuf, Digest, String) {
    let relative = "candidate.txt";
    fs::write(
        fixture.root.join(relative),
        b"exact candidate source bytes\n",
    )
    .unwrap();
    process::text(&fixture.root, &["add", relative]).unwrap();
    let tree = process::text(&fixture.root, &["write-tree"]).unwrap();
    let candidate = process::text(
        &fixture.root,
        &[
            "-c",
            "user.name=Package Fixture",
            "-c",
            "user.email=package-fixture@example.invalid",
            "commit-tree",
            &tree,
            "-p",
            &fixture.base,
            "-m",
            "finite changed candidate",
        ],
    )
    .unwrap();
    let source_identity = workload::source_identity(
        &fixture.root,
        &fixture.directory.path().join("candidate-source.diff"),
    )
    .unwrap();
    let producer = fixture.closure.join("producer.json");
    let mut document: serde_json::Value =
        serde_json::from_slice(&fs::read(&producer).unwrap()).unwrap();
    document["source"] = serde_json::to_value(source_identity).unwrap();
    fs::write(producer, serde_json::to_vec(&document).unwrap()).unwrap();
    let receipt = fixture.admitted.join("source-plan.json");
    let mut document: serde_json::Value =
        serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
    document["selected_revision"] = candidate.clone().into();
    fs::write(receipt, serde_json::to_vec(&document).unwrap()).unwrap();
    let branch = "refs/heads/llama-canary/repair-fixture";
    process::text(&fixture.root, &["update-ref", branch, &candidate]).unwrap();
    let bundle = fixture.directory.path().join("candidate.bundle");
    process::git(
        &fixture.root,
        &[
            "bundle".into(),
            "create".into(),
            bundle.clone().into(),
            branch.into(),
            format!("^{}", fixture.base).into(),
        ],
        None,
    )
    .unwrap();
    let (package, digest) = fixture.pack_candidate(&candidate, Some(bundle));
    (package, digest, candidate)
}

#[test]
fn selected_upgrade_restores_changed_source_with_a_separate_frozen_controller() {
    if isolated("selected_upgrade_restores_changed_source_with_a_separate_frozen_controller") {
        return;
    }
    let mut fixture = Fixture::new();
    let controller = fixture.directory.path().join("protected-controller");
    process::git(
        &fixture.root,
        &[
            "clone".into(),
            "--quiet".into(),
            fixture.root.clone().into(),
            controller.clone().into(),
        ],
        None,
    )
    .unwrap();
    fs::write(controller.join("controller.txt"), b"protected controller").unwrap();
    let revision = commit(&controller, "separate controller revision");
    fixture.context.controller_root = controller.clone();
    fixture.context.controller_revision = revision.clone();
    fixture.context.selected_source = fixture.base.clone();
    let receipt = fixture.admitted.join("source-plan.json");
    let mut document: serde_json::Value =
        serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
    document["controller_revision"] = revision.clone().into();
    fs::write(receipt, serde_json::to_vec(&document).unwrap()).unwrap();
    let (package, digest, candidate) = changed_package(&fixture);
    let root = fixture.consumer();
    let request = restoring::Input {
        context: fixture.context.clone(),
        root: root.clone(),
        package,
        identity_sha256: digest,
    };
    let mut foreign = request.context.clone();
    foreign.selected_source = "f".repeat(40);
    assert!(
        restoring::restore(&restoring::Input {
            context: foreign,
            root: request.root.clone(),
            package: request.package.clone(),
            identity_sha256: request.identity_sha256.clone()
        })
        .is_err()
    );
    restoring::restore(&request).unwrap();
    assert_eq!(
        process::text(&root, &["rev-parse", "HEAD"]).unwrap(),
        candidate
    );
    assert_eq!(
        process::text(&controller, &["rev-parse", "HEAD"]).unwrap(),
        revision
    );
    assert_eq!(
        process::text(&root, &["rev-parse", "HEAD^"]).unwrap(),
        fixture.base
    );
}

#[test]
fn changed_candidate_bundle_restores_exact_detached_identity_and_preserves_controller() {
    if isolated(
        "changed_candidate_bundle_restores_exact_detached_identity_and_preserves_controller",
    ) {
        return;
    }
    let fixture = Fixture::new();
    let (package, digest, candidate) = changed_package(&fixture);
    let root = fixture.consumer();
    assert_eq!(
        process::text(&root, &["rev-parse", "HEAD"]).unwrap(),
        fixture.base
    );
    let result = restoring::restore(&restoring::Input {
        context: fixture.context.clone(),
        root: root.clone(),
        package,
        identity_sha256: digest,
    })
    .unwrap();
    assert_eq!(result["candidate"], candidate);
    assert_eq!(
        process::text(&root, &["rev-parse", "HEAD"]).unwrap(),
        candidate
    );
    assert_eq!(
        process::text(&root, &["rev-parse", "--abbrev-ref", "HEAD"]).unwrap(),
        "HEAD"
    );
    assert_eq!(
        fs::read(root.join("candidate.txt")).unwrap(),
        b"exact candidate source bytes\n"
    );
    assert_eq!(
        process::text(&root, &["status", "--porcelain", "--untracked-files=no"]).unwrap(),
        ""
    );
    assert_eq!(
        process::text(&fixture.root, &["rev-parse", "HEAD"]).unwrap(),
        fixture.base
    );
    assert_eq!(source::prepared(&root).unwrap().head, fixture.native);
    assert!(
        workload::verify_producer(
            &root,
            &root.join(".deps/canary-workload-oracles"),
            &fixture.directory.path().join("restored-source.diff"),
            &fixture.native
        )
        .is_ok()
    );
}

#[test]
fn digest_bound_protected_candidate_bundle_refuses_before_consumer_publication() {
    if isolated("digest_bound_protected_candidate_bundle_refuses_before_consumer_publication") {
        return;
    }
    let fixture = Fixture::new();
    let (package, _, _) = changed_package(&fixture);
    process::text(&fixture.root, &["reset", "--hard", &fixture.base]).unwrap();
    fs::create_dir(fixture.root.join("scripts")).unwrap();
    fs::write(
        fixture.root.join("scripts/control.sh"),
        b"protected candidate payload\n",
    )
    .unwrap();
    let candidate = commit(&fixture.root, "finite protected source change");
    process::text(
        &fixture.root,
        &[
            "update-ref",
            "refs/heads/llama-canary/repair-fixture",
            &candidate,
        ],
    )
    .unwrap();
    let bundle = package.join("candidate.bundle");
    fs::remove_file(&bundle).unwrap();
    process::git(
        &fixture.root,
        &[
            "bundle".into(),
            "create".into(),
            bundle.clone().into(),
            "refs/heads/llama-canary/repair-fixture".into(),
            format!("^{}", fixture.base).into(),
        ],
        None,
    )
    .unwrap();
    process::text(
        &fixture.root,
        &["checkout", "--quiet", "--detach", &fixture.base],
    )
    .unwrap();
    let identity_path = package.join("identity.json");
    let mut identity: serde_json::Value =
        serde_json::from_slice(&fs::read(&identity_path).unwrap()).unwrap();
    identity["candidate"] = candidate.into();
    identity["bundle_sha256"] = serde_json::to_value(Digest::of_file(&bundle).unwrap()).unwrap();
    let bytes = serde_json::to_vec(&identity).unwrap();
    fs::write(identity_path, &bytes).unwrap();
    let root = fixture.consumer();
    let result = restoring::restore(&restoring::Input {
        context: fixture.context.clone(),
        root: root.clone(),
        package,
        identity_sha256: Digest::of_bytes(&bytes),
    });
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("protected orchestration")
    );
    assert_eq!(
        process::text(&root, &["rev-parse", "HEAD"]).unwrap(),
        fixture.base
    );
    assert!(!root.join("scripts/control.sh").exists());
    assert!(!root.join("candidate.txt").exists());
    assert!(!root.join("target").exists());
    assert!(!root.join(".deps").exists());
    assert!(
        process::text(&root, &["status", "--porcelain"])
            .unwrap()
            .is_empty()
    );
}

#[path = "handoff_sealing_tests.rs"]
mod handoff_sealing_tests;
