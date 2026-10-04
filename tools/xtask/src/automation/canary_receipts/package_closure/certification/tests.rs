use super::{
    host_lock,
    observation::{self, Host},
};

#[test]
fn vm_stat_admission_counts_only_reclaimable_pages_and_preserves_reserve() {
    let text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages speculative: 30.\nPages purgeable: 500.\nPages wired down: 1000.\nPages occupied by compressor: 2000.\n";
    assert_eq!(observation::available(text).unwrap(), 60 * 16384);
    assert!(observation::available("page size of 4096 bytes\nPages free: 1.\n").is_err());
    assert!(observation::available(&format!("{text}Pages free: 3.\n")).is_err());
    assert_eq!(
        observation::admission(
            Host {
                total: 1000,
                available: 900
            },
            800
        )
        .unwrap(),
        100
    );
    assert!(
        observation::admission(
            Host {
                total: 1000,
                available: 899
            },
            800
        )
        .is_err()
    );
    assert!(
        observation::admission(
            Host {
                total: 1000,
                available: 1000
            },
            901
        )
        .is_err()
    );
    assert!(
        observation::admission(
            Host {
                total: 0,
                available: 0
            },
            1
        )
        .is_err()
    );
}

#[cfg(unix)]
#[test]
fn stable_lock_rejects_symlink_hardlink_and_runner_writable_directory() {
    use std::{
        fs,
        os::unix::fs::{MetadataExt, PermissionsExt, symlink},
    };
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("locks");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o750)).unwrap();
    let uid = fs::metadata(&root).unwrap().uid();
    let leaf = root.join("mesh-canary-family-host.lock");
    fs::write(&leaf, b"").unwrap();
    fs::set_permissions(&leaf, fs::Permissions::from_mode(0o666)).unwrap();
    assert!(host_lock::open(&root, uid).is_ok());
    let alias = directory.path().join("alias");
    symlink(&root, &alias).unwrap();
    assert!(host_lock::open(&alias, uid).is_err());
    let second = root.join("hard-link");
    fs::hard_link(&leaf, &second).unwrap();
    assert!(host_lock::open(&root, uid).is_err());
    fs::remove_file(second).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o770)).unwrap();
    assert!(host_lock::open(&root, uid).is_err());
    fs::set_permissions(&root, fs::Permissions::from_mode(0o750)).unwrap();
    fs::remove_file(&leaf).unwrap();
    symlink(directory.path().join("external"), &leaf).unwrap();
    assert!(host_lock::open(&root, uid).is_err());
}

fn wrong_tier_fixture() -> (tempfile::TempDir, super::Input) {
    use crate::automation::canary_receipts::Digest;
    use serde_json::json;
    use std::fs;
    let directory = tempfile::tempdir().unwrap();
    let controller = directory.path().join("controller");
    fs::create_dir(&controller).unwrap();
    super::super::process::text(&controller, &["init", "--quiet"]).unwrap();
    super::super::process::text(
        &controller,
        &[
            "-c",
            "user.name=Finite Memory",
            "-c",
            "user.email=finite-memory@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "finite controller",
        ],
    )
    .unwrap();
    let revision = super::super::process::text(&controller, &["rev-parse", "HEAD"]).unwrap();
    let consumer = controller.join("canary-source");
    fs::create_dir(&consumer).unwrap();
    let package = directory.path().join("package");
    fs::create_dir(&package).unwrap();
    let plan = json!({"selected_models":[{"family":"fixture","class":"causal_generation","certification_lanes":[],"artifact":{"files":["model.gguf"],"file_integrity":{"model.gguf":{"size_bytes":1}}},"resources":{"estimated_model_bytes":1}}],"github_matrix":{"include":[{"id":"shard-0","shard_index":0,"families":"fixture","estimated_work_bytes":1}]},"shards":[{"shard_index":0,"families":["fixture"]}],"required_certification_lanes":["chain","single-step","state-handoff"]});
    fs::write(
        package.join("plan.json"),
        serde_json::to_vec(&plan).unwrap(),
    )
    .unwrap();
    let mut identity = json!({"schema":3,"platform":"macos-arm64-metal","candidate":revision,"base":revision,"controller":revision,"mesh_source":"","branch":"llama-canary/repair-finite","pass_id":"repair-1","run_id":"123","run_attempt":"1","bundle_sha256":null,"plan_sha256":Digest::of_file(&package.join("plan.json")).unwrap()});
    for (file, field) in [
        ("binaries.tar", "binaries_sha256"),
        ("workload-oracles.tar", "workload_oracles_sha256"),
        ("llama-source.bundle", "llama_bundle_sha256"),
        ("llama-source.json", "llama_provenance_sha256"),
    ] {
        fs::write(package.join(file), b"private never-restored payload").unwrap();
        identity[field] = json!(Digest::of_file(&package.join(file)).unwrap());
    }
    let bytes = serde_json::to_vec(&identity).unwrap();
    fs::write(package.join("identity.json"), &bytes).unwrap();
    let input=serde_json::from_value(json!({"context":{"controller_root":controller,"controller_revision":revision,"selected_source":"","run_id":"123","run_attempt":"1"},"root":consumer,"package":package,"identity_sha256":Digest::of_bytes(&bytes),"shard_index":0,"memory_tier":"accelerator-memory-256plus","evidence":directory.path().join("evidence"),"max_seconds":3})).unwrap();
    (directory, input)
}
#[test]
fn actual_certification_wrong_tier_refuses_before_consumer_native_lock_or_child_work() {
    let (directory, input) = wrong_tier_fixture();
    let result = super::execute(&input);
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("scheduled memory tier differs")
    );
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("evidence/memory-admission.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["status"], "failed");
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("scheduled memory tier differs")
    );
    assert!(report.get("host_lock_contended").is_none());
    assert!(report.get("physical_bytes").is_none());
    assert!(
        !directory
            .path()
            .join("controller/canary-source/target")
            .exists()
    );
    assert!(
        !directory
            .path()
            .join("evidence/certify.stdout.log")
            .exists()
    );
}

#[cfg(unix)]
#[test]
fn actual_battery_admission_refuses_insufficient_host_before_child_or_output_files() {
    use crate::process::{ProcessSpec, Value};
    use serde_json::json;
    use std::{
        collections::BTreeMap,
        time::{Duration, Instant},
    };
    for (host, peak) in [
        (
            Host {
                total: 1000,
                available: 899,
            },
            800,
        ),
        (
            Host {
                total: 1000,
                available: 1000,
            },
            901,
        ),
        (
            Host {
                total: 0,
                available: 0,
            },
            1,
        ),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("child-started");
        let spec = ProcessSpec {
            executable: "/bin/sh".into(),
            arguments: ["-c", "printf started > child-started"]
                .into_iter()
                .map(|v| Value::Public(v.into()))
                .collect(),
            cwd: directory.path().to_owned(),
            environment: BTreeMap::new(),
        };
        let mut report = json!({"status":"failed"});
        let result = super::run_admitted_battery(
            &spec,
            Instant::now() + Duration::from_secs(2),
            host,
            peak,
            &mut report,
            directory.path(),
        );
        assert!(result.is_err(), "insufficient host admitted");
        assert!(!marker.exists(), "battery started before host admission");
        assert!(!directory.path().join("certify.stdout.log").exists());
        assert!(!directory.path().join("certify.stderr.log").exists());
        assert_eq!(report["status"], "failed");
    }
}
