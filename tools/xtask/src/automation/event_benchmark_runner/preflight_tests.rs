use super::super::{
    identity_worker::{BinaryIdentity, Evidence, FileIdentity, ModelMetadata},
    worker_frontends::Receipt,
};
use super::*;
use sha2::{Digest, Sha256};
use std::fs;
#[cfg(unix)]
fn executable(path: &Path, text: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
#[cfg(unix)]
fn fixture(
    verifier_failure: bool,
    bad_correlation: bool,
) -> (
    tempfile::TempDir,
    options::Command,
    PathBuf,
    PathBuf,
    PathBuf,
) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let binary = root.join("binary");
    executable(
        &binary,
        "#!/bin/sh\ncase \"$*\" in --version) printf 'fixture version\\n';; '-g therm') printf 'fixture thermal\\n';; *) exit 9;; esac\n",
    );
    let model = root.join("model.gguf");
    fs::write(&model, b"mock model boundary").unwrap();
    let package = root.join("native-runtimes/runtime");
    fs::create_dir_all(&package).unwrap();
    fs::write(package.join("manifest.json"), b"mock manifest boundary").unwrap();
    let command = options::Command::parse(
        &[
            "--binary",
            binary.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--output-dir",
            root.to_str().unwrap(),
            "--pairs-primary",
            "1",
            "--pairs-scenario",
            "1",
            "--seed",
            "42",
            "--mode",
            "production",
            "--mode",
            "event-disabled",
            "--scenario",
            "fixture",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>(),
    )
    .unwrap();
    let request = IdentityRequest::Admit {
        input: identity_worker::Input {
            schema_version: 1,
            binaries: [binary.clone(), binary.clone()],
            model: model.clone(),
            minimum_context_tokens: 192,
        },
    };
    let record = || BinaryIdentity {
        file: FileIdentity {
            path: binary.clone(),
            sha256: hex::encode(Sha256::digest(fs::read(&binary).unwrap())),
            bytes: fs::metadata(&binary).unwrap().len(),
        },
        adjacent_runtime_root: root.join("native-runtimes"),
        runtime_validation: "pending_owning_native_package_and_loader_policy".into(),
    };
    let model_digest = hex::encode(Sha256::digest(b"mock model boundary"));
    let identity = Evidence {
        schema_version: 1,
        binaries: [record(), record()],
        model: FileIdentity {
            path: model,
            sha256: model_digest.clone(),
            bytes: 19,
        },
        model_metadata: ModelMetadata {
            sha256: model_digest,
            architecture: "mock".into(),
            native_context_tokens: 1024,
        },
    };
    let data = IdentityData::Admitted {
        identity: Box::new(identity),
        runtime_packages: [vec![package.clone()], vec![package]],
        thermal_state: serde_json::json!({"available":false}),
    };
    let receipt = root.join("fixture-receipt");
    evidence_io::publish(
        &receipt,
        &Receipt {
            schema_version: 1,
            request_sha256: if bad_correlation {
                "0".repeat(64)
            } else {
                worker_frontends::request_sha256(&request).unwrap()
            },
            data,
        },
        worker_frontends::IDENTITY_BYTES,
    )
    .unwrap();
    let tool = root.join("tool");
    let calls = root.join("calls");
    executable(
        &tool,
        &format!(
            "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$1 $2 $3\" >> '{}'\ncase \"$1 $2 $3\" in\n'automation event-benchmark-run identity-worker') [ \"$4\" = --input ] && [ \"$6\" = --output ]; cp '{}' \"$7\";;\n'native verify-runtime-package --portable') exit {};;\n*) exit 91;;\nesac\n",
            calls.display(),
            receipt.display(),
            if verifier_failure { 7 } else { 0 }
        ),
    );
    let phase = root.join("phase");
    fs::create_dir(&phase).unwrap();
    (directory, command, phase, tool, binary)
}
#[cfg(unix)]
#[test]
fn parent_admission_runs_exact_worker_and_deduplicated_native_policy_with_separate_logs() {
    let (_dir, command, phase, tool, pmset) = fixture(false, false);
    let prepared = prepare_with_tool(
        &command,
        &phase,
        Duration::from_secs(10),
        &Cancellation::default(),
        &tool,
        &pmset,
    )
    .unwrap();
    assert!(prepared.metadata.runtime_packages_verified);
    assert_eq!(
        prepared.metadata.binaries[0].version.as_deref(),
        Some("fixture version")
    );
    assert_eq!(
        fs::read_to_string(phase.parent().unwrap().join("calls")).unwrap(),
        "automation event-benchmark-run identity-worker\nnative verify-runtime-package --portable\n"
    );
    assert!(phase.join("identity-worker.stdout.log").is_file());
    assert!(phase.join("runtime-package-1.stderr.log").is_file());
    assert!(!phase.join("runtime-package-2.stdout.log").exists());
}
#[cfg(unix)]
#[test]
fn native_verifier_failure_never_promotes_runtime_package_admission() {
    let (_dir, command, phase, tool, pmset) = fixture(true, false);
    assert!(
        prepare_with_tool(
            &command,
            &phase,
            Duration::from_secs(10),
            &Cancellation::default(),
            &tool,
            &pmset
        )
        .is_err()
    );
    assert!(phase.join("runtime-package-1.stderr.log").is_file());
}
#[cfg(unix)]
#[test]
fn mismatched_identity_receipt_stops_before_runtime_verification() {
    let (_dir, command, phase, tool, pmset) = fixture(false, true);
    assert!(
        prepare_with_tool(
            &command,
            &phase,
            Duration::from_secs(10),
            &Cancellation::default(),
            &tool,
            &pmset
        )
        .is_err()
    );
    assert!(!phase.join("runtime-package-1.stdout.log").exists());
}
#[cfg(unix)]
#[test]
fn cancelled_or_expired_parent_budget_never_launches_worker() {
    for cancelled in [true, false] {
        let (_dir, command, phase, tool, pmset) = fixture(false, false);
        let cancel = Cancellation::default();
        if cancelled {
            cancel.cancel();
        }
        assert!(
            prepare_with_tool(
                &command,
                &phase,
                if cancelled {
                    Duration::from_secs(10)
                } else {
                    Duration::ZERO
                },
                &cancel,
                &tool,
                &pmset
            )
            .is_err()
        );
        assert!(!phase.parent().unwrap().join("calls").exists());
    }
}

#[test]
fn preflight_reserves_grace_force_and_separate_eof_drain_before_launch() {
    assert!(
        Deadline::new(Duration::from_millis(750))
            .unwrap()
            .execution()
            .is_err()
    );
    assert!(
        Deadline::new(Duration::from_millis(500))
            .unwrap()
            .execution()
            .is_err()
    );
    let execution = Deadline::new(Duration::from_secs(2))
        .unwrap()
        .execution()
        .unwrap();
    assert!(execution <= Duration::from_millis(1250));
}
