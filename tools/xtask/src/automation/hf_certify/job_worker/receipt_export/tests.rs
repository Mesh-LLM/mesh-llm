use super::*;
fn input() -> Input {
    let root = std::env::current_dir().unwrap().canonicalize().unwrap();
    Input {
        schema_version: 1,
        repo: "fixture/evidence".into(),
        parent_commit: "a".repeat(40),
        artifact: Artifact {
            path: root.join("native-job.json"),
            path_in_repo: "runs/native-job.json".into(),
            sha256: "b".repeat(64),
            byte_size: 10,
        },
        receipt_request_sha256: "c".repeat(64),
        credential_file: root.join("credential"),
        execution_timeout_ms: 1000,
    }
}
fn receipt(input: &Input, hash: &str) -> Value {
    json!({"schema_version":1,"request_sha256":hash,"status":"PUBLISHED_REGULAR_RECEIPT","receipt_request_sha256":input.receipt_request_sha256,"artifact_sha256":input.artifact.sha256,"input_custody_verified":true,"error":null,"publication":{"schema_version":1,"repo":input.repo,"parent_commit":input.parent_commit,"commit_oid":"d".repeat(40),"completed":true,"source_custody_verified":true,"mutation_attempted":true,"remote_verified_paths":[input.artifact.path_in_repo],"error":null}})
}
#[test]
fn regular_export_correlates_exact_typed_request_and_rejects_contradictory_success() {
    let input = input();
    let hash = admission::digest(&serde_json::to_vec(&input).unwrap());
    let value = receipt(&input, &hash);
    assert_eq!(
        correlated(&project(&value, &input), &input, &hash).unwrap(),
        "d".repeat(40)
    );
    for key in [
        "error",
        "request_sha256",
        "artifact_sha256",
        "input_custody_verified",
    ] {
        let mut changed = value.clone();
        changed[key] = json!("private-untrusted-diagnostic");
        assert!(correlated(&project(&changed, &input), &input, &hash).is_err());
        assert!(
            !serde_json::to_string(&project(&changed, &input))
                .unwrap()
                .contains("private-untrusted-diagnostic")
        );
    }
    for key in [
        "error",
        "completed",
        "source_custody_verified",
        "commit_oid",
        "remote_verified_paths",
    ] {
        let mut changed = value.clone();
        changed["publication"][key] = json!("private-untrusted-diagnostic");
        assert!(correlated(&project(&changed, &input), &input, &hash).is_err());
        assert!(
            !serde_json::to_string(&project(&changed, &input))
                .unwrap()
                .contains("private-untrusted-diagnostic")
        );
    }
}
#[test]
fn regular_export_closed_destination_and_exhausted_budget_refuse_before_child() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let config = Config {
        export_budget_secs: 60,
        helper: admission::Artifact {
            path: base.join("missing-helper"),
            sha256: "a".repeat(64),
        },
        helper_source: admission::Artifact {
            path: base.join("missing-source"),
            sha256: "b".repeat(64),
        },
        repo: "fixture/evidence".into(),
        parent_commit: "c".repeat(40),
        credential_file: Some(base.join("credential")),
        path_in_repo: "runs/native-job.json".into(),
        credential_environment: false,
    };
    config.validate().unwrap();
    let mut evidence = json!({});
    assert!(
        execute(
            &config,
            &base.join("missing-native"),
            &base,
            Instant::now(),
            &Cancellation::default(),
            None,
            &mut evidence
        )
        .is_err()
    );
    assert!(!base.join("receipt-export-input.json").exists());
    let mut changed = config;
    changed.path_in_repo = "../native-job.json".into();
    assert!(changed.validate().is_err());
    root.close().unwrap();
}

#[test]
fn regular_export_progress_is_correlated_bounded_and_never_terminal_success() {
    let root = tempfile::tempdir().unwrap();
    let input = input();
    let hash = admission::digest(&serde_json::to_vec(&input).unwrap());
    let path = root.path().join("progress.json");
    let mut value = receipt(&input, &hash);
    value["status"] = json!("IN_PROGRESS");
    value["input_custody_verified"] = json!(false);
    value["publication"]["completed"] = json!(false);
    value["publication"]["source_custody_verified"] = json!(false);
    value["publication"]["remote_verified_paths"] = json!([]);
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let observed = observe(&path, &input, &hash, true).unwrap().unwrap();
    assert_eq!(observed["publication"]["commit_oid"], "d".repeat(40));
    assert!(correlated(&observed, &input, &hash).is_err());
    for (outer, key) in [
        (true, "request_sha256"),
        (false, "completed"),
        (false, "remote_verified_paths"),
    ] {
        let mut changed = value.clone();
        if outer {
            changed[key] = json!("private-untrusted-error");
        } else {
            changed["publication"][key] = json!("private-untrusted-error");
        }
        std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(observe(&path, &input, &hash, true).is_err());
    }
    std::fs::remove_file(&path).unwrap();
    assert!(observe(&path, &input, &hash, true).unwrap().is_none());
    std::fs::create_dir(&path).unwrap();
    assert!(observe(&path, &input, &hash, true).is_err());
    root.close().unwrap();
}
