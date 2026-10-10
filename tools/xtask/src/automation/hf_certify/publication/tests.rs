use super::*;
use contract::{Artifact, PublisherInput};
#[cfg(unix)]
fn request(root: &Path, mode: &str) -> Request {
    let file = root.join("model.gguf");
    let bytes = format!("GGUF{mode}").into_bytes();
    std::fs::write(&file, &bytes).unwrap();
    let source = root.join("source.txt");
    std::fs::write(&source, b"fixture source").unwrap();
    let debug = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let helper = debug.join("examples/l7_daemon_fixture");
    assert!(helper.is_file());
    let pin = |path: &Path| super::super::admission::Artifact {
        path: path.into(),
        sha256: admission::digest(&std::fs::read(path).unwrap()),
    };
    Request {
        helper: pin(&helper),
        helper_source: pin(&source),
        input: PublisherInput {
            schema_version: 1,
            repo: "fixture/model".into(),
            parent_commit: "a".repeat(40),
            shards: vec![Artifact {
                path: file,
                path_in_repo: "model.gguf".into(),
                sha256: admission::digest(&bytes),
                byte_size: bytes.len() as u64,
            }],
            sidecars: vec![],
            credential_file: Some(root.join("explicit-credential-reference")),
            execution_timeout_ms: 30000,
        },
    }
}
#[test]
fn publication_declared_private_credential_and_closed_input_are_required_before_launch() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let helper = base.join("helper");
    std::fs::write(&helper, b"inert not launched").unwrap();
    let artifact = super::super::admission::Artifact {
        path: helper,
        sha256: "a".repeat(64),
    };
    let mut request = Request {
        helper: artifact.clone(),
        helper_source: artifact,
        input: PublisherInput {
            schema_version: 1,
            repo: "fixture/model".into(),
            parent_commit: "a".repeat(40),
            shards: vec![Artifact {
                path: base.join("model.gguf"),
                path_in_repo: "model.gguf".into(),
                sha256: "b".repeat(64),
                byte_size: 4,
            }],
            sidecars: vec![],
            credential_file: None,
            execution_timeout_ms: 1000,
        },
    };
    assert!(request.validate().is_err());
    request.input.credential_file = Some(base.join("credential"));
    request.validate().unwrap();
    let mut value = serde_json::to_value(&request).unwrap();
    value["ambient_token"] = json!(true);
    assert!(serde_json::from_value::<Request>(value).is_err());
    request.input.parent_commit = "main".into();
    assert!(request.validate().is_err());
    root.close().unwrap();
}
#[cfg(unix)]
#[test]
fn publication_actual_inert_child_correlates_complete_receipt_and_refuses_failed_custody() {
    for mode in [
        "ok",
        "outer-error",
        "object-error",
        "partial-errors",
        "wrong-hash",
        "wrong-roster",
        "missing-custody",
        "nonzero",
        "source-drift",
    ] {
        let root = tempfile::tempdir().unwrap();
        let base = root.path().canonicalize().unwrap();
        let request = request(&base, mode);
        let phase = base.join("phase");
        std::fs::create_dir(&phase).unwrap();
        let mut evidence = json!({});
        let result = execute(
            &request,
            &phase,
            Instant::now() + Duration::from_secs(20),
            &Cancellation::default(),
            &mut evidence,
        );
        assert_eq!(result.is_ok(), mode == "ok");
        assert_eq!(
            evidence["status"],
            if mode == "ok" { "PUBLISHED" } else { "FAILED" }
        );
        assert_eq!(evidence["pre_identity"], true);
        assert!(!evidence["partial_progress"].is_null());
        assert_eq!(evidence["process"]["cleanup"]["complete"], true);
        assert_eq!(evidence["process"]["cleanup"]["forced"], false);
        assert_eq!(evidence["process"]["stdout"]["line_capture_complete"], true);
        if mode == "outer-error" || mode == "object-error" {
            assert_eq!(evidence["final_error"], true);
            assert!(evidence["final_receipt"].is_null());
        }
        if mode == "partial-errors" {
            assert_eq!(evidence["final_error"], false);
            assert_eq!(evidence["final_receipt"]["status"], "FAILED");
            assert_eq!(evidence["final_receipt"]["error_present"], true);
            assert_eq!(
                evidence["final_receipt"]["publication"]["error_present"],
                true
            );
            assert_eq!(
                evidence["final_receipt"]["publication"]["objects"][0]["error_present"],
                true
            );
        }
        if mode == "source-drift" {
            assert_eq!(evidence["post_identity"], false);
            assert!(!evidence["final_receipt"].is_null());
        }
        root.close().unwrap();
    }
}
#[cfg(unix)]
#[test]
fn publication_actual_owned_cancel_and_deadline_preserve_partial_commit_without_success() {
    for (cancelled, forced) in [(true, false), (false, false), (true, true)] {
        let root = tempfile::tempdir().unwrap();
        let base = root.path().canonicalize().unwrap();
        let request = request(&base, if forced { "forced" } else { "held" });
        let phase = base.join("phase");
        std::fs::create_dir(&phase).unwrap();
        let cancel = Cancellation::default();
        let owned_cancel = cancel.clone();
        let child_phase = phase.clone();
        let worker = std::thread::spawn(move || {
            let mut evidence = json!({});
            let deadline = Instant::now() + Duration::from_secs(if cancelled { 20 } else { 7 });
            let result = execute(
                &request,
                &child_phase,
                deadline,
                &owned_cancel,
                &mut evidence,
            );
            (result.is_err(), evidence)
        });
        let marker = phase.join("helper-output/started.json");
        let until = Instant::now() + Duration::from_secs(5);
        while !marker.exists() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        let observed = marker.exists();
        if cancelled || !observed {
            cancel.cancel();
        }
        let (failed, evidence) = worker.join().unwrap();
        assert!(observed && failed);
        assert_eq!(evidence["status"], "FAILED");
        assert_eq!(
            evidence["process"]["outcome"],
            if cancelled { "Cancelled" } else { "Deadline" }
        );
        assert_eq!(evidence["process"]["cleanup"]["complete"], true);
        assert_eq!(evidence["process"]["cleanup"]["forced"], forced);
        assert_eq!(
            evidence["partial_progress"]["publication"]["commit_oid"],
            "c".repeat(40)
        );
        if forced {
            assert!(evidence["final_receipt"].is_null());
            assert_eq!(evidence["final_error"], true);
        } else {
            assert_eq!(evidence["final_receipt"]["status"], "FAILED");
        }
        root.close().unwrap();
    }
}
