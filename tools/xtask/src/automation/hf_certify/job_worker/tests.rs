use super::*;
use contract::{Certification, Input, Workflow};
fn input() -> Input {
    let root = std::env::current_dir().unwrap().canonicalize().unwrap();
    Input {
        receipt_export: None,
        schema_version: 1,
        workflow: Workflow::Certification,
        timeout_secs: 30,
        runner: admission::Artifact {
            path: root.join("runner"),
            sha256: "a".repeat(64),
        },
        bootstrap: bootstrap::contract::Input {
            schema_version: 1,
            mesh_commit: "b".repeat(40),
            git_tree: "c".repeat(40),
            llama_commit: "d".repeat(40),
            upstream_file_sha256: "e".repeat(64),
            image: format!("fixture/image@sha256:{}", "f".repeat(64)),
            native_profile: "standalone-static-skippy-quantize-cpu".into(),
            tools: [
                "git", "just", "cargo", "rustc", "cmake", "c++", "ld.lld", "curl",
            ]
            .into_iter()
            .map(|name| bootstrap::contract::Tool {
                name: name.into(),
                path: root.join("tools").join(name),
                sha256: "1".repeat(64),
            })
            .collect(),
            path_directories: vec![root.join("tools")],
            timeout_seconds: 30,
            cpu_plan_receipt_sha256: "2".repeat(64),
            declared_estimate_usd: 1.0,
            max_cost_usd: 2.0,
        },
        certification: Certification {
            mode: admission::Mode::ProjectorOnly,
            projector: admission::Artifact {
                path: root.join("projector.gguf"),
                sha256: "3".repeat(64),
            },
            target_parts: Vec::new(),
            expected_parts: 0,
            mtp_draft: None,
            layer_count: 1,
            mtp_layer_count: None,
            ctx_size: 64,
        },
        projector: acquisition::Projector::Supplied {
            artifact: admission::Artifact {
                path: root.join("projector.gguf"),
                sha256: "3".repeat(64),
            },
        },
    }
}
#[test]
fn native_job_closed_schema_and_shared_budget_preserve_original_profile() {
    let base = input();
    base.validate().unwrap();
    let mut value = serde_json::to_value(&base).unwrap();
    value["done"] = json!(true);
    assert!(serde_json::from_value::<Input>(value).is_err());
    let mut mismatch = input();
    mismatch.bootstrap.timeout_seconds = 31;
    assert!(mismatch.validate().is_err());
    let mut mismatch = input();
    mismatch.projector = acquisition::Projector::Supplied {
        artifact: admission::Artifact {
            path: mismatch.certification.projector.path.clone(),
            sha256: "4".repeat(64),
        },
    };
    assert!(mismatch.validate().is_err());
}
#[test]
fn native_job_template_binds_only_actual_bootstrap_identity() {
    let base = input();
    let observed = bootstrap::execution::ObservedBootstrap {
        binary: admission::Artifact {
            path: base.runner.path.with_file_name("actual-output"),
            sha256: "9".repeat(64),
        },
        mesh_commit: "8".repeat(40),
        prepared_llama_commit: "7".repeat(40),
    };
    let bound = base.certification.bind(&observed, 30).unwrap();
    assert!(bound.binary == observed.binary);
    assert_eq!(bound.supplied_mesh_revision, observed.mesh_commit);
    assert!(bound.projector == base.certification.projector);
    assert_eq!(bound.ctx_size, 64);
    assert_eq!(
        bound.native_profile,
        "standalone-static-skippy-quantize-cpu"
    );
}
#[test]
fn native_job_unsupported_compose_refuses_before_output_or_bootstrap() {
    let root = tempfile::tempdir().unwrap();
    let mut value = input();
    value.workflow = Workflow::NemotronCompose;
    let source = root.path().join("input.json");
    std::fs::write(&source, serde_json::to_vec(&value).unwrap()).unwrap();
    let output = root.path().join("never-created");
    let result = run(&[
        "--input".into(),
        source.to_string_lossy().into_owned(),
        "--output-directory".into(),
        output.to_string_lossy().into_owned(),
    ]);
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unsupported before launch")
    );
    assert!(!output.exists());
    root.close().unwrap();
}
#[test]
fn native_job_cancel_deadline_and_runner_mismatch_prevent_phase_launch() {
    let root = tempfile::tempdir().unwrap();
    let mut value = input();
    value.runner.path = std::env::current_exe().unwrap().canonicalize().unwrap();
    for mode in 0..3 {
        let cancel = Cancellation::default();
        if mode == 0 {
            cancel.cancel();
        }
        let deadline = if mode == 1 {
            Instant::now()
        } else {
            Instant::now() + Duration::from_secs(10)
        };
        let mut evidence = json!({});
        assert!(execute(&value, root.path(), deadline, &cancel, &mut evidence).is_err());
        assert!(!root.path().join("bootstrap").exists());
    }
    root.close().unwrap();
}
#[cfg(unix)]
#[test]
fn native_job_writerless_fifo_refuses_before_output_or_interrupt() {
    use std::os::unix::ffi::OsStrExt;
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("fifo");
    let name = std::ffi::CString::new(source.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let output = root.path().join("never-created");
    assert!(
        run(&[
            "--input".into(),
            source.to_string_lossy().into_owned(),
            "--output-directory".into(),
            output.to_string_lossy().into_owned()
        ])
        .is_err()
    );
    assert!(!output.exists());
    root.close().unwrap();
}

#[test]
fn native_job_environment_input_is_bounded_and_jobs_budget_keeps_native_phase_cap() {
    assert!(environment_input(None).is_err());
    assert!(environment_input(Some("".into())).is_err());
    assert!(environment_input(Some("x".repeat(65537).into())).is_err());
    let base = input();
    let bytes = serde_json::to_vec(&base).unwrap();
    assert_eq!(
        environment_input(Some(String::from_utf8(bytes.clone()).unwrap().into())).unwrap(),
        bytes
    );
    let overall = Instant::now() + Duration::from_secs(7200);
    let (phase, secs) = certification_window(overall, &Cancellation::default()).unwrap();
    assert!(phase <= overall && (3590..=3600).contains(&secs));
    assert!(
        certification_window(
            Instant::now() + Duration::from_secs(4),
            &Cancellation::default()
        )
        .is_err()
    );
    let cancelled = Cancellation::default();
    cancelled.cancel();
    assert!(certification_window(overall, &cancelled).is_err());
    let mut long = input();
    long.timeout_secs = 7200;
    long.bootstrap.timeout_seconds = 7200;
    long.validate().unwrap();
    let observed = bootstrap::execution::ObservedBootstrap {
        binary: long.runner.clone(),
        mesh_commit: long.bootstrap.mesh_commit.clone(),
        prepared_llama_commit: long.bootstrap.llama_commit.clone(),
    };
    assert_eq!(
        long.certification
            .bind(&observed, long.timeout_secs.min(3600))
            .unwrap()
            .timeout_secs,
        3600
    );
    long.timeout_secs = 86401;
    long.bootstrap.timeout_seconds = 86401;
    assert!(long.validate().is_err());
}

#[test]
fn native_job_private_export_credential_and_shared_reserve_are_bounded() {
    let root = tempfile::tempdir().unwrap();
    for value in [
        None,
        Some("".into()),
        Some("bad\nvalue".into()),
        Some("x".repeat(8193).into()),
    ] {
        assert!(publication_credential(root.path(), value).is_err());
    }
    let file = publication_credential(root.path(), Some("inert-private-fixture".into())).unwrap();
    let path = file.path().to_owned();
    assert_eq!(std::fs::read(&path).unwrap(), b"inert-private-fixture");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    file.close().unwrap();
    assert!(!path.exists());
    let mut request = input();
    request.receipt_export = Some(super::receipt_export::Config {
        helper: request.runner.clone(),
        helper_source: request.runner.clone(),
        repo: "fixture/evidence".into(),
        parent_commit: "a".repeat(40),
        credential_file: None,
        credential_environment: true,
        path_in_repo: "run/native-job.json".into(),
        export_budget_secs: 10,
    });
    request.validate().unwrap();
    request.receipt_export.as_mut().unwrap().export_budget_secs = request.timeout_secs;
    assert!(request.validate().is_err());
    root.close().unwrap();
}

#[test]
fn native_job_supplied_compose_tag_has_closed_distinct_input_and_refuses_checkpoint_claim() {
    let base = input();
    let mut value = serde_json::to_value(&base).unwrap();
    value["workflow"] = json!("supplied-converted-compose");
    value.as_object_mut().unwrap().remove("certification");
    value.as_object_mut().unwrap().remove("projector");
    let root = std::env::current_dir().unwrap().canonicalize().unwrap();
    let pin = |p| json!({"path":p,"sha256":"b".repeat(64)});
    value["composition"] = json!({"target_parts":(1..=3).map(|i|pin(root.join(format!("Target-{i:05}-of-00003.gguf")))).collect::<Vec<_>>(),"mtp":{"kind":"supplied-converted","artifact":pin(root.join("mtp.gguf"))},"target_basename":"Target","composite_basename":"Composite","expected_parts":3,"mtp_block":88,"composite_repo":"fixture/composite"});
    let input: contract::JobInput = serde_json::from_value(value.clone()).unwrap();
    input.validate().unwrap();
    assert_eq!(input.native_status(), "COMPOSED");
    let mut mixed = value.clone();
    mixed["certification"] = json!({});
    assert!(serde_json::from_value::<contract::JobInput>(mixed).is_err());
    value["composition"]["mtp"] =
        json!({"kind":"nemotron-checkpoint","directory":root.join("checkpoint")});
    let input: contract::JobInput = serde_json::from_value(value).unwrap();
    assert!(input.validate().is_err());
}

#[test]
fn native_job_raw_tag_requires_complete_pins_and_remains_distinct_from_supplied_input() {
    let base = input();
    let mut value = serde_json::to_value(&base).unwrap();
    let root = std::env::current_dir().unwrap().canonicalize().unwrap();
    let source = root.join("checkpoint");
    let pin = |p| json!({"path":p,"sha256":"b".repeat(64)});
    value["workflow"] = json!("native-nemotron-compose");
    value.as_object_mut().unwrap().remove("certification");
    value.as_object_mut().unwrap().remove("projector");
    let checkpoint_files = ["config.json", "tokenizer.json", "weights.safetensors"]
        .iter()
        .map(|n| pin(source.join(n)))
        .collect::<Vec<_>>();
    value["conversion"] = json!({"checkpoint_directory":source,"checkpoint_files":checkpoint_files,"tokenizer_profile":pin(root.join("profile.json")),"target_parts":(1..=3).map(|i|pin(root.join(format!("Target-{i:05}-of-00003.gguf")))).collect::<Vec<_>>(),"target_basename":"Target","composite_basename":"Composite","expected_parts":3,"mtp_block":88,"composite_repo":"fixture/composite"});
    let admitted: contract::JobInput = serde_json::from_value(value.clone()).unwrap();
    admitted.validate().unwrap();
    assert_eq!(admitted.native_status(), "COMPOSED");
    let mut mixed = value.clone();
    mixed["composition"] = json!({});
    assert!(serde_json::from_value::<contract::JobInput>(mixed).is_err());
    value["conversion"]["checkpoint_files"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    let refused: contract::JobInput = serde_json::from_value(value).unwrap();
    assert!(refused.validate().is_err());
}
