//! Portable request, ordered window-record and terminal refusal ownership.
use super::*;
fn input(count: u32) -> (tempfile::TempDir, Input) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let pin = |name: &str| json!({"path":root.join(name),"sha256":"a".repeat(64)});
    let request=serde_json::from_value(json!({"schema_version":1,"workflow":"quantization","timeout_seconds":259200,
        "window_template":{"schema_version":1,"tool_kind":"supplied-window-quantizer","profile_version":"window-v1",
        "tool":pin("tool"),"tool_source":pin("tool-source"),"runtime":pin("runtime"),"manifest":pin("manifest"),
        "recipe":pin("recipe"),"helper":pin("helper"),"helper_source":pin("helper-source"),
        "source_repo":"owner/source","source_revision":"a".repeat(40),"source_root":root.join("source"),"source_prefix":"BF16",
        "source_parts":(1..=count).map(|i|pin(&format!("source/BF16/model-{i:05}-of-{count:05}.gguf"))).collect::<Vec<_>>(),
        "target_repo":"owner/quant","target_root":root.join("target"),"target_prefix":"Q4_K_M","basename":"model","quant":"Q4_K_M",
        "expected_splits":count,"ordinal":1,"work_root":root.join("work"),"credential_file":root.join("credential"),
        "publication_confirmed":true,"timeout_seconds":86400,"resume":null},"resumes":[],"loader":pin("loader"),"package":null})).unwrap();
    (directory, request)
}
#[test]
fn quant_job_large_complete_request_and_workflow_allowances_are_typed() {
    let (_pins, mut i) = input(1024);
    i.validate().unwrap();
    assert_eq!(i.window(1024).unwrap().ordinal, 1024);
    i.timeout_seconds = 259201;
    assert!(i.validate().is_err());
    i.workflow = contract::Workflow::QuantizationAndPackage;
    i.timeout_seconds = 345600;
    assert!(i.validate().is_err());
    let pin = |name: &str| json!({"path":_pins.path().canonicalize().unwrap().join(name),"sha256":"b".repeat(64)});
    i.package = Some(
        serde_json::from_value(json!({"writer":pin("writer"),
      "writer_source":pin("writer-source"),"generation_defaults":pin("defaults"),
      "target_repo":"owner/package","max_artifact_bytes":8589934592_u64}))
        .unwrap(),
    );
    i.validate().unwrap();
    i.timeout_seconds = 345601;
    assert!(i.validate().is_err());
}
#[test]
fn quant_job_verified_window_records_preserve_full_roster_and_reject_wrong_context() {
    let (_pins, i) = input(2);
    let root = tempfile::tempdir().unwrap();
    let request = i.window(1).unwrap();
    let record = json!({"context_sha256":request.context_sha256().unwrap(),"ordinal":1,"relative_path":request.remote_path(),
      "window_uploaded":true,"artifact":{"sha256":"b".repeat(64),"byte_size":24}});
    admission::publish(&root.path().join("window-record.json"), &record).unwrap();
    let row = json!({"window_uploaded":true,"record_commit":"c".repeat(40)});
    let mut roster = vec![];
    final_commit::window_artifacts(&request, root.path(), &row, &mut roster).unwrap();
    assert_eq!(roster.len(), 2);
    assert_eq!(roster[0]["path"], request.remote_path());
    assert!(final_commit::window_artifacts(&request, root.path(), &row, &mut roster).is_err());
    assert!(
        final_commit::window_artifacts(&i.window(2).unwrap(), root.path(), &row, &mut vec![])
            .is_err()
    );
}
#[test]
fn quant_job_precancel_and_expired_deadline_retain_incomplete_request_without_launch() {
    let (_pins, i) = input(1);
    let root = tempfile::tempdir().unwrap();
    let cancel = Cancellation::default();
    cancel.cancel();
    let mut receipt = Value::Null;
    assert!(
        execute(
            &i,
            root.path(),
            Instant::now() + Duration::from_secs(60),
            &cancel,
            &mut receipt
        )
        .is_err()
    );
    assert_eq!(receipt["completed_job"], false);
    assert_eq!(receipt["status"], "FAILED");
    assert_eq!(
        receipt["request_sha256"],
        admission::digest(&serde_json::to_vec(&i).unwrap())
    );
    assert!(receipt["windows"].as_array().unwrap().is_empty());
    assert!(
        execute(
            &i,
            root.path(),
            Instant::now(),
            &Cancellation::default(),
            &mut receipt
        )
        .is_err()
    );
    assert_eq!(receipt["full_roster_verified"], false);
    assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
}
#[test]
fn quant_job_sequential_owner_covers_every_ordinal_and_retains_failed_window() {
    let (_pins, i) = input(4);
    let root = tempfile::tempdir().unwrap();
    let until = Instant::now() + Duration::from_secs(60);
    let mut evidence = initial(&i).unwrap();
    let mut seen = vec![];
    let roster = windows(
        &i,
        root.path(),
        until,
        &Cancellation::default(),
        &mut evidence,
        |w, at, deadline, _, row| {
            assert!(deadline <= until);
            seen.push(w.ordinal);
            admission::publish(
                &at.join("window-record.json"),
                &json!({"context_sha256":w.context_sha256()?,
            "ordinal":w.ordinal,"relative_path":w.remote_path(),"window_uploaded":true,
            "artifact":{"sha256":"b".repeat(64),"byte_size":24}}),
            )?;
            row["window_uploaded"] = json!(true);
            row["record_commit"] = json!("c".repeat(40));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, vec![1, 2, 3, 4]);
    assert_eq!(roster.len(), 8);
    assert_eq!(evidence["completed_job"], false);
    assert_eq!(evidence["tool_profile_qualified"], false);
    let failed = tempfile::tempdir().unwrap();
    let mut evidence = initial(&i).unwrap();
    assert!(windows(&i,failed.path(),until,&Cancellation::default(),&mut evidence,|w,at,_,_,row| {
        if w.ordinal==3 {row["error"]=json!("causal window refusal");return Err("causal refusal".into());}
        admission::publish(&at.join("window-record.json"),&json!({"context_sha256":w.context_sha256()?,"ordinal":w.ordinal,
          "relative_path":w.remote_path(),"window_uploaded":true,"artifact":{"sha256":"b".repeat(64),"byte_size":24}}))?;
        row["window_uploaded"]=json!(true);row["record_commit"]=json!("c".repeat(40));Ok(())
    }).is_err());
    assert_eq!(evidence["windows"].as_array().unwrap().len(), 3);
    assert_eq!(evidence["windows"][2]["error"], "causal window refusal");
    assert!(!failed.path().join("window-00004").exists());
}
#[test]
fn quant_job_missing_or_corrupt_loader_refuses_before_repository_or_window_mutation() {
    let (_pins, i) = input(1);
    let root = tempfile::tempdir().unwrap();
    let root = root.path().canonicalize().unwrap();
    for bytes in [None, Some(b"corrupt loader".as_slice())] {
        if let Some(bytes) = bytes {
            std::fs::write(&i.loader.path, bytes).unwrap();
        }
        let mut receipt = Value::Null;
        assert!(
            execute(
                &i,
                &root,
                Instant::now() + Duration::from_secs(60),
                &Cancellation::default(),
                &mut receipt
            )
            .is_err()
        );
        assert_eq!(receipt["completed_job"], false);
        assert_eq!(receipt["status"], "FAILED");
        assert!(receipt["quant-repository"].is_null());
        assert!(receipt["windows"].as_array().unwrap().is_empty());
        assert!(std::fs::read_dir(&root).unwrap().next().is_none());
    }
}
