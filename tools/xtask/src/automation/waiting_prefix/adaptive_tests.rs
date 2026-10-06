use super::{adaptive_identity as identity, adaptive_telemetry::Observation};
use serde_json::json;
use std::time::Duration;
fn arm(root: &std::path::Path) -> identity::Input {
    serde_json::from_value(json!({"schema_version":1,"round":1,"version":"old","binary":root.join("binary"),"binary_sha256":"a".repeat(64),"commit":"b".repeat(40),
        "native_build":root.join("runtime"),"native_build_sha256":"c".repeat(64),"native_profile":"standalone-static-skippy-server","model":root.join("model.gguf"),"model_sha256":"d".repeat(64),
        "model_id":"fixture","ctx_size":512,"split_layer":2,"layer_end":4,"n_gpu_layers":999,"adaptive_target_ms":10.0,"stage_ports":[12001,12002],"openai_port":12003})).unwrap()
}
#[test]
fn adaptive_arm_configs_preserve_split_context_offload_and_new_only_target_policy() {
    let root = tempfile::tempdir().unwrap();
    let mut input = arm(root.path());
    input.validate().unwrap();
    let first = input.config(0);
    let last = input.config(1);
    assert_eq!(first["layer_start"], 0);
    assert_eq!(first["layer_end"], last["layer_start"]);
    assert_eq!(last["layer_end"], 4);
    assert_eq!(first["downstream"]["endpoint"], "tcp://127.0.0.1:12002");
    assert!(first["upstream"].is_null());
    assert_eq!(last["upstream"]["endpoint"], "tcp://127.0.0.1:12001");
    assert!(last["downstream"].is_null());
    assert_eq!(first["ctx_size"], 512);
    assert_eq!(first["n_gpu_layers"], 999);
    assert_eq!(first["kv_cache"]["mode"], "disabled");
    let old = super::adaptive_cell::arguments(&input, root.path(), 0);
    assert!(!old.iter().any(
        |a| matches!(a,crate::process::Value::Public(v) if v=="--openai-prefill-adaptive-target-ms")
    ));
    input.version = super::acceptance::Version::New;
    let new = super::adaptive_cell::arguments(&input, root.path(), 0);
    assert!(new.windows(2).any(|a|matches!((&a[0],&a[1]),(crate::process::Value::Public(k),crate::process::Value::Public(v))if k=="--openai-prefill-adaptive-target-ms"&&v=="10")));
    input.split_layer = 4;
    assert!(input.validate().is_err());
    input.split_layer = 2;
    input.ctx_size = 0;
    assert!(input.validate().is_err());
    input.ctx_size = 512;
    input.native_profile = "dynamic-host".into();
    assert!(input.validate().is_err());
    root.close().unwrap();
}
fn event() -> serde_json::Value {
    json!({"event":"stage.openai_prefill","attributes":{"llama_stage.prefill_chunk_count":3,"llama_stage.prefill_min_chunk_size":128,"llama_stage.prefill_max_chunk_size":384,"llama_stage.elapsed_ms":5.0,"skippy.kv.chain_cache_errors":0,"skippy.kv.stage0_cache_errors":0}})
}

#[test]
fn adaptive_identity_worker_binds_actual_local_bytes_context_configs_and_refuses_mutation() {
    let root = tempfile::tempdir().unwrap();
    let mut input = arm(root.path());
    std::fs::create_dir_all(&input.native_build).unwrap();
    std::fs::write(
        input.native_build.join("inert-library"),
        b"fixture runtime bytes",
    )
    .unwrap();
    std::fs::write(&input.binary, b"inert binary bytes").unwrap();
    let string = |bytes: &mut Vec<u8>, text: &str| {
        bytes.extend((text.len() as u64).to_le_bytes());
        bytes.extend(text.as_bytes());
    };
    let mut model = b"GGUF".to_vec();
    model.extend(3_u32.to_le_bytes());
    model.extend(0_u64.to_le_bytes());
    model.extend(4_u64.to_le_bytes());
    string(&mut model, "general.architecture");
    model.extend(8_u32.to_le_bytes());
    string(&mut model, "llama");
    for (key, value) in [
        ("llama.context_length", 512_u32),
        ("llama.block_count", 4),
        ("llama.embedding_length", 16),
    ] {
        string(&mut model, key);
        model.extend(4_u32.to_le_bytes());
        model.extend(value.to_le_bytes());
    }
    std::fs::write(&input.model, &model).unwrap();
    input.model_sha256 = identity::digest(&model);
    input.binary_sha256 = identity::digest(b"inert binary bytes");
    input.native_build_sha256 = crate::product::digest::tree_sha256(&input.native_build)
        .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error));
    let request = root.path().join("input.json");
    let output = root.path().join("output.json");
    let bytes = serde_json::to_vec(&input).unwrap();
    std::fs::write(&request, &bytes).unwrap();
    let args = vec![
        "--input".into(),
        request.display().to_string(),
        "--output".into(),
        output.display().to_string(),
    ];
    identity::run(&args).unwrap();
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(receipt["request_sha256"], identity::digest(&bytes));
    assert_eq!(receipt["model_identity"]["sha256"], input.model_sha256);
    assert_eq!(receipt["model_identity"]["native_context_tokens"], 512);
    assert_eq!(
        receipt["configs"][0]["layer_end"],
        receipt["configs"][1]["layer_start"]
    );
    for path in [
        &input.binary,
        &input.model,
        &input.native_build.join("inert-library"),
    ] {
        let original = std::fs::read(path).unwrap();
        std::fs::write(path, b"changed").unwrap();
        std::fs::remove_file(&output).unwrap();
        assert!(identity::run(&args).is_err());
        assert!(!output.exists());
        std::fs::write(path, original).unwrap();
        identity::run(&args).unwrap();
    }
    std::fs::remove_file(&output).unwrap();
    input.ctx_size = 513;
    std::fs::write(&request, serde_json::to_vec(&input).unwrap()).unwrap();
    assert!(identity::run(&args).is_err());
    assert!(!output.exists());
    root.close().unwrap();
}
#[test]
fn adaptive_prefill_requires_exact_complete_calibration_and_measured_typed_roster() {
    let mut observed = Observation::default();
    observed.observe(&serde_json::to_vec(&event()).unwrap());
    assert!(observed.measured(1, true).is_err());
    observed.observe(&serde_json::to_vec(&event()).unwrap());
    assert_eq!(observed.measured(1, true).unwrap().len(), 1);
    assert!(observed.measured(1, false).is_err());
    observed.observe(&serde_json::to_vec(&event()).unwrap());
    assert!(observed.measured(1, true).is_err());
    for (key, value) in [
        ("llama_stage.prefill_chunk_count", json!(0)),
        ("llama_stage.prefill_min_chunk_size", json!(999)),
        ("llama_stage.elapsed_ms", json!(-1)),
        ("skippy.kv.chain_cache_errors", json!(1)),
    ] {
        let mut row = event();
        row["attributes"][key] = value;
        let mut observation = Observation::default();
        observation.observe(&serde_json::to_vec(&row).unwrap());
        assert!(observation.measured(0, true).is_err(), "{key}");
    }
    let mut row = event();
    row["attributes"]
        .as_object_mut()
        .unwrap()
        .remove("llama_stage.elapsed_ms");
    let mut observation = Observation::default();
    observation.observe(&serde_json::to_vec(&row).unwrap());
    assert!(observation.error.is_some());
}
#[cfg(unix)]
#[test]
fn adaptive_retained_pair_owns_worker_and_typed_prefill_through_clean_shutdown() {
    use crate::process::{
        Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
        retained::{ExpectedExit, Launch, MemberId},
    };
    let root = tempfile::tempdir().unwrap();
    let mut row = event();
    row["attributes"]["token"] = json!("fixture-secret");
    let line = serde_json::to_string(&row).unwrap();
    let stage0 = format!(
        "trap 'exit 0' TERM; printf '%s\\n' '{line}' '{line}' >&2; : > upstream-ready; while :; do :; done"
    );
    let stage1 = "trap 'exit 0' TERM; : > downstream-ready; while :; do :; done";
    let worker = "while [ ! -f upstream-ready ] || [ ! -f downstream-ready ]; do :; done; exit 0";
    let execution = Duration::from_secs(5);
    let launch = |member, name: &str, script: String| Launch {
        member,
        spec: ProcessSpec {
            executable: "/bin/bash".into(),
            arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
            cwd: root.path().into(),
            environment: Default::default(),
        },
        files: OutputFiles {
            stdout: Some(root.path().join(format!("{name}.out"))),
            stderr: Some(root.path().join(format!("{name}.err"))),
        },
        readiness_deadline: execution,
    };
    let mut owner = super::adaptive_owner::Owner {
        downstream: Some(launch(MemberId::Seed, "down", stage1.into())),
        upstream: Some(launch(MemberId::WorkerOne, "up", stage0)),
        worker: Some(launch(MemberId::WorkerTwo, "worker", worker.into())),
        policy: ExpectedExit::new(&[0, 1], execution).unwrap(),
        telemetry: Observation::default(),
        stopping: 0,
    };
    let limits = Limits {
        execution,
        graceful_shutdown: Duration::from_millis(250),
        forced_shutdown: Duration::from_millis(250),
        retained_bytes_per_stream: 4096,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = crate::process::retained::run(
        &mut owner,
        &limits,
        &crate::process::Cancellation::default(),
    )
    .unwrap();
    assert!(report.recovery_success(), "{:?}", report.outcome);
    assert_eq!(report.members.len(), 3);
    for member in &report.members {
        assert!(member.process.failure.is_none());
        assert!(member.process.cleanup.complete);
        assert!(!member.process.cleanup.forced);
        assert!(!member.process.cleanup.graceful_signal_failed);
        assert!(member.process.cleanup.failure.is_none());
        assert_eq!(
            member
                .process
                .status
                .as_ref()
                .and_then(std::process::ExitStatus::code),
            Some(0)
        );
        assert!(member.process.stdout.line_capture_complete);
        assert!(member.process.stderr.line_capture_complete);
    }
    assert_eq!(owner.telemetry.measured(1, true).unwrap().len(), 1);
    assert!(
        !std::fs::read_to_string(root.path().join("up.err"))
            .unwrap()
            .contains("fixture-secret")
    );
    root.close().unwrap();
}
