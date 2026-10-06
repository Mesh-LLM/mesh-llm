use super::{
    contract::{Host, Input, profile},
    owner::Owner,
};
use crate::process::{
    ObservedLine, Stream,
    retained::{Action, Context, Coordinator, MemberId, MemberState, Snapshot},
};
use serde_json::json;
use std::time::{Duration, Instant};
fn input() -> Input {
    let root = std::env::current_dir().unwrap().canonicalize().unwrap();
    let mut input: Input = serde_json::from_value(json!({"schema_version":1,"host":"skippy-new","binary":"/tmp/server","binary_sha256":"a".repeat(64),"source_commit":"a".repeat(40),
    "native_build":"/tmp/build","native_build_sha256":"b".repeat(64),"model":"/tmp/model","model_sha256":"c".repeat(64),"model_id":"fixture","layer_end":6,"ctx_size":128,"lane_count":1,"n_gpu_layers":-1,"port":12345,
    "environment":{"OMP_NUM_THREADS":"2"},"worker":{"schema_version":1,"cohort":"openai-concurrent","base_url":"http://127.0.0.1:12345/v1","prompt":"fixed","model_id":"fixture","requests":2,"concurrency":2,"output_tokens":32,"request_timeout_ms":1000,"execution_timeout_ms":10000},"startup_timeout_secs":5,"execution_timeout_secs":30})).unwrap();
    input.binary = root.join("inert-server-not-launched");
    input.native_build = root.join("inert-build-not-opened");
    input.model = root.join("inert-model-not-opened");
    input
}
fn owner() -> Owner {
    Owner {
        server: None,
        readiness: None,
        measurement: None,
        input: input(),
        marker: false,
        stopping: false,
    }
}
#[test]
fn cache_cell_keeps_queued_concurrency_distinct_from_host_lane_capacity() {
    let value = input();
    value.validate().unwrap();
    assert!(value.worker.concurrency > value.lane_count as usize);
    let mut value = value;
    value.worker.model_id = Some("other-model".into());
    assert!(value.validate().is_err());
}
#[test]
fn cache_cell_profile_refuses_credential_and_discovery_aliases() {
    for (name, value) in [
        ("OMP_NUM_THREADS", "2"),
        ("CUDA_VISIBLE_DEVICES", "0,1"),
        ("GGML_CUDA_FORCE_CUBLAS", "1"),
    ] {
        assert!(profile(name, value));
    }
    for name in [
        "GGML_API_TOKEN",
        "GGML_BACKEND_DL_PATH",
        "HOME",
        "LLAMA_STAGE_BUILD_DIR",
        "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
    ] {
        assert!(!profile(name, "secret"));
    }
    assert!(!profile("OMP_NUM_THREADS", "2\nsecret"));
}
#[test]
fn cache_cell_owned_marker_requires_exact_address_model_capacity_and_source_prefix() {
    let mut owner = owner();
    for line in [
        "unrelated skippy-server listening: openai=127.0.0.1:12345 model_id=fixture backend=f generation_concurrency=1",
        "skippy-server listening: openai=127.0.0.1:12346 model_id=fixture backend=f generation_concurrency=1",
        "skippy-server listening: openai=127.0.0.1:12345 model_id=other backend=f generation_concurrency=1",
        "skippy-server listening: openai=127.0.0.1:12345 model_id=fixture backend=f generation_concurrency=2",
    ] {
        owner.captured_line(
            MemberId::Seed,
            ObservedLine {
                ending: crate::process::LineEnding::Lf,
                stream: Stream::Stdout,
                bytes: line.as_bytes(),
            },
        );
        assert!(!owner.marker);
    }
    owner.captured_line(MemberId::WorkerTwo,ObservedLine{ending:crate::process::LineEnding::Lf,stream:Stream::Stdout,bytes:b"skippy-server listening: openai=127.0.0.1:12345 model_id=fixture backend=f generation_concurrency=1"});
    assert!(!owner.marker);
    owner.captured_line(MemberId::Seed,ObservedLine{ending:crate::process::LineEnding::Lf,stream:Stream::Stdout,bytes:b"skippy-server listening: openai=127.0.0.1:12345 model_id=fixture backend=f generation_concurrency=1"});
    assert!(owner.marker);
    owner.input.host = Host::NativeBaseline;
    owner.marker = false;
    for line in [
        "unrelated listening on http://127.0.0.1:12345",
        "srv  llama_server: listening on http://127.0.0.1:12346",
    ] {
        owner.captured_line(
            MemberId::Seed,
            ObservedLine {
                ending: crate::process::LineEnding::Lf,
                stream: Stream::Stdout,
                bytes: line.as_bytes(),
            },
        );
        assert!(!owner.marker);
    }
    owner.captured_line(
        MemberId::Seed,
        ObservedLine {
            ending: crate::process::LineEnding::Lf,
            stream: Stream::Stdout,
            bytes: b"srv  llama_server: listening on http://127.0.0.1:12345",
        },
    );
    assert!(owner.marker);
}
#[test]
fn cache_cell_failed_http_readiness_cannot_progress_to_measurement() {
    let mut owner = owner();
    let member = Snapshot {
        member: MemberId::WorkerOne,
        pid: 1,
        started: Instant::now(),
        state: MemberState::ExpectedExit {
            status: 1,
            elapsed: Duration::from_secs(1),
        },
    };
    assert!(matches!(
        owner.tick(Context {
            elapsed: Duration::from_secs(1),
            remaining: Duration::from_secs(5),
            members: &[member]
        }),
        Action::Reject(_)
    ));
}

#[test]
fn cache_cell_http_readiness_uses_remaining_host_startup_budget() {
    let mut owner = owner();
    let launch = crate::process::retained::Launch {
        member: MemberId::WorkerOne,
        spec: crate::process::ProcessSpec {
            executable: "/inert-not-launched".into(),
            arguments: vec![],
            cwd: "/tmp".into(),
            environment: std::collections::BTreeMap::new(),
        },
        files: crate::process::OutputFiles::default(),
        readiness_deadline: Duration::from_secs(5),
    };
    owner.readiness = Some(launch);
    owner.marker = true;
    let member = Snapshot {
        member: MemberId::Seed,
        pid: 1,
        started: Instant::now(),
        state: MemberState::Ready {
            elapsed: Duration::from_secs(4),
        },
    };
    let action = owner.tick(Context {
        elapsed: Duration::from_secs(4),
        remaining: Duration::from_secs(20),
        members: &[member],
    });
    let Action::StartExpected { policy, .. } = action else {
        panic!("readiness was not admitted");
    };
    assert_eq!(policy.deadline(), Duration::from_secs(1));
}

#[test]
fn cache_cell_complete_shards_require_all_pins_and_refuse_package_serving() {
    use crate::automation::cache_family_correctness::artifact::{Artifact, Kind};
    let mut value = input();
    value.layer_end = 62;
    value.model = value
        .model
        .parent()
        .unwrap()
        .join("MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf");
    value.artifact = Some(Artifact {
        kind: Kind::CompleteShards,
        tool: value.binary.clone(),
        tool_sha256: "a".repeat(64),
        shard_pins: (1..=3)
            .map(|i| {
                (
                    format!("MiniMax-M2.7-UD-Q2_K_XL-{i:05}-of-00003.gguf"),
                    "c".repeat(64),
                )
            })
            .collect(),
    });
    value.validate().unwrap();
    value
        .artifact
        .as_mut()
        .unwrap()
        .shard_pins
        .remove("MiniMax-M2.7-UD-Q2_K_XL-00003-of-00003.gguf");
    assert!(value.validate().is_err());
    value.artifact.as_mut().unwrap().kind = Kind::LayerPackage;
    assert!(value.validate().is_err());
}

#[test]
fn cache_family_cell_terminal_admission_retains_observed_rows_process_and_prior_failure() {
    for (cancelled, expired, finish_ok) in [
        (false, false, true),
        (true, false, true),
        (false, true, true),
        (false, false, false),
    ] {
        let mut receipt = json!({"status":"completed","rows":[{"status":"pass"}],"measurement":{"status":"completed"},"process":{"cleanup_complete":true},"error":"prior classified failure"});
        super::finalize(&mut receipt, cancelled, expired, finish_ok);
        assert_eq!(
            receipt["status"],
            if cancelled || expired || !finish_ok {
                "incomplete"
            } else {
                "completed"
            }
        );
        assert_eq!(receipt["rows"][0]["status"], "pass");
        assert_eq!(receipt["measurement"]["status"], "completed");
        assert_eq!(receipt["process"]["cleanup_complete"], true);
        assert_eq!(receipt["error"], "prior classified failure");
        if cancelled || expired || !finish_ok {
            assert_eq!(receipt["terminal_refusal"]["cancelled"], cancelled);
            assert_eq!(receipt["terminal_refusal"]["deadline_expired"], expired);
            assert_eq!(
                receipt["terminal_refusal"]["interrupt_finish_failed"],
                !finish_ok
            );
        }
    }
}

#[test]
fn cache_cell_remaining_budget_preserves_sweep_primary_and_bounds_every_stage() {
    let mut original = input();
    let first = original.worker.clone();
    let mut second = first.clone();
    second.concurrency = 4;
    second.requests = 4;
    second.execution_timeout_ms = 5000;
    original.worker_sweep = vec![first, second];
    original.validate().unwrap();
    let admitted =
        super::execution::bound_measurement(&original, Duration::from_millis(3000)).unwrap();
    admitted.validate().unwrap();
    assert_eq!(admitted.worker.execution_timeout_ms, 3000);
    assert!(
        admitted
            .worker_sweep
            .iter()
            .all(|stage| stage.execution_timeout_ms == 3000)
    );
    assert_eq!(
        serde_json::to_value(&admitted.worker).unwrap(),
        serde_json::to_value(&admitted.worker_sweep[0]).unwrap()
    );
    assert_eq!(original.worker.execution_timeout_ms, 10000);
    assert_eq!(original.worker_sweep[1].execution_timeout_ms, 5000);
    let generous = super::execution::bound_measurement(&original, Duration::from_secs(20)).unwrap();
    assert_eq!(generous.worker_sweep[1].execution_timeout_ms, 5000);
    assert!(super::execution::bound_measurement(&original, Duration::ZERO).is_err());
}
