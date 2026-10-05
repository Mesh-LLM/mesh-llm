use super::*;
use serde_json::json;

fn input() -> Input {
    serde_json::from_value(json!({"schema_version":1,
        "binary":std::env::temp_dir().join("skippy-server"),"binary_sha256":"a".repeat(64),
        "native_runtime_root":std::env::temp_dir(),"admission_concurrency":0,"execution_timeout_secs":10,
        "stage":{"model_id":"fixture","model_path":std::env::temp_dir().join("fixture.gguf"),
            "source_model_sha256":"b".repeat(64),"layer_end":2,"ctx_size":512,"lane_count":2,
            "n_gpu_layers":0,"payload":"resident-kv","cache_entries":1},
        "worker":{"schema_version":1,"server_log":std::env::temp_dir().join("server.log"),
            "startup_timeout_secs":1,"telemetry_timeout_secs":1,"cache_seed":null,
            "phase":{"schema_version":1,"round":1,"version":"old","base_url":"http://127.0.0.1:12345/v1",
                "model":"fixture","output_tokens":2,"request_timeout_secs":1.0,"stagger_ms":0.0,
                "prompts":[{"family":"family-0","prompt":"task"},{"family":"family-1","prompt":"task"}]}}})).unwrap()
}

#[test]
fn admission_resolves_zero_to_the_full_measurement_and_never_underfills_or_exceeds_lanes() {
    let mut input = input();
    assert_eq!(input.admission().unwrap(), 2);
    input.validate().unwrap();
    input.admission_concurrency = 1;
    assert!(input.validate().is_err());
    input.admission_concurrency = 3;
    assert!(input.validate().is_err());
    input.admission_concurrency = 2;
    input.worker.phase.model = "other".into();
    assert!(input.validate().is_err());
    input.worker.phase.model = "fixture".into();
    input.worker.startup_timeout_secs = 10;
    assert!(input.validate().is_err());
}

fn launch(member: MemberId) -> Launch {
    Launch {
        member,
        spec: ProcessSpec {
            executable: "fixture".into(),
            arguments: vec![],
            cwd: std::env::temp_dir(),
            environment: Default::default(),
        },
        files: Default::default(),
        readiness_deadline: Duration::from_secs(10),
    }
}

#[test]
fn owner_starts_worker_only_after_server_admission_and_stops_server_after_failed_worker() {
    use crate::process::retained::Snapshot;
    let mut owner = Owner {
        server: Some(launch(MemberId::Seed)),
        worker: Some(launch(MemberId::WorkerOne)),
        stopping: false,
        policy: ExpectedExit::new(&[0, 1], Duration::from_secs(10)).unwrap(),
        telemetry: None,
    };
    fn context(members: &[crate::process::retained::Snapshot]) -> Context<'_> {
        Context {
            elapsed: Duration::ZERO,
            remaining: Duration::from_secs(10),
            members,
        }
    }
    assert!(matches!(owner.tick(context(&[])), Action::Start(_)));
    assert!(matches!(owner.tick(context(&[])), Action::Pending));
    let mut members = vec![Snapshot {
        member: MemberId::Seed,
        pid: 1,
        started: std::time::Instant::now(),
        state: MemberState::Starting,
    }];
    assert!(matches!(
        owner.tick(context(&members)),
        Action::Admit(MemberId::Seed)
    ));
    members[0].state = MemberState::Ready {
        elapsed: Duration::ZERO,
    };
    assert!(matches!(
        owner.tick(context(&members)),
        Action::StartExpected { .. }
    ));
    members.push(Snapshot {
        member: MemberId::WorkerOne,
        pid: 2,
        started: std::time::Instant::now(),
        state: MemberState::ExpectedExit {
            status: 1,
            elapsed: Duration::from_secs(1),
        },
    });
    assert!(matches!(
        owner.tick(context(&members)),
        Action::Stop(MemberId::Seed)
    ));
    assert!(matches!(owner.tick(context(&[])), Action::Complete));
}
