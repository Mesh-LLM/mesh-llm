use super::{l3_contract::*, l3_execution::DiskRoots, l3_gates, l3_management::Client};
use serde_json::json;
use std::{collections::BTreeMap, time::Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
fn config() -> Config {
    serde_json::from_value(json!({
    "model":"hf:org/model@immutable/model.gguf","backend":"metal","lifecycle_cohort":"l3","required_sources":["buzz","opencode","goose"],"concurrency":[1],"prompt_min":18000,"prompt_max":24000,"cold_samples":1,"restart_samples":1,"identical_repeats":2,"max_output_tokens":2048,"disk_budget":"auto","minimum_free":"1GiB","low_space_disk_budget":"32GiB","low_space_minimum_free":"1TiB","max_l3_ttft_ratio":0.5,"max_payload_write_amplification":1.2,"max_decode_p99_regression_pct":5
})).unwrap()
}
fn status(writes: u64, reserved: u64, used: u64, manifests: u64) -> Status {
    serde_json::from_value(json!({"version":1,"effective":{"state":"active"},"activity":{"fills":0,"hits":0,"misses":0,"writes":writes,"bytes_read":0,"bytes_written":100,"evictions":0,"corrupt_entries":0},"usage":{"manifests":manifests,"used_bytes":used,"reserved_inflight_bytes":reserved}})).unwrap()
}
fn request(source: &str, id: &str, ttft: f64) -> Request {
    serde_json::from_value(json!({"request_id":id,"session_id":source,"source_dataset":source,"assistant_turn":1,"prompt_tokens":19000,"ttft_seconds":ttft,"content_sha256":"matching-content"})).unwrap()
}
fn phase() -> Phase {
    Phase {
        requests: vec![request("buzz", "buzz:1", 2.0)],
        ..Default::default()
    }
}
fn passing() -> Run {
    let sources = ["buzz", "opencode", "goose"];
    let mut phases = BTreeMap::new();
    let mut cold = Phase::default();
    let mut restart = Phase::default();
    for source in sources {
        cold.requests
            .push(request(source, &format!("{source}:off:1"), 2.0));
        restart
            .requests
            .push(request(source, &format!("{source}:restart:1"), 0.5));
    }
    restart.activity_deltas = vec![Activity {
        fills: 3,
        bytes_read: 300,
        ..Default::default()
    }];
    phases.insert("disk_off_cold".into(), cold);
    phases.insert("restart_l3".into(), restart);
    for name in [
        "disk_on_empty",
        "multi_turn_growth",
        "same_process_l1",
        "concurrent_fill",
        "concurrent_record",
        "low_space",
        "lifecycle_under_traffic",
    ] {
        phases.insert(name.into(), phase());
    }
    phases.get_mut("disk_on_empty").unwrap().activity_delta = Some(Activity {
        writes: 1,
        bytes_written: 100,
        ..Default::default()
    });
    let growth = phases.get_mut("multi_turn_growth").unwrap();
    growth.requests = sources
        .iter()
        .map(|source| request(source, &format!("{source}:1"), 2.0))
        .collect();
    growth.activity_delta = Some(Activity {
        writes: 3,
        ..Default::default()
    });
    phases.get_mut("same_process_l1").unwrap().activity_delta = Some(Activity::default());
    let fill = phases.get_mut("concurrent_fill").unwrap();
    fill.requests.push(request("buzz", "buzz:fill:2", 0.5));
    fill.activity_delta = Some(Activity {
        fills: 1,
        bytes_read: 100,
        ..Default::default()
    });
    let record = phases.get_mut("concurrent_record").unwrap();
    record.requests.push(request("buzz", "buzz:record:2", 2.0));
    record.activity_delta = Some(Activity {
        writes: 1,
        bytes_written: 120,
        ..Default::default()
    });
    let low = phases.get_mut("low_space").unwrap();
    low.activity_delta = Some(Activity::default());
    let mut low_status = status(0, 0, 0, 0);
    low_status.effective.state = "read_only_low_space".into();
    low.status_after = Some(low_status);
    let traffic = phases.get_mut("lifecycle_under_traffic").unwrap();
    let operation = Operation {
        status: status(0, 0, 0, 0),
        freed_bytes: 0,
    };
    traffic.prune = Some(operation.clone());
    traffic.clear = Some(operation.clone());
    traffic.final_clear = Some(operation);
    for (disk, p99) in [("off", 0.1), ("on", 0.104)] {
        let mut high = phase();
        high.summary = Some(Summary {
            failed_requests: 0,
            decode_inter_token_p99_seconds: Some(p99),
            content_sha256_by_request: BTreeMap::from([(
                "buzz:1".into(),
                "matching-content".into(),
            )]),
        });
        phases.insert(format!("high_load_{disk}_c1"), high);
    }
    Run {
        schema_version: 1,
        kind: "disk-l3-lifecycle".into(),
        config: config(),
        build: json!({"commit":"fixture"}),
        inputs: json!({}),
        phases,
        completed_at: Some("2026-10-02T00:00:00+00:00".into()),
        gates: None,
        evidence: Default::default(),
    }
}
#[test]
fn complete_lifecycle_evidence_passes_all_gates() {
    assert!(l3_gates::evaluate(&passing()).unwrap().passed);
}
#[test]
fn gates_reject_fill_record_output_prompt_restart_lowspace_and_decode_failures() {
    type InvalidRun = (&'static str, fn(&mut Run));
    let cases: [InvalidRun; 8] = [
        ("single_physical_fill", |r| {
            r.phases
                .get_mut("concurrent_fill")
                .unwrap()
                .activity_delta
                .as_mut()
                .unwrap()
                .fills = 2
        }),
        ("single_physical_write", |r| {
            r.phases
                .get_mut("concurrent_record")
                .unwrap()
                .activity_delta
                .as_mut()
                .unwrap()
                .bytes_written = 121
        }),
        ("output_identity", |r| {
            r.phases.get_mut("same_process_l1").unwrap().requests[0].content_sha256 =
                Some("changed".into())
        }),
        ("prompt_token_range", |r| {
            r.phases.get_mut("restart_l3").unwrap().requests[0].prompt_tokens = Some(24001)
        }),
        ("every_restart_reads_l3", |r| {
            r.phases.get_mut("restart_l3").unwrap().activity_deltas[0].fills = 0
        }),
        ("low_space_falls_back_cold", |r| {
            r.phases
                .get_mut("low_space")
                .unwrap()
                .activity_delta
                .as_mut()
                .unwrap()
                .writes = 1
        }),
        ("high_load_c1", |r| {
            r.phases
                .get_mut("high_load_on_c1")
                .unwrap()
                .summary
                .as_mut()
                .unwrap()
                .decode_inter_token_p99_seconds = Some(0.106)
        }),
        ("lifecycle_under_traffic", |r| {
            r.phases
                .get_mut("lifecycle_under_traffic")
                .unwrap()
                .final_clear
                .as_mut()
                .unwrap()
                .status
                .usage
                .as_mut()
                .unwrap()
                .reserved_inflight_bytes = 1
        }),
    ];
    for (name, mutate) in cases {
        let mut run = passing();
        mutate(&mut run);
        let gates = l3_gates::evaluate(&run).unwrap();
        assert!(!gates.passed, "{name}");
        assert!(
            !gates.checks.iter().find(|c| c.name == name).unwrap().passed,
            "{name}"
        );
    }
}
#[test]
fn missing_empty_and_nonfinite_evidence_fail_closed() {
    let mut run = passing();
    run.phases.remove("concurrent_record");
    assert!(l3_gates::evaluate(&run).is_err());
    let mut run = passing();
    run.phases
        .get_mut("disk_off_cold")
        .unwrap()
        .requests
        .clear();
    assert!(!l3_gates::evaluate(&run).unwrap().passed);
    let mut run = passing();
    run.phases.get_mut("restart_l3").unwrap().requests[0].ttft_seconds = Some(f64::NAN);
    assert!(!l3_gates::evaluate(&run).unwrap().passed);
    let mut run = passing();
    run.completed_at = None;
    assert!(!l3_gates::evaluate(&run).unwrap().passed);
}
#[test]
fn counter_reset_and_invalid_config_are_rejected() {
    assert!(
        Activity::default()
            .delta(&Activity {
                writes: 1,
                ..Default::default()
            })
            .is_err()
    );
    let mut config = config();
    config.concurrency = vec![64, 128, 256];
    assert!(config.validate(true).is_ok());
    config.disk_budget = "18446744073709551615TiB".into();
    assert!(config.validate(true).is_err());
    config.disk_budget = "auto".into();
    config.max_l3_ttft_ratio = f64::NAN;
    assert!(config.validate(true).is_err());
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}
async fn responder(
    listener: tokio::net::TcpListener,
    replies: Vec<(&str, serde_json::Value)>,
) -> Vec<String> {
    let mut requests = Vec::new();
    for (expected, body) in replies {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let mut chunk = [0; 2048];
        loop {
            let length = socket.read(&mut chunk).await.unwrap();
            assert!(length > 0);
            bytes.extend_from_slice(&chunk[..length]);
            if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        let received = String::from_utf8(bytes).unwrap();
        assert!(received.starts_with(expected));
        requests.push(received);
        let body = serde_json::to_vec(&body).unwrap();
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        socket.write_all(&body).await.unwrap();
    }
    requests
}
#[test]
fn management_http_waits_for_stable_writer_then_four_empty_clear_receipts() {
    runtime().block_on(async {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let client = Client {
            base: format!("http://{}", listener.local_addr().unwrap()),
            timeout: Duration::from_secs(2),
            interval: Duration::from_millis(1),
            cancellation: None,
        };
        let mut replies = vec![
            (
                "GET /api/runtime/kv-cache ",
                serde_json::to_value(status(1, 5, 100, 1)).unwrap(),
            ),
            (
                "GET /api/runtime/kv-cache ",
                serde_json::to_value(status(1, 0, 100, 1)).unwrap(),
            ),
            (
                "GET /api/runtime/kv-cache ",
                serde_json::to_value(status(1, 0, 100, 1)).unwrap(),
            ),
        ];
        for manifests in [0, 1, 0, 0, 0, 0] {
            replies.push((
                "DELETE /api/runtime/kv-cache ",
                json!({"status":status(1,0,0,manifests),"freed_bytes":0}),
            ));
        }
        let server = responder(listener, replies);
        let measure = async {
            let committed = client.committed(1).await.unwrap();
            assert_eq!(committed.usage.unwrap().reserved_inflight_bytes, 0);
            let empty = client.empty().await.unwrap();
            assert_eq!(empty.status.usage.unwrap().manifests, 0);
        };
        let (requests, ()) = tokio::join!(server, measure);
        assert_eq!(requests.len(), 9);
    });
}
#[test]
fn management_rejects_remote_endpoint_and_hung_server_with_finite_timeout() {
    runtime().block_on(async {
        let remote = Client {
            base: "http://192.0.2.1:1".into(),
            timeout: Duration::from_millis(10),
            interval: Duration::from_millis(1),
            cancellation: None,
        };
        assert!(remote.status().await.is_err());
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let client = Client {
            base: format!("http://{}", listener.local_addr().unwrap()),
            ..remote
        };
        let server = async {
            let (socket, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_millis(30)).await;
            drop(socket);
        };
        let (result, ()) = tokio::join!(client.status(), server);
        assert!(result.is_err());
    });
}
#[test]
fn persistent_disk_root_survives_separate_runtime_states_and_finishes_exact_owner() {
    let parent = tempfile::tempdir().unwrap();
    let sentinel = parent.path().join("unrelated");
    std::fs::write(&sentinel, b"keep").unwrap();
    let mut roots = DiskRoots::create(parent.path()).unwrap();
    let cache = roots.cache().to_owned();
    std::fs::write(cache.join("payload"), b"persist").unwrap();
    let first = roots.server_state().unwrap();
    let second = roots.server_state().unwrap();
    assert_ne!(first, second);
    std::fs::remove_dir(first).unwrap();
    assert_eq!(std::fs::read(cache.join("payload")).unwrap(), b"persist");
    roots.finish().unwrap();
    assert!(!cache.exists());
    assert_eq!(std::fs::read(sentinel).unwrap(), b"keep");
}

#[cfg(unix)]
#[test]
fn retained_process_restarts_use_same_disk_root_and_separate_runtime_states() {
    use crate::process::{
        self,
        retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
    };
    struct Owner {
        launch: Option<Launch>,
    }
    impl Coordinator for Owner {
        type Rejection = String;
        fn line(
            &mut self,
            _: MemberId,
            _: process::ObservedLine<'_>,
        ) -> process::ProbeDecision<String> {
            process::ProbeDecision::Pending
        }
        fn tick(&mut self, context: Context<'_>) -> Action<String> {
            if let Some(launch) = self.launch.take() {
                return Action::StartExpected {
                    launch,
                    policy: ExpectedExit::new(&[0], Duration::from_secs(2)).unwrap(),
                };
            }
            if context
                .members
                .iter()
                .any(|m| matches!(m.state, MemberState::ExpectedExit { .. }))
            {
                Action::Complete
            } else {
                Action::Pending
            }
        }
    }
    let parent = tempfile::tempdir().unwrap();
    let mut roots = DiskRoots::create(parent.path()).unwrap();
    for script in [
        "printf persist > \"$L3_CACHE/payload\"; printf first > \"$L3_STATE/receipt\"",
        "test \"$(cat \"$L3_CACHE/payload\")\" = persist; printf second > \"$L3_STATE/receipt\"",
    ] {
        let state = roots.server_state().unwrap();
        let launch = Launch {
            member: MemberId::Seed,
            spec: process::ProcessSpec {
                executable: "/bin/sh".into(),
                arguments: vec![
                    process::Value::Public("-c".into()),
                    process::Value::Public(script.into()),
                ],
                cwd: parent.path().into(),
                environment: std::collections::BTreeMap::from([
                    (
                        "L3_CACHE".into(),
                        process::Value::Public(roots.cache().as_os_str().into()),
                    ),
                    (
                        "L3_STATE".into(),
                        process::Value::Public(state.as_os_str().into()),
                    ),
                ]),
            },
            files: Default::default(),
            readiness_deadline: Duration::from_secs(2),
        };
        let report = process::retained::run(
            &mut Owner {
                launch: Some(launch),
            },
            &super::server_cell_worker::limits(Duration::from_secs(3)),
            &process::Cancellation::default(),
        )
        .unwrap();
        assert!(report.recovery_success());
        assert!(state.join("receipt").is_file());
        assert!(roots.cache().join("payload").is_file());
    }
    roots.finish().unwrap();
}

#[test]
fn management_prune_and_clear_use_owned_loopback_routes_and_typed_receipts() {
    runtime().block_on(async {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let client = Client {
            base: format!("http://{}", listener.local_addr().unwrap()),
            timeout: Duration::from_secs(1),
            interval: Duration::from_millis(1),
            cancellation: None,
        };
        let operation = json!({"status":status(1,0,0,0),"freed_bytes":100});
        let server = responder(
            listener,
            vec![
                ("POST /api/runtime/kv-cache/prune ", operation.clone()),
                ("DELETE /api/runtime/kv-cache ", operation),
            ],
        );
        let measure = async {
            assert_eq!(client.prune().await.unwrap().freed_bytes, 100);
            assert_eq!(
                client
                    .clear()
                    .await
                    .unwrap()
                    .status
                    .usage
                    .unwrap()
                    .manifests,
                0
            );
        };
        let (requests, ()) = tokio::join!(server, measure);
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("content-type: application/json")
        );
    });
}

#[test]
fn actual_local_sse_final_checkpoint_records_content_usage_and_pure_raw_jsonl() {
    runtime().block_on(async {
        let listener=tokio::net::TcpListener::bind(("127.0.0.1",0)).await.unwrap(); let base=format!("http://{}/v1",listener.local_addr().unwrap());
        let trajectory:super::recorded_requests::Trajectory=serde_json::from_value(json!({"session_id":"buzz","source_dataset":"buzz","agent_framework":"goose","messages":[{"role":"user","content":"captured prompt"},{"role":"assistant","content":"recorded answer"}]})).unwrap();
        let server=async {
            let (mut socket,_)=listener.accept().await.unwrap(); let mut bytes=Vec::new(); let mut chunk=[0;4096];
            loop {
                let length=socket.read(&mut chunk).await.unwrap(); assert!(length>0); bytes.extend_from_slice(&chunk[..length]);
                if let Some(end)=bytes.windows(4).position(|w|w==b"\r\n\r\n") {
                    let headers=String::from_utf8_lossy(&bytes[..end]); let length=headers.lines().find_map(|line|line.to_ascii_lowercase().strip_prefix("content-length:").map(|v|v.trim().parse::<usize>().unwrap())).unwrap();
                    if bytes.len()>=end+4+length { break; }
                }
            }
            let end=bytes.windows(4).position(|w|w==b"\r\n\r\n").unwrap(); let body:serde_json::Value=serde_json::from_slice(&bytes[end+4..]).unwrap();
            assert_eq!(body["temperature"],0); assert_eq!(body["seed"],42); assert_eq!(body["messages"][0]["content"],"captured prompt");
            let sse="data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":19000,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n";
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",sse.len()).as_bytes()).await.unwrap();
        };
        let output=tempfile::tempdir().unwrap(); let raw=output.path().join("requests.jsonl");
        let trajectories=[trajectory]; let config=config();
        let endpoint=super::l3_requests::Endpoint { base:&base,model:"admitted-serving-model",timeout:Duration::from_secs(1),cancellation:None };
        let measure=super::l3_requests::measure(&endpoint,&trajectories,&config,super::l3_execution::Mode::Final,"fixture",&raw);
        let ((),phase)=tokio::join!(server,measure); let phase=phase.unwrap(); assert!(phase.requests[0].error.is_none(), "request error: {:?}", phase.requests[0].error); assert_eq!(phase.requests[0].prompt_tokens,Some(19000));
        let raw=std::fs::read_to_string(raw).unwrap(); assert_eq!(raw.lines().count(),1); let record:serde_json::Value=serde_json::from_str(raw.trim()).unwrap(); assert!(record.get("body").is_none()); assert!(record.get("event").is_none()); assert_eq!(record["request_id"],"buzz:0");
    });
}

struct PhaseDriver {
    reference: Run,
    status: Status,
    starts: Vec<std::path::PathBuf>,
    active: bool,
    trace: Vec<String>,
    checkpoint: std::path::PathBuf,
    fail: Option<&'static str>,
}
impl super::l3_execution::Driver for PhaseDriver {
    async fn start(
        &mut self,
        disk: super::l3_execution::Disk,
        roots: &mut DiskRoots,
    ) -> crate::command::DynResult<()> {
        if self.active {
            return Err("driver started overlapping owned servers".into());
        }
        self.active = true;
        self.starts.push(roots.server_state()?);
        self.status = status(0, 0, 0, 0);
        self.status.effective.state = match disk {
            super::l3_execution::Disk::Off => "off",
            super::l3_execution::Disk::On => "active",
            super::l3_execution::Disk::LowSpace => "read_only_low_space",
        }
        .into();
        self.trace.push("start".into());
        Ok(())
    }
    async fn stop(&mut self) -> crate::command::DynResult<()> {
        self.active = false;
        self.trace.push("stop".into());
        Ok(())
    }
    async fn status(&mut self) -> crate::command::DynResult<Status> {
        assert!(self.active);
        Ok(self.status.clone())
    }
    async fn committed(&mut self, writes: u64) -> crate::command::DynResult<Status> {
        assert!(self.status.activity.as_ref().unwrap().writes >= writes);
        Ok(self.status.clone())
    }
    async fn empty(&mut self) -> crate::command::DynResult<Operation> {
        self.status.usage = Some(Usage {
            manifests: 0,
            used_bytes: 0,
            reserved_inflight_bytes: 0,
        });
        self.trace.push("empty".into());
        Ok(Operation {
            status: self.status.clone(),
            freed_bytes: 0,
        })
    }
    async fn measure(
        &mut self,
        name: &str,
        mode: super::l3_execution::Mode,
    ) -> crate::command::DynResult<Phase> {
        self.trace.push(name.into());
        if self.fail == Some(name) {
            return Err("injected request failure".into());
        }
        let key = if name.starts_with("disk_off_cold_") {
            "disk_off_cold"
        } else if name.starts_with("restart_l3_") {
            "restart_l3"
        } else {
            name
        };
        let phase = if key == "disk_on_additional_sources" {
            Phase {
                requests: self.reference.phases["disk_off_cold"]
                    .requests
                    .iter()
                    .skip(1)
                    .cloned()
                    .collect(),
                activity_delta: Some(Activity {
                    writes: 2,
                    bytes_written: 200,
                    ..Default::default()
                }),
                ..Default::default()
            }
        } else {
            self.reference.phases[key].clone()
        };
        let delta = phase
            .activity_delta
            .as_ref()
            .or_else(|| phase.activity_deltas.first())
            .cloned()
            .unwrap_or_default();
        let activity = self.status.activity.as_mut().unwrap();
        activity.writes += delta.writes;
        activity.fills += delta.fills;
        activity.bytes_read += delta.bytes_read;
        activity.bytes_written += delta.bytes_written;
        if delta.writes > 0 {
            self.status.usage = Some(Usage {
                manifests: delta.writes,
                used_bytes: delta.bytes_written.max(100),
                reserved_inflight_bytes: 0,
            });
        }
        if let super::l3_execution::Mode::Identical(count) = mode {
            assert_eq!(phase.requests.len(), count);
        }
        Ok(phase)
    }
    async fn traffic(&mut self) -> crate::command::DynResult<Phase> {
        self.trace.push("traffic".into());
        Ok(self.reference.phases["lifecycle_under_traffic"].clone())
    }
    fn checkpoint(&mut self, run: &Run) -> crate::command::DynResult<()> {
        crate::command::write_json_file(&self.checkpoint, run)
    }
}
fn driver(output: &std::path::Path) -> PhaseDriver {
    PhaseDriver {
        reference: passing(),
        status: status(0, 0, 0, 0),
        starts: Vec::new(),
        active: false,
        trace: Vec::new(),
        checkpoint: output.join("run.json"),
        fail: None,
    }
}
#[test]
fn phase_orchestration_preserves_restarts_and_checkpoints_completed_gates() {
    runtime().block_on(async {
        let parent = tempfile::tempdir().unwrap();
        let mut roots = DiskRoots::create(parent.path()).unwrap();
        let cache = roots.cache().to_owned();
        let mut driver = driver(parent.path());
        let mut run = passing();
        run.phases.clear();
        run.completed_at = None;
        super::l3_execution::execute(&mut driver, &mut roots, &mut run)
            .await
            .unwrap();
        assert!(!driver.active);
        assert!(run.gates.as_ref().unwrap().passed);
        assert!(run.completed_at.is_some());
        assert_eq!(driver.starts.len(), 11);
        assert_eq!(
            driver
                .starts
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            11
        );
        assert_eq!(
            driver
                .trace
                .iter()
                .filter(|v| v.as_str() == "empty")
                .count(),
            2
        );
        assert!(cache.is_dir());
        let saved: Run =
            serde_json::from_slice(&std::fs::read(parent.path().join("run.json")).unwrap())
                .unwrap();
        assert!(saved.gates.unwrap().passed);
        roots.finish().unwrap();
    });
}
#[test]
fn phase_failure_always_finalizes_owned_server_and_preserves_previous_evidence() {
    runtime().block_on(async {
        let parent = tempfile::tempdir().unwrap();
        let mut roots = DiskRoots::create(parent.path()).unwrap();
        let mut driver = driver(parent.path());
        driver.fail = Some("multi_turn_growth");
        let mut run = passing();
        run.phases.clear();
        run.completed_at = None;
        assert!(
            super::l3_execution::execute(&mut driver, &mut roots, &mut run)
                .await
                .is_err()
        );
        assert!(!driver.active);
        assert_eq!(driver.trace.last().unwrap(), "stop");
        assert!(run.completed_at.is_none());
        let saved: Run =
            serde_json::from_slice(&std::fs::read(parent.path().join("run.json")).unwrap())
                .unwrap();
        assert!(saved.phases.contains_key("disk_off_cold"));
        assert!(saved.gates.is_none());
        roots.finish().unwrap();
    });
}

#[cfg(unix)]
#[test]
fn production_session_adapter_gracefully_stops_and_joins_actual_retained_child() {
    use crate::process::{
        self,
        retained::{Launch, MemberId},
    };
    let parent = tempfile::tempdir().unwrap();
    let marker = parent.path().join("started");
    std::thread::scope(|scope| {
        let launch=Launch { member:MemberId::Seed,spec:process::ProcessSpec { executable:"/bin/sh".into(),arguments:vec![process::Value::Public("-c".into()),process::Value::Public("trap 'exit 0' TERM; printf started > \"$L3_MARKER\"; while :; do sleep 0.05; done".into())],cwd:parent.path().into(),environment:std::collections::BTreeMap::from([("L3_MARKER".into(),process::Value::Public(marker.as_os_str().into())),("PATH".into(),process::Value::Public("/bin:/usr/bin".into()))]) },files:Default::default(),readiness_deadline:Duration::from_secs(2) };
        let session = super::l3_server::Session::start(
            scope,
            launch,
            Duration::from_secs(3),
            process::Cancellation::default(),
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !marker.is_file() {
            assert!(std::time::Instant::now() < deadline);
            assert!(!session.finished());
            std::thread::sleep(Duration::from_millis(5));
        }
        session.admit();
        std::thread::sleep(Duration::from_millis(30));
        let report = session.finish().unwrap();
        assert!(report.recovery_success());
        assert_eq!(report.members.len(), 1);
        assert!(report.members[0].process.cleanup.complete);
        assert!(!report.members[0].process.cleanup.forced);
    });
}
#[cfg(unix)]
#[test]
fn production_session_adapter_observes_unexpected_early_exit_and_joins() {
    use crate::process::{
        self,
        retained::{Launch, MemberId},
    };
    let parent = tempfile::tempdir().unwrap();
    std::thread::scope(|scope| {
        let launch = Launch {
            member: MemberId::Seed,
            spec: process::ProcessSpec {
                executable: "/bin/sh".into(),
                arguments: vec![
                    process::Value::Public("-c".into()),
                    process::Value::Public("exit 7".into()),
                ],
                cwd: parent.path().into(),
                environment: Default::default(),
            },
            files: Default::default(),
            readiness_deadline: Duration::from_secs(1),
        };
        let session = super::l3_server::Session::start(
            scope,
            launch,
            Duration::from_secs(2),
            process::Cancellation::default(),
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !session.finished() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        let report = session.finish().unwrap();
        assert!(!report.recovery_success());
        assert_eq!(
            report.members[0].process.status.as_ref().unwrap().code(),
            Some(7)
        );
        assert!(report.members[0].process.cleanup.complete);
    });
}
#[test]
fn captured_l3_manifest_rejects_cohort_provenance_capacity_and_source_mismatches() {
    let trajectory = |id: &str, source: &str| json!({"session_id":id,"source_dataset":source,"agent_framework":"goose","recorded_model":null,"messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]});
    let document = json!({"cohorts":{"l3":[trajectory("buzz","buzz"),trajectory("oc","opencode"),trajectory("goose","goose")],"1":[trajectory("high","buzz")]}});
    let manifest: super::manifest_preflight::Manifest =
        serde_json::from_value(document.clone()).unwrap();
    assert!(super::l3_manifest::validate(&manifest, &config()).is_ok());
    let mut duplicate = document.clone();
    duplicate["cohorts"]["1"][0]["session_id"] = "buzz".into();
    let manifest = serde_json::from_value(duplicate).unwrap();
    assert!(super::l3_manifest::validate(&manifest, &config()).is_err());
    let parent = tempfile::tempdir().unwrap();
    let source = parent.path().join("source.json");
    std::fs::write(&source, serde_json::to_vec(&document).unwrap()).unwrap();
    let output = parent.path().join("output");
    std::fs::create_dir(&output).unwrap();
    let (_, selected, input) = super::l3_manifest::import(&source, &output, &config()).unwrap();
    assert_eq!(selected.len(), 3);
    assert_eq!(input["source_manifest_sha256"], input["manifest_sha256"]);
    assert_eq!(selected[1].session_id, "oc");
    let mut missing = config();
    missing.required_sources[2] = "absent".into();
    assert!(super::l3_manifest::import(&source, &output, &missing).is_err());
}

#[test]
fn legacy_l3_report_preserves_read_only_receipt_without_importing_python() {
    let output = tempfile::tempdir().unwrap();
    let run = passing();
    let gates = l3_gates::evaluate(&run).unwrap();
    let old = serde_json::json!({"schema_version":1,"kind":"disk-l3-lifecycle","build":{"commit":"legacy"},"config":{"model":"legacy-model","backend":"metal","prompt_token_range":"18000:24000"},"gates":gates});
    let path = super::l3_report::legacy(output.path(), old.clone()).unwrap();
    let markdown = std::fs::read_to_string(path).unwrap();
    assert!(markdown.contains("legacy-model"));
    assert!(markdown.contains("**PASS**"));
    assert!(output.path().join("artifact-sha256.txt").is_file());
    let mut invalid = old;
    invalid["gates"]["passed"] = false.into();
    assert!(super::l3_report::legacy(output.path(), invalid).is_err());
}
#[test]
fn production_request_cancellation_retains_error_record_and_drops_stalled_http() {
    runtime().block_on(async {
        let listener=tokio::net::TcpListener::bind(("127.0.0.1",0)).await.unwrap(); let base=format!("http://{}/v1",listener.local_addr().unwrap());
        let cancellation=crate::process::Cancellation::default(); let cancel=cancellation.clone();
        let endpoint=super::l3_requests::Endpoint { base:&base,model:"admitted",timeout:Duration::from_secs(1),cancellation:Some(cancellation) };
        let trajectory=serde_json::from_value(json!({"session_id":"buzz","source_dataset":"buzz","agent_framework":"goose","recorded_model":null,"messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]})).unwrap();
        let trajectories=[trajectory]; let config=config(); let output=tempfile::tempdir().unwrap(); let raw=output.path().join("raw.jsonl");
        let server=async { let (mut socket,_)=listener.accept().await.unwrap(); let mut bytes=[0;512]; assert!(socket.read(&mut bytes).await.unwrap()>0); cancel.cancel(); tokio::time::sleep(Duration::from_millis(30)).await; };
        let request=super::l3_requests::measure(&endpoint,&trajectories,&config,super::l3_execution::Mode::Final,"cancel",&raw);
        let ((),phase)=tokio::join!(server,request); let phase=phase.unwrap(); assert!(phase.requests[0].error.as_ref().unwrap().contains("interrupted")); assert_eq!(std::fs::read_to_string(raw).unwrap().lines().count(),1);
    });
}
