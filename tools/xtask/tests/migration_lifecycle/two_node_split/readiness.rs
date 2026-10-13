use super::{
    fixture::{Fixture, repository, stderr, stdout},
    http_fixture::Server,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    time::{Duration, Instant},
};
fn payloads(observer: &str) -> BTreeMap<String, Vec<u8>> {
    [
        ("/api/status", "status"),
        ("/api/runtime/stages", "stages"),
        ("/v1/models", "models"),
    ]
    .into_iter()
    .map(|(endpoint, kind)| {
        let path = repository().join(format!(
            "tools/xtask/tests/fixtures/migration/split_evidence/{observer}-{kind}.json"
        ));
        (endpoint.into(), fs::read(path).unwrap())
    })
    .collect()
}
fn readiness(fixture: &Fixture, timeout: u8) -> String {
    fs::create_dir_all(fixture.root.join("work")).unwrap();
    let source = fixture.functions(&[("configure_split_evidence_paths", "start_node")]);
    format!(
        "{source}\nPRIMARY_MODEL_LABEL=dense\nMODEL_LABEL=dense\nREADINESS_TIMEOUT_SECONDS={timeout}\nSNAPSHOT_REQUEST_TIMEOUT_SECONDS=1\nSEED_PID=$$\nWORKER_PID=$$\nSEED_LOG=\"$WORK_DIR/seed.log\"\nWORKER_LOG=\"$WORK_DIR/worker.log\"\nprintf 'seed-fixture-log\\n' >\"$SEED_LOG\"\nprintf 'worker-fixture-log\\n' >\"$WORKER_LOG\"\nwait_for_split_topology \"\"\nprintf 'driver=%s;port=%s\\n' \"$DRIVER_LABEL\" \"$DRIVER_API_PORT\"\n"
    )
}
fn ports(seed: &Server, worker: &Server) -> Vec<(&'static str, String)> {
    [
        ("SEED_API_PORT", seed.address.port()),
        ("SEED_CONSOLE_PORT", seed.address.port()),
        ("WORKER_API_PORT", worker.address.port()),
        ("WORKER_CONSOLE_PORT", worker.address.port()),
    ]
    .into_iter()
    .map(|(key, port)| (key, port.to_string()))
    .collect()
}
fn snapshots(fixture: &Fixture) {
    let root = fixture.root.join("work/split-evidence-snapshots");
    let mut names = fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(
        names,
        [
            "seed-models.json",
            "seed-stages.json",
            "seed-status.json",
            "worker-models.json",
            "worker-stages.json",
            "worker-status.json"
        ]
    );
    for name in names {
        let _: Value = serde_json::from_slice(&fs::read(root.join(name)).unwrap()).unwrap();
    }
}
#[test]
fn actual_split_snapshot_http_capture_persists_only_status_identity_and_removes_raw_transfer() {
    let fixture = Fixture::new();
    let body = json!({"node_id":"seed-node","mesh_id":"mesh-a","token":"top-secret","storage_path":"/private/model/path","nested":{"api_key":"nested-secret","path":"/nested/path"},"peers":[{"id":"worker-node","token":"peer-secret","materialized_path":"/peer/path"}]});
    let server = Server::start(
        BTreeMap::from([("/api/status".into(), serde_json::to_vec(&body).unwrap())]),
        Duration::ZERO,
    );
    let output = fixture.root.join("status.json");
    let script = fixture.functions(&[("capture_json_snapshot", "capture_split_snapshots")])
        + "capture_json_snapshot status \"$URL\" \"$OUTPUT\" 2\n";
    let result = fixture.run(
        script,
        &[
            ("URL", format!("http://{}/api/status", server.address)),
            ("OUTPUT", output.display().to_string()),
        ],
    );
    assert!(result.process.success(), "{}", stderr(&result));
    let bytes = fs::read(&output).unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        value,
        json!({"mesh_id":"mesh-a","node_id":"seed-node","peers":[{"id":"worker-node"}]})
    );
    let raw = String::from_utf8(bytes).unwrap();
    for secret in [
        "top-secret",
        "nested-secret",
        "peer-secret",
        "/private/model/path",
        "/nested/path",
        "/peer/path",
    ] {
        assert!(!raw.contains(secret));
    }
    assert!(!fs::read_dir(&fixture.root).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".curl.")
    }));
    assert_eq!(server.calls(), ["/api/status"]);
}
#[test]
fn actual_split_readiness_captures_both_observers_and_binds_the_ready_stage_zero_driver() {
    let fixture = Fixture::new();
    let seed = Server::start(payloads("seed"), Duration::ZERO);
    let worker = Server::start(payloads("worker"), Duration::ZERO);
    let result = fixture.run(readiness(&fixture, 5), &ports(&seed, &worker));
    assert!(result.process.success(), "{}", stderr(&result));
    assert!(stdout(&result).contains("Selected seed as stage-0 OpenAI driver"));
    assert!(stdout(&result).contains(&format!("driver=seed;port={}", seed.address.port())));
    snapshots(&fixture);
    let evidence: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("work/split-evidence.json")).unwrap())
            .unwrap();
    assert_eq!(evidence["status"], "ready");
    for server in [&seed, &worker] {
        let mut calls = server.calls();
        calls.sort();
        assert_eq!(calls, ["/api/runtime/stages", "/api/status", "/v1/models"]);
    }
}
#[test]
fn actual_split_hung_http_snapshots_respect_the_total_deadline_and_retain_all_six_failures() {
    let fixture = Fixture::new();
    let seed = Server::start(payloads("seed"), Duration::from_secs(5));
    let worker = Server::start(payloads("worker"), Duration::from_secs(5));
    let started = Instant::now();
    let result = fixture.run(readiness(&fixture, 2), &ports(&seed, &worker));
    assert!(!result.process.success());
    assert!(started.elapsed() < Duration::from_secs(4));
    assert!(stderr(&result).contains("timed out after 2s"));
    for diagnostic in [
        "seed log tail at timeout",
        "worker log tail at timeout",
        "seed-fixture-log",
        "worker-fixture-log",
    ] {
        assert!(stderr(&result).contains(diagnostic), "missing {diagnostic}");
    }
    assert!(!stdout(&result).contains("driver="));
    snapshots(&fixture);
    let evidence: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("work/split-evidence.json")).unwrap())
            .unwrap();
    assert_eq!(evidence["status"], "failed");
    for server in [&seed, &worker] {
        assert!(server.calls().len() >= 3);
    }
}
#[test]
fn actual_split_disagreeing_observer_evidence_never_admits_a_driver() {
    let fixture = Fixture::new();
    let seed = Server::start(payloads("seed"), Duration::ZERO);
    let mut different = payloads("worker");
    let mut stages: Value =
        serde_json::from_slice(different.get("/api/runtime/stages").unwrap()).unwrap();
    stages["topologies"][0]["run_id"] = "foreign-run".into();
    different.insert(
        "/api/runtime/stages".into(),
        serde_json::to_vec(&stages).unwrap(),
    );
    let worker = Server::start(different, Duration::ZERO);
    let result = fixture.run(readiness(&fixture, 2), &ports(&seed, &worker));
    assert!(!result.process.success());
    assert!(!stdout(&result).contains("driver="));
    snapshots(&fixture);
    let evidence: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("work/split-evidence.json")).unwrap())
            .unwrap();
    assert_eq!(evidence["status"], "failed");
}
