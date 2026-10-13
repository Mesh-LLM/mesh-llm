use super::*;
use crate::process::{Cancellation, Outcome, OutputFiles};

#[test]
fn d12_zero_retention_still_admits_owned_raw_correlation() {
    let directory = tempfile::tempdir().unwrap();
    let native = directory.path().join("native");
    std::fs::create_dir(&native).unwrap();
    let fixture = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples")
        .join(format!(
            "migration_daemon_fixture{}",
            std::env::consts::EXE_SUFFIX
        ));
    let plan = serde_json::json!({
        "status": "{\"api_port\":API,\"local_instances\":[{\"pid\":PID,\"is_self\":true}]}",
        "status_failures": 0, "models_code": 200, "models_body": [], "wire": null,
        "record": "{\"request_id\":\"ID\",\"source\":\"direct_http\",\"route\":\"models\",\"method\":\"GET\",\"request_kind\":\"model_listing\",\"status_code\":200,\"event\":\"request_completed\",\"outcome\":\"completed\"}\n",
        "before_body": true, "behavior": "Clean"
    });
    std::fs::write(
        native.join("daemon.json"),
        serde_json::to_vec(&plan).unwrap(),
    )
    .unwrap();
    let options = Options {
        binary: fixture,
        native_runtime_root: native,
        state_parent: directory.path().to_owned(),
        ready_max_wait: Duration::from_secs(3),
        shutdown_max_wait: Duration::from_secs(1),
    };
    let state = PrivateState::create(directory.path(), "zero-retention").unwrap();
    state.prepare().unwrap();
    let reservation = Reservation::acquire().unwrap();
    let ports = reservation.ports;
    let spec = daemon_spec(directory.path(), &options, (&state, ports));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut limits = limits(&options);
    limits.retained_bytes_per_stream = 0;
    std::thread::scope(|scope| {
        let (requests, work) = mpsc::sync_channel(1);
        let (results, responses) = mpsc::sync_channel(1);
        let worker = scope.spawn(move || http::work(ports, (work, results), runtime));
        let mut observer = Observer::new(requests, responses, RequestId::generate().unwrap(), 3);
        drop(reservation);
        let report = process::supervise_with_probe(
            &spec,
            &limits,
            &Cancellation::default(),
            OutputFiles::default(),
            process::Probe {
                observer: &mut observer,
                deadline: options.ready_max_wait,
            },
        )
        .unwrap();
        let facts = observer.facts();
        drop(observer);
        worker.join().unwrap();
        assert_eq!(report.process.outcome, Outcome::Ready);
        assert!(report.process.stdout.bytes_retained.is_empty());
        assert!(report.process.stdout.bytes_seen > 0);
        receipt::accept(report, facts, (false, true))
            .unwrap()
            .finish(Ok(()))
            .unwrap();
    });
    state.finish(Ok::<(), Error>(())).unwrap();
}
