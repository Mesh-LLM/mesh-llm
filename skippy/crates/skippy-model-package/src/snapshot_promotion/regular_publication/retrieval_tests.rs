use super::super::policy::ArtifactIdentity;
use super::fixture::{Reply, Server};
use super::*;
use sha2::Digest as _;
use std::time::Duration;
fn publisher(server: &Server) -> Publisher {
    fixture_retrieval_publisher(&server.endpoint)
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}
fn plan() -> Plan {
    Plan {
        repo: "fixture/evidence".into(),
        parent_commit: "a".repeat(40),
        paths: vec!["runs/native-job.json".into()],
    }
}
fn identity(bytes: &[u8]) -> ArtifactIdentity {
    ArtifactIdentity {
        sha256: sha2::Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        byte_size: bytes.len() as u64,
    }
}
#[test]
fn immutable_regular_receipt_retrieval_consumes_exact_bytes_at_commit_and_retains_auth_scope() {
    let body = b"{\"schema_version\":1}";
    let server = Server::start(vec![Reply {
        status: 200,
        body: body.to_vec(),
        hold: false,
        changed_file: None,
    }]);
    let bytes = runtime()
        .block_on(publisher(&server).retrieve_json_until(
            &plan(),
            "runs/native-job.json",
            &"b".repeat(40),
            &identity(body),
            Instant::now() + Duration::from_secs(2),
            std::future::pending(),
        ))
        .unwrap();
    assert_eq!(bytes, body);
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let request = String::from_utf8_lossy(&requests[0]);
    assert!(request.starts_with(&format!(
        "GET /fixture/evidence/resolve/{}/runs/native-job.json ",
        "b".repeat(40)
    )));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer fixture-token")
    );
}
#[test]
fn immutable_regular_receipt_retrieval_refuses_changed_excess_status_and_immediate_cancel() {
    for (status, body) in [
        (200, b"wrong".to_vec()),
        (200, b"too-long".to_vec()),
        (503, b"private-error-body".to_vec()),
    ] {
        let server = Server::start(vec![Reply {
            status,
            body,
            hold: false,
            changed_file: None,
        }]);
        let error = runtime()
            .block_on(publisher(&server).retrieve_json_until(
                &plan(),
                "runs/native-job.json",
                &"b".repeat(40),
                &identity(b"right"),
                Instant::now() + Duration::from_secs(2),
                std::future::pending(),
            ))
            .err()
            .unwrap();
        assert!(!error.to_string().contains("private-error-body"));
    }
    let server = Server::start(vec![]);
    assert!(
        runtime()
            .block_on(publisher(&server).retrieve_json_until(
                &plan(),
                "runs/native-job.json",
                &"b".repeat(40),
                &identity(b"right"),
                Instant::now() + Duration::from_secs(2),
                async {}
            ))
            .is_err()
    );
    assert!(server.requests.lock().unwrap().is_empty());
}
#[test]
fn immutable_regular_receipt_retrieval_deadline_bounds_held_response() {
    let server = Server::start(vec![Reply {
        status: 200,
        body: b"right".to_vec(),
        hold: true,
        changed_file: None,
    }]);
    let error = runtime()
        .block_on(publisher(&server).retrieve_json_until(
            &plan(),
            "runs/native-job.json",
            &"b".repeat(40),
            &identity(b"right"),
            Instant::now() + Duration::from_millis(300),
            std::future::pending(),
        ))
        .err()
        .unwrap();
    assert!(error.to_string().contains("deadline"));
    assert_eq!(server.requests.lock().unwrap().len(), 1);
}

#[test]
fn regular_publication_observer_preserves_known_commit_under_owned_cancel_or_deadline() {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    for cancelled in [true, false] {
        let root = tempfile::tempdir().unwrap();
        let bytes = b"{\"v\":1}";
        let path = root.path().join("native-job.json");
        std::fs::write(&path, bytes).unwrap();
        let plan = plan();
        let server=Server::start(vec![
            Reply{status:200,body:serde_json::to_vec(&serde_json::json!({"files":[{"path":"runs/native-job.json","uploadMode":"regular","shouldIgnore":false}]})).unwrap(),hold:false,changed_file:None},
            Reply{status:200,body:serde_json::to_vec(&serde_json::json!({"commitOid":"d".repeat(40)})).unwrap(),hold:false,changed_file:None},
            Reply{status:200,body:bytes.to_vec(),hold:true,changed_file:None},
        ]);
        let known = Arc::new(AtomicBool::new(false));
        let signal = known.clone();
        let cancellation = async {
            if !cancelled {
                std::future::pending::<()>().await;
            }
            while !signal.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        };
        let mut snapshots = Vec::new();
        let mut observer = |receipt: &Receipt| {
            snapshots.push(serde_json::to_value(receipt).unwrap());
            if receipt.commit_oid.is_some() {
                known.store(true, Ordering::SeqCst);
            }
            Ok(())
        };
        let result = runtime().block_on(publisher(&server).publish_observed_until(
            &plan,
            vec![LocalFile {
                path_in_repo: "runs/native-job.json".into(),
                file: std::fs::File::open(&path).unwrap(),
                identity: identity(bytes),
            }],
            Instant::now() + Duration::from_millis(700),
            cancellation,
            &mut observer,
        ));
        assert!(known.load(Ordering::SeqCst));
        assert!(!result.completed && result.mutation_attempted);
        assert_eq!(result.commit_oid, Some("d".repeat(40)));
        assert!(result.error.is_some());
        assert!(
            snapshots
                .iter()
                .any(|s| s["mutation_attempted"] == true && s["commit_oid"].is_null())
        );
        assert!(
            snapshots
                .iter()
                .any(|s| s["commit_oid"] == "d".repeat(40) && s["completed"] == false)
        );
        drop(server);
        root.close().unwrap();
    }
}

#[test]
fn regular_publication_expired_progress_callback_refuses_before_commit_send() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"{\"v\":1}";
    let path = root.path().join("native-job.json");
    std::fs::write(&path, bytes).unwrap();
    let plan = plan();
    let server=Server::start(vec![Reply{status:200,body:serde_json::to_vec(&serde_json::json!({"files":[{"path":"runs/native-job.json","uploadMode":"regular","shouldIgnore":false}]})).unwrap(),hold:false,changed_file:None}]);
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut called = false;
    let mut observer = |receipt: &Receipt| {
        called = true;
        assert!(receipt.mutation_attempted && receipt.commit_oid.is_none());
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            std::thread::park_timeout(remaining);
        }
        Ok(())
    };
    let receipt = runtime().block_on(publisher(&server).publish_observed_until(
        &plan,
        vec![LocalFile {
            path_in_repo: "runs/native-job.json".into(),
            file: std::fs::File::open(path).unwrap(),
            identity: identity(bytes),
        }],
        deadline,
        std::future::pending(),
        &mut observer,
    ));
    assert!(called && !receipt.completed && receipt.mutation_attempted);
    assert!(receipt.commit_oid.is_none() && receipt.error.is_some());
    let requests = server.finish();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with(b"POST /api/models/fixture/evidence/preupload/"));
    root.close().unwrap();
}
