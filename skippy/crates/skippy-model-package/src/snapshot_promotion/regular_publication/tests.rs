use super::super::policy::ArtifactIdentity;
use super::fixture::{Reply, Server};
use super::*;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Duration};
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn reply(body: serde_json::Value) -> Reply {
    Reply {
        status: 200,
        body: serde_json::to_vec(&body).unwrap(),
        hold: false,
        changed_file: None,
    }
}
fn classification(mode: &str) -> Reply {
    reply(
        serde_json::json!({"files":[{"path":"config.json","uploadMode":mode,"shouldIgnore":false}]}),
    )
}
fn commit() -> Reply {
    reply(serde_json::json!({"commitOid":"c".repeat(40)}))
}
fn fixture_file(root: &Path) -> (Plan, Vec<LocalFile>) {
    let path = root.join("local.json");
    std::fs::write(&path, b"{\"v\":1}").unwrap();
    (
        Plan {
            repo: "fixture/model".into(),
            parent_commit: "a".repeat(40),
            paths: vec!["config.json".into()],
        },
        vec![LocalFile {
            path_in_repo: "config.json".into(),
            file: std::fs::File::open(path).unwrap(),
            identity: ArtifactIdentity {
                byte_size: 7,
                sha256: hash(b"{\"v\":1}"),
            },
        }],
    )
}
fn publisher(server: &Server) -> Publisher {
    Publisher {
        http: reqwest::Client::builder()
            .no_proxy()
            .pool_max_idle_per_host(0)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap(),
        origin: reqwest::Url::parse(&server.endpoint).unwrap(),
        token: Secret::new("fixture-secret".into()).unwrap(),
    }
}
fn runtime() -> tokio::runtime::Runtime {
    let _ = skippy_model_hf::configure_hf_tls_provider();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}
#[test]
fn regular_publication_streams_parent_bound_payload_and_verifies_immutable_bytes() {
    let root = tempfile::tempdir().unwrap();
    let (plan, files) = fixture_file(root.path());
    let server = Server::start(vec![
        classification("regular"),
        commit(),
        Reply {
            status: 200,
            body: b"{\"v\":1}".to_vec(),
            hold: false,
            changed_file: None,
        },
    ]);
    let receipt = runtime().block_on(publisher(&server).publish_until(
        &plan,
        files,
        Instant::now() + Duration::from_secs(10),
        futures::future::pending(),
    ));
    assert!(receipt.completed && receipt.source_custody_verified && receipt.mutation_attempted);
    assert_eq!(receipt.commit_oid, Some("c".repeat(40)));
    assert_eq!(receipt.remote_verified_paths, ["config.json"]);
    let requests = server.finish();
    assert_eq!(requests.len(), 3);
    for request in &requests {
        assert!(
            String::from_utf8_lossy(request)
                .to_ascii_lowercase()
                .contains("authorization: bearer fixture-secret\r\n")
        );
    }
    let commit = String::from_utf8_lossy(&requests[1]);
    assert!(commit.starts_with("POST /api/models/fixture/model/commit/main "));
    assert!(commit.contains(&format!("\"parentCommit\":\"{}\"", "a".repeat(40))));
    assert!(commit.contains("\"content\":\"eyJ2IjoxfQ==\""));
    assert!(String::from_utf8_lossy(&requests[2]).starts_with(&format!(
        "GET /fixture/model/resolve/{}/config.json ",
        "c".repeat(40)
    )));
    assert!(
        !serde_json::to_string(&receipt)
            .unwrap()
            .contains("fixture-secret")
    );
    root.close().unwrap();
}
#[test]
fn regular_publication_refuses_lfs_and_gguf_before_commit() {
    let root = tempfile::tempdir().unwrap();
    let (plan, files) = fixture_file(root.path());
    let server = Server::start(vec![classification("lfs")]);
    let receipt = runtime().block_on(publisher(&server).publish_until(
        &plan,
        files,
        Instant::now() + Duration::from_secs(10),
        futures::future::pending(),
    ));
    assert!(!receipt.completed && !receipt.mutation_attempted);
    assert_eq!(server.finish().len(), 1);
    let mut gguf = plan;
    gguf.paths = vec!["model.gguf".into()];
    assert!(gguf.validate().is_err());
    let (mut plan, mut files) = fixture_file(root.path());
    std::fs::write(root.path().join("local.json"), b"GGUFabc").unwrap();
    files[0].identity.sha256 = hash(b"GGUFabc");
    assert!(contract::Staged::new(&plan, files, Instant::now() + Duration::from_secs(1)).is_err());
    plan.paths.push("../outside.txt".into());
    assert!(plan.validate().is_err());
    root.close().unwrap();
}
#[test]
fn regular_publication_preserves_commit_on_failed_remote_or_final_source_identity() {
    for source_drift in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let (plan, files) = fixture_file(root.path());
        let mut committed = commit();
        if source_drift {
            committed.changed_file = Some(root.path().join("local.json"));
        }
        let remote = if source_drift {
            b"{\"v\":1}".to_vec()
        } else {
            b"{\"v\":2}".to_vec()
        };
        let server = Server::start(vec![
            classification("regular"),
            committed,
            Reply {
                status: 200,
                body: remote,
                hold: false,
                changed_file: None,
            },
        ]);
        let receipt = runtime().block_on(publisher(&server).publish_until(
            &plan,
            files,
            Instant::now() + Duration::from_secs(10),
            futures::future::pending(),
        ));
        assert!(
            !receipt.completed && receipt.mutation_attempted && !receipt.source_custody_verified
        );
        assert_eq!(receipt.commit_oid, Some("c".repeat(40)));
        assert_eq!(
            receipt.remote_verified_paths.len(),
            usize::from(source_drift)
        );
        assert_eq!(server.finish().len(), 3);
        root.close().unwrap();
    }
}
#[test]
fn regular_publication_bounds_commit_receipt_and_never_retries_uncertain_mutation() {
    for body in [b"{}".to_vec(), vec![b' '; 65537]] {
        let root = tempfile::tempdir().unwrap();
        let (plan, files) = fixture_file(root.path());
        let server = Server::start(vec![
            classification("regular"),
            Reply {
                status: 200,
                body,
                hold: false,
                changed_file: None,
            },
        ]);
        let receipt = runtime().block_on(publisher(&server).publish_until(
            &plan,
            files,
            Instant::now() + Duration::from_secs(10),
            futures::future::pending(),
        ));
        assert!(!receipt.completed && receipt.mutation_attempted && receipt.commit_oid.is_none());
        assert_eq!(server.finish().len(), 2);
        root.close().unwrap();
    }
}
#[test]
fn regular_publication_cancel_after_actual_commit_request_retains_uncertain_outcome() {
    let root = tempfile::tempdir().unwrap();
    let (plan, files) = fixture_file(root.path());
    let server = Server::start(vec![
        classification("regular"),
        Reply {
            status: 200,
            body: Vec::new(),
            hold: true,
            changed_file: None,
        },
    ]);
    let captured = server.requests.clone();
    let (sender, receiver) = futures::channel::oneshot::channel();
    let watcher = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(8);
        while captured.lock().unwrap().len() < 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        let observed = captured.lock().unwrap().len() == 2;
        let _ = sender.send(());
        observed
    });
    let receipt = runtime().block_on(publisher(&server).publish_until(
        &plan,
        files,
        Instant::now() + Duration::from_secs(10),
        async move {
            let _ = receiver.await;
        },
    ));
    let observed = watcher.join().unwrap();
    let requests = server.finish();
    assert!(observed);
    assert_eq!(requests.len(), 2);
    assert!(!receipt.completed && receipt.mutation_attempted && receipt.commit_oid.is_none());
    assert!(receipt.error.unwrap().contains("cancelled"));
    root.close().unwrap();
}
#[test]
fn regular_publication_pre_cancel_deadline_and_secret_debug_have_no_mutation() {
    let root = tempfile::tempdir().unwrap();
    let (plan, files) = fixture_file(root.path());
    let publisher = Publisher::new(Secret::new("fixture-secret".into()).unwrap()).unwrap();
    assert_eq!(format!("{:?}", publisher.token), "Secret([redacted])");
    let receipt = runtime().block_on(publisher.publish_until(
        &plan,
        files,
        Instant::now() + Duration::from_secs(1),
        futures::future::ready(()),
    ));
    assert!(!receipt.completed && !receipt.mutation_attempted);
    let (plan, files) = fixture_file(root.path());
    let receipt = runtime().block_on(publisher.publish_until(
        &plan,
        files,
        Instant::now(),
        futures::future::pending(),
    ));
    assert!(!receipt.completed && !receipt.mutation_attempted);
    root.close().unwrap();
}

#[test]
fn regular_publication_terminal_boundary_refuses_late_deadline_and_cancellation() {
    assert!(terminal(Instant::now(), false).is_err());
    assert!(terminal(Instant::now() + Duration::from_secs(1), true).is_err());
    assert!(terminal(Instant::now() + Duration::from_secs(1), false).is_ok());
}
