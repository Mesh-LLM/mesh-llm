use super::*;
use sha2::{Digest, Sha256};
use std::time::Duration;
#[path = "fixture.rs"]
mod fixture;
use fixture::{Reply, Server};
fn reply(bytes: Vec<u8>) -> Reply {
    Reply {
        status: 200,
        body: bytes,
        hold: false,
        changed_file: None,
        headers: Vec::new(),
    }
}
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn input(root: &std::path::Path) -> Input {
    let mut shards = Vec::new();
    for index in 1..=3 {
        let path = format!("composite-{index:05}-of-00003.gguf");
        let bytes = format!("GGUFfixture-shard-{index}").into_bytes();
        std::fs::write(root.join(&path), &bytes).unwrap();
        shards.push(Shard {
            path_in_repo: path.clone(),
            object: lfs_transfer::Object {
                file: std::fs::File::open(root.join(path)).unwrap(),
                oid: hash(&bytes),
                size: bytes.len() as u64,
            },
        });
    }
    let bytes = b"{\"model\":\"fixture\"}";
    std::fs::write(root.join("config.json"), bytes).unwrap();
    Input {
        repo: "fixture/composite".into(),
        parent_commit: "a".repeat(40),
        shards,
        sidecars: vec![regular_publication::LocalFile {
            path_in_repo: "config.json".into(),
            file: std::fs::File::open(root.join("config.json")).unwrap(),
            identity: ArtifactIdentity {
                byte_size: bytes.len() as u64,
                sha256: hash(bytes),
            },
        }],
    }
}
fn runtime() -> tokio::runtime::Runtime {
    let _ = skippy_model_hf::configure_hf_tls_provider();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}
fn publisher(server: &Server) -> Publisher {
    Publisher {
        client: lfs_transfer::Client {
            http: reqwest::Client::builder()
                .no_proxy()
                .pool_max_idle_per_host(0)
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap(),
            origin: reqwest::Url::parse(&server.endpoint).unwrap(),
            token: "fixture-private-token".into(),
        },
    }
}
fn replies(value: &Input, url: &str) -> Vec<Reply> {
    let mut rows = Vec::new();
    for (index, shard) in value.shards.iter().enumerate() {
        rows.push(reply(serde_json::to_vec(&serde_json::json!({"transfer":"basic","objects":[{"oid":shard.object.oid,"size":shard.object.size,"actions":{"upload":{"href":format!("{url}/signed-{index}")}}}]})).unwrap()));
        rows.push(reply(Vec::new()));
    }
    rows.push(reply(
        serde_json::to_vec(&serde_json::json!({"commitOid":"c".repeat(40)})).unwrap(),
    ));
    for index in 1..=3 {
        rows.push(reply(format!("GGUFfixture-shard-{index}").into_bytes()));
    }
    rows.push(reply(b"{\"model\":\"fixture\"}".to_vec()));
    rows
}
fn body(request: &[u8]) -> &[u8] {
    let index = request.windows(4).position(|b| b == b"\r\n\r\n").unwrap();
    &request[index + 4..]
}
#[test]
fn model_publication_orders_complete_shards_and_sidecar_in_one_parent_bound_commit() {
    let root = tempfile::tempdir().unwrap();
    let value = input(root.path());
    let expected = value
        .shards
        .iter()
        .map(|s| s.path_in_repo.clone())
        .chain(value.sidecars.iter().map(|s| s.path_in_repo.clone()))
        .collect::<Vec<_>>();
    let server = Server::start(|url| replies(&value, url));
    let receipt = runtime().block_on(publisher(&server).publish_until(
        value,
        Instant::now() + Duration::from_secs(10),
        futures::future::pending(),
    ));
    let requests = server.finish();
    assert!(receipt.completed && receipt.final_source_custody_verified);
    assert!(receipt.objects.iter().all(|r| r.completed));
    assert_eq!(receipt.ordered_paths, expected);
    assert_eq!(receipt.remote_verified_paths, expected);
    assert_eq!(receipt.commit_oid, Some("c".repeat(40)));
    assert_eq!(requests.len(), 11);
    let rows = body(&requests[6])
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
        .map(|b| serde_json::from_slice::<serde_json::Value>(b).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0]["value"]["parentCommit"], "a".repeat(40));
    for index in 0..3 {
        assert_eq!(rows[index + 1]["key"], "lfsFile");
        assert_eq!(rows[index + 1]["value"]["path"], expected[index]);
    }
    assert_eq!(rows[4]["key"], "file");
    assert_eq!(rows[4]["value"]["path"], "config.json");
    assert!(
        requests[7..]
            .iter()
            .all(|r| String::from_utf8_lossy(r).contains(&"c".repeat(40)))
    );
    assert!(
        !serde_json::to_string(&receipt)
            .unwrap()
            .contains("fixture-private-token")
    );
    root.close().unwrap();
}
#[test]
fn model_publication_refuses_missing_reordered_duplicate_shards_before_upload() {
    for mode in 0..4 {
        let root = tempfile::tempdir().unwrap();
        let mut value = input(root.path());
        match mode {
            0 => {
                value.shards.remove(1);
            }
            1 => value.shards.swap(0, 1),
            2 => value.shards[1].path_in_repo = value.shards[0].path_in_repo.clone(),
            _ => value.parent_commit = "main".into(),
        };
        assert!(admission::validate(&mut value, Instant::now() + Duration::from_secs(5)).is_err());
        root.close().unwrap();
    }
}
#[test]
fn model_publication_retains_commit_and_verified_subset_on_remote_or_final_source_drift() {
    for local in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let value = input(root.path());
        let server = Server::start(|url| {
            let mut rows = replies(&value, url);
            if local {
                rows[6].changed_file = Some(root.path().join("composite-00002-of-00003.gguf"));
            } else {
                rows[8].body = b"GGUFdifferent-byte-content".to_vec();
                rows.truncate(9);
            }
            rows
        });
        let receipt = runtime().block_on(publisher(&server).publish_until(
            value,
            Instant::now() + Duration::from_secs(10),
            futures::future::pending(),
        ));
        let requests = server.finish();
        assert!(!receipt.completed && !receipt.final_source_custody_verified);
        assert!(receipt.commit_attempted);
        assert_eq!(receipt.commit_oid, Some("c".repeat(40)));
        assert_eq!(receipt.objects.len(), 3);
        assert!(receipt.objects.iter().all(|r| r.completed));
        assert_eq!(
            receipt.remote_verified_paths.len(),
            if local { 4 } else { 1 }
        );
        assert_eq!(requests.len(), if local { 11 } else { 9 });
        root.close().unwrap();
    }
}
#[test]
fn model_publication_cancels_after_actual_immutable_request_without_erasing_uploaded_objects() {
    let root = tempfile::tempdir().unwrap();
    let value = input(root.path());
    let server = Server::start(|url| {
        let mut rows = replies(&value, url);
        rows.truncate(8);
        rows[7].hold = true;
        rows
    });
    let observed = server.requests.clone();
    let (send, recv) = futures::channel::oneshot::channel();
    let watcher = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(8);
        while observed.lock().unwrap().len() < 8 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(2));
        }
        let yes = observed.lock().unwrap().len() == 8;
        let _ = send.send(());
        yes
    });
    let receipt = runtime().block_on(publisher(&server).publish_until(
        value,
        Instant::now() + Duration::from_secs(10),
        async move {
            let _ = recv.await;
        },
    ));
    let yes = watcher.join().unwrap();
    let requests = server.finish();
    assert!(yes);
    assert_eq!(requests.len(), 8);
    assert!(!receipt.completed && !receipt.final_source_custody_verified);
    assert_eq!(receipt.objects.len(), 3);
    assert!(receipt.objects.iter().all(|r| r.completed));
    assert_eq!(receipt.commit_oid, Some("c".repeat(40)));
    assert!(receipt.remote_verified_paths.is_empty());
    assert!(receipt.error.unwrap().contains("cancelled"));
    root.close().unwrap();
}

#[test]
fn model_publication_progress_precedes_mutation_and_callback_refusal_retains_commit() {
    for at_commit in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let value = input(root.path());
        let server = Server::start(|url| replies(&value, url));
        let mut snapshots = Vec::new();
        let mut observer = |receipt: &Receipt| {
            snapshots.push(serde_json::to_value(receipt).unwrap());
            if !at_commit || receipt.commit_oid.is_some() {
                anyhow::bail!("fixture progress refusal");
            }
            Ok(())
        };
        let receipt = runtime().block_on(publisher(&server).publish_observed_until(
            value,
            Instant::now() + Duration::from_secs(10),
            futures::future::pending(),
            &mut observer,
        ));
        let requests = server.finish();
        assert!(!receipt.completed);
        assert!(
            receipt
                .error
                .as_deref()
                .unwrap()
                .contains("fixture progress refusal")
        );
        assert!(snapshots.iter().all(|s| s["completed"] == false));
        if at_commit {
            assert_eq!(requests.len(), 7);
            assert_eq!(receipt.commit_oid, Some("c".repeat(40)));
            assert!(receipt.commit_attempted && receipt.remote_verified_paths.is_empty());
            assert_eq!(snapshots.last().unwrap()["commit_oid"], "c".repeat(40));
        } else {
            assert!(requests.is_empty());
            assert_eq!(receipt.object_attempted_paths.len(), 1);
            assert!(!receipt.commit_attempted);
        }
        root.close().unwrap();
    }
}

#[test]
fn model_publication_held_put_deadline_retains_current_object_and_durable_observation() {
    let root = tempfile::tempdir().unwrap();
    let value = input(root.path());
    let server = Server::start(|url| {
        let mut rows = replies(&value, url);
        rows[1].hold = true;
        rows.truncate(2);
        rows
    });
    let mut durable = None;
    let mut observer = |receipt: &Receipt| {
        durable = Some(serde_json::to_value(receipt).unwrap());
        Ok(())
    };
    let receipt = runtime().block_on(publisher(&server).publish_observed_until(
        value,
        Instant::now() + Duration::from_secs(2),
        futures::future::pending(),
        &mut observer,
    ));
    let requests = server.finish();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].starts_with(b"PUT "));
    assert!(!receipt.completed && !receipt.commit_attempted);
    assert!(receipt.commit_oid.is_none());
    assert_eq!(receipt.objects.len(), 1);
    assert!(receipt.objects[0].mutation_attempted);
    assert!(!receipt.objects[0].completed);
    assert_eq!(receipt.objects[0].uploaded_parts, 0);
    assert_eq!(
        durable.unwrap()["objects"],
        serde_json::to_value(&receipt.objects).unwrap()
    );
    root.close().unwrap();
}
