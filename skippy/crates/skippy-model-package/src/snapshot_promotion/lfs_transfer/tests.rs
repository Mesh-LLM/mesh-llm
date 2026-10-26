use super::*;
use sha2::{Digest, Sha256};
use std::time::Duration;
#[path = "fixture.rs"]
mod fixture;
use fixture::{Reply, Server};
const TOKEN: &str = "fixture-hf-secret";
fn reply(value: serde_json::Value) -> Reply {
    Reply {
        status: 200,
        body: serde_json::to_vec(&value).unwrap(),
        hold: false,
        changed_file: None,
        headers: Vec::new(),
    }
}
fn empty() -> Reply {
    Reply {
        status: 200,
        body: Vec::new(),
        hold: false,
        changed_file: None,
        headers: Vec::new(),
    }
}
fn runtime() -> tokio::runtime::Runtime {
    let _ = skippy_model_hf::configure_hf_tls_provider();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}
fn object(root: &std::path::Path) -> Object {
    let bytes = b"GGUFfixture-object";
    let p = root.join("object.gguf");
    std::fs::write(&p, bytes).unwrap();
    Object {
        file: std::fs::File::open(p).unwrap(),
        oid: Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        size: bytes.len() as u64,
    }
}
fn client(server: &Server) -> Client {
    Client {
        http: reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .pool_max_idle_per_host(0)
            .build()
            .unwrap(),
        origin: reqwest::Url::parse(&server.endpoint).unwrap(),
        token: TOKEN.into(),
    }
}
fn batch(object: &Object, actions: serde_json::Value) -> Reply {
    reply(
        serde_json::json!({"transfer":"basic","objects":[{"oid":object.oid,"size":object.size,"actions":actions}]}),
    )
}
fn body(request: &[u8]) -> &[u8] {
    let at = request.windows(4).position(|b| b == b"\r\n\r\n").unwrap();
    &request[at + 4..]
}
#[test]
fn lfs_basic_streams_actual_bytes_and_verifies_without_forwarding_hf_token() {
    let root = tempfile::tempdir().unwrap();
    let object = object(root.path());
    let oid = object.oid.clone();
    let size = object.size;
    let server = Server::start(|url| {
        vec![
            batch(
                &object,
                serde_json::json!({"upload":{"href":format!("{url}/signed-upload"),"header":{"X-Upload-Key":"scoped"}},"verify":{"href":format!("{url}/verify")}}),
            ),
            empty(),
            empty(),
        ]
    });
    let receipt = runtime().block_on(client(&server).upload_until(
        "fixture/model",
        object,
        Instant::now() + Duration::from_secs(10),
        futures::future::pending(),
    ));
    let requests = server.finish();
    assert!(
        receipt.completed
            && receipt.source_custody_verified
            && receipt.object_present
            && receipt.mutation_attempted
    );
    assert_eq!(receipt.uploaded_parts, 1);
    assert_eq!(requests.len(), 3);
    assert!(
        String::from_utf8_lossy(&requests[0])
            .starts_with("POST /fixture/model.git/info/lfs/objects/batch ")
    );
    let value: serde_json::Value = serde_json::from_slice(body(&requests[0])).unwrap();
    assert_eq!(value["objects"][0]["oid"], oid);
    assert_eq!(value["objects"][0]["size"], size);
    assert_eq!(
        value["transfers"],
        serde_json::json!(["basic", "multipart"])
    );
    assert_eq!(body(&requests[1]), b"GGUFfixture-object");
    assert!(
        String::from_utf8_lossy(&requests[1])
            .to_ascii_lowercase()
            .contains("x-upload-key: scoped")
    );
    assert!(!String::from_utf8_lossy(&requests[1]).contains(TOKEN));
    assert!(String::from_utf8_lossy(&requests[2]).contains(TOKEN));
    assert!(!serde_json::to_string(&receipt).unwrap().contains(TOKEN));
    root.close().unwrap();
}
#[test]
fn lfs_multipart_streams_ordered_ranges_and_requires_etags_before_completion() {
    let root = tempfile::tempdir().unwrap();
    let object = object(root.path());
    let server = Server::start(|url| {
        let mut first = empty();
        first.headers.push(("ETag".into(), "part-one".into()));
        let mut second = empty();
        second.headers.push(("ETag".into(), "part-two".into()));
        vec![
            batch(
                &object,
                serde_json::json!({"upload":{"href":format!("{url}/complete"),"header":{"chunk_size":"10","1":format!("{url}/one"),"2":format!("{url}/two")}}}),
            ),
            first,
            second,
            empty(),
        ]
    });
    let receipt = runtime().block_on(client(&server).upload_until(
        "fixture/model",
        object,
        Instant::now() + Duration::from_secs(10),
        futures::future::pending(),
    ));
    let requests = server.finish();
    assert!(receipt.completed);
    assert_eq!(receipt.uploaded_parts, 2);
    assert_eq!(requests.len(), 4);
    assert_eq!(body(&requests[1]), &b"GGUFfixture-object"[..10]);
    assert_eq!(body(&requests[2]), &b"GGUFfixture-object"[10..]);
    let completed: serde_json::Value = serde_json::from_slice(body(&requests[3])).unwrap();
    assert_eq!(
        completed["parts"],
        serde_json::json!([{"partNumber":1,"etag":"part-one"},{"partNumber":2,"etag":"part-two"}])
    );
    assert!(
        requests[1..]
            .iter()
            .all(|r| !String::from_utf8_lossy(r).contains(TOKEN))
    );
    root.close().unwrap();
}
#[test]
fn lfs_missing_etag_preserves_attempted_mutation_without_completion_request() {
    let root = tempfile::tempdir().unwrap();
    let object = object(root.path());
    let server = Server::start(|url| {
        vec![
            batch(
                &object,
                serde_json::json!({"upload":{"href":format!("{url}/complete"),"header":{"chunk_size":"10","1":format!("{url}/one"),"2":format!("{url}/two")}}}),
            ),
            empty(),
        ]
    });
    let receipt = runtime().block_on(client(&server).upload_until(
        "fixture/model",
        object,
        Instant::now() + Duration::from_secs(10),
        futures::future::pending(),
    ));
    let requests = server.finish();
    assert_eq!(requests.len(), 2);
    assert!(receipt.mutation_attempted && !receipt.completed && !receipt.object_present);
    assert_eq!(receipt.uploaded_parts, 0);
    assert!(receipt.error.unwrap().contains("ETag"));
    root.close().unwrap();
}
#[test]
fn lfs_malformed_batch_refuses_before_mutation_and_final_source_drift_is_retained() {
    for drift in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let object = object(root.path());
        let server = Server::start(|url| {
            if drift {
                let mut done = empty();
                done.changed_file = Some(root.path().join("object.gguf"));
                vec![
                    batch(
                        &object,
                        serde_json::json!({"upload":{"href":format!("{url}/upload")}}),
                    ),
                    done,
                ]
            } else {
                vec![reply(serde_json::json!({"transfer":"xet","objects":[]}))]
            }
        });
        let receipt = runtime().block_on(client(&server).upload_until(
            "fixture/model",
            object,
            Instant::now() + Duration::from_secs(10),
            futures::future::pending(),
        ));
        let requests = server.finish();
        assert!(!receipt.completed && !receipt.source_custody_verified);
        assert_eq!(receipt.mutation_attempted, drift);
        assert_eq!(receipt.object_present, drift);
        assert_eq!(requests.len(), if drift { 2 } else { 1 });
        root.close().unwrap();
    }
}
#[test]
fn lfs_storage_and_header_policy_refuses_credential_redirect_destinations() {
    let client = Client::new(TOKEN.into()).unwrap();
    for url in [
        "http://huggingface.co/object",
        "https://evil.invalid/object",
        "https://user:secret@huggingface.co/object",
        "https://huggingface.co/object#secret",
        "https://huggingface.co:9443/object",
    ] {
        assert!(client.address(url).is_err());
    }
    assert!(
        client
            .address("https://bucket.s3.us-east-1.amazonaws.com/object?signed=secret")
            .is_ok()
    );
    assert!(
        client
            .headers(&serde_json::json!({"header":{"Host":"evil.invalid"}}))
            .is_err()
    );
    assert!(
        client
            .headers(&serde_json::json!({"header":{"X-Key":"bad\nsecret"}}))
            .is_err()
    );
}
#[test]
fn lfs_cancel_after_observed_upload_keeps_uncertain_object_receipt() {
    let root = tempfile::tempdir().unwrap();
    let object = object(root.path());
    let server = Server::start(|url| {
        let mut hold = empty();
        hold.hold = true;
        vec![
            batch(
                &object,
                serde_json::json!({"upload":{"href":format!("{url}/upload")}}),
            ),
            hold,
        ]
    });
    let observed = server.requests.clone();
    let (send, recv) = futures::channel::oneshot::channel();
    let watcher = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(8);
        while observed.lock().unwrap().len() < 2 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(2));
        }
        let yes = observed.lock().unwrap().len() == 2;
        let _ = send.send(());
        yes
    });
    let receipt = runtime().block_on(client(&server).upload_until(
        "fixture/model",
        object,
        Instant::now() + Duration::from_secs(10),
        async move {
            let _ = recv.await;
        },
    ));
    let yes = watcher.join().unwrap();
    let requests = server.finish();
    assert!(yes);
    assert_eq!(requests.len(), 2);
    assert!(receipt.mutation_attempted && !receipt.completed && !receipt.object_present);
    assert!(receipt.error.unwrap().contains("cancelled"));
    root.close().unwrap();
}

#[test]
fn lfs_precancel_and_expired_budget_refuse_before_any_object_mutation() {
    let root = tempfile::tempdir().unwrap();
    let client = Client::new(TOKEN.into()).unwrap();
    for cancelled in [false, true] {
        let receipt = runtime().block_on(client.upload_until(
            "fixture/model",
            object(root.path()),
            Instant::now(),
            async move {
                if !cancelled {
                    futures::future::pending::<()>().await;
                }
            },
        ));
        assert!(!receipt.completed && !receipt.mutation_attempted && !receipt.object_present);
    }
    root.close().unwrap();
}

#[test]
fn lfs_sparse_multipart_roster_refuses_before_any_put_with_typed_receipt() {
    let root = tempfile::tempdir().unwrap();
    let object = object(root.path());
    let server = Server::start(|url| {
        vec![batch(
            &object,
            serde_json::json!({"upload":{"href":format!("{url}/complete"),"header":{"chunk_size":"10","1":format!("{url}/one"),"3":format!("{url}/three")}}}),
        )]
    });
    let receipt = runtime().block_on(client(&server).upload_until(
        "fixture/model",
        object,
        Instant::now() + Duration::from_secs(10),
        futures::future::pending(),
    ));
    let requests = server.finish();
    assert_eq!(requests.len(), 1);
    assert!(!receipt.completed && !receipt.mutation_attempted && !receipt.object_present);
    assert_eq!(receipt.uploaded_parts, 0);
    assert!(receipt.error.unwrap().contains("numbered part URL absent"));
    root.close().unwrap();
}

#[test]
fn immutable_projector_sink_retains_exact_bytes_size_and_optional_digest() {
    let temp = tempfile::tempdir().unwrap();
    let bytes = b"GGUFfile";
    let sha: String = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    for expected in [None, Some(sha.as_str())] {
        let server = Server::start(|_| {
            let mut r = empty();
            r.body = bytes.to_vec();
            vec![r]
        });
        let mut sink = tempfile::NamedTempFile::new_in(temp.path()).unwrap();
        let identity = runtime()
            .block_on(client(&server).acquire_repository_until(
                RepositoryRead {
                    repo: "fixture/model",
                    dataset: false,
                    commit: &"b".repeat(40),
                    path: "nested/mm proj.gguf",
                    size: 8,
                    expected_sha256: expected,
                },
                sink.as_file_mut(),
                Instant::now() + Duration::from_secs(5),
            ))
            .unwrap();
        assert_eq!(identity.sha256, sha);
        assert_eq!(identity.byte_size, 8);
        assert_eq!(std::fs::read(sink.path()).unwrap(), bytes);
        let requests = server.finish();
        let header = String::from_utf8_lossy(&requests[0]);
        assert!(header.starts_with(&format!(
            "GET /fixture/model/resolve/{}/nested/mm%20proj.gguf ",
            "b".repeat(40)
        )));
        assert!(
            header
                .to_ascii_lowercase()
                .contains(&format!("authorization: bearer {TOKEN}"))
        );
    }
    temp.close().unwrap();
}
#[test]
fn immutable_projector_sink_refuses_wrong_size_digest_and_owned_held_deadline() {
    let temp = tempfile::tempdir().unwrap();
    for (size, expected) in [(7, None), (9, None), (8, Some("a".repeat(64)))] {
        let server = Server::start(|_| {
            let mut r = empty();
            r.body = b"GGUFfile".to_vec();
            vec![r]
        });
        let mut sink = tempfile::NamedTempFile::new_in(temp.path()).unwrap();
        assert!(
            runtime()
                .block_on(client(&server).acquire_repository_until(
                    RepositoryRead {
                        repo: "fixture/model",
                        dataset: false,
                        commit: &"b".repeat(40),
                        path: "mm.gguf",
                        size,
                        expected_sha256: expected.as_deref()
                    },
                    sink.as_file_mut(),
                    Instant::now() + Duration::from_secs(5)
                ))
                .is_err()
        );
        assert_eq!(server.finish().len(), 1);
    }
    let server = Server::start(|_| {
        let mut r = empty();
        r.hold = true;
        vec![r]
    });
    let mut sink = tempfile::NamedTempFile::new_in(temp.path()).unwrap();
    let error = runtime()
        .block_on(client(&server).acquire_repository_until(
            RepositoryRead {
                repo: "fixture/model",
                dataset: false,
                commit: &"b".repeat(40),
                path: "mm.gguf",
                size: 8,
                expected_sha256: None,
            },
            sink.as_file_mut(),
            Instant::now() + Duration::from_millis(200),
        ))
        .err()
        .unwrap();
    assert!(error.to_string().contains("deadline"));
    assert_eq!(server.finish().len(), 1);
    assert_eq!(sink.as_file().metadata().unwrap().len(), 0);
    drop(sink);
    temp.close().unwrap();
}
#[test]
fn immutable_projector_redirect_refuses_untrusted_origin_without_private_token() {
    let temp = tempfile::tempdir().unwrap();
    let cdn = Server::start(|_| {
        let mut r = empty();
        r.body = b"GGUFfile".to_vec();
        vec![r]
    });
    let server = Server::start(|_| {
        let mut r = empty();
        r.status = 302;
        r.headers
            .push(("Location".into(), format!("{}/signed", cdn.endpoint)));
        vec![r]
    });
    let mut sink = tempfile::NamedTempFile::new_in(temp.path()).unwrap();
    runtime()
        .block_on(client(&server).acquire_repository_until(
            RepositoryRead {
                repo: "fixture/model",
                dataset: false,
                commit: &"b".repeat(40),
                path: "mm.gguf",
                size: 8,
                expected_sha256: None,
            },
            sink.as_file_mut(),
            Instant::now() + Duration::from_secs(5),
        ))
        .unwrap_err();
    assert!(
        String::from_utf8_lossy(&server.finish()[0])
            .to_ascii_lowercase()
            .contains("authorization:")
    );
    assert!(cdn.finish().is_empty());
    drop(sink);
    temp.close().unwrap();
}

#[test]
fn immutable_projector_cancel_drops_owned_held_future_without_accepted_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(|_| {
        let mut r = empty();
        r.hold = true;
        vec![r]
    });
    let captured = server.requests.clone();
    let mut sink = tempfile::NamedTempFile::new_in(temp.path()).unwrap();
    let transport = client(&server);
    let commit = "b".repeat(40);
    runtime().block_on(async {
        tokio::select! {
            result = transport.acquire_repository_until(RepositoryRead { repo: "fixture/model", dataset: false, commit: &commit, path: "mm.gguf", size: 8, expected_sha256: None }, sink.as_file_mut(), Instant::now() + Duration::from_secs(5)) => panic!("held transfer unexpectedly settled: {}", result.is_ok()),
            () = async {
                let until = Instant::now() + Duration::from_secs(2);
                while captured.lock().unwrap().is_empty() {
                    assert!(Instant::now() < until, "owned request checkpoint missing");
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            } => (),
        }
    });
    assert_eq!(server.finish().len(), 1);
    assert_eq!(sink.as_file().metadata().unwrap().len(), 0);
    drop(sink);
    temp.close().unwrap();
}
