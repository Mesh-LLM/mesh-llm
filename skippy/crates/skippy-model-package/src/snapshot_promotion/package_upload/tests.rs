use super::*;
use serde_json::json;
use std::{
    fs,
    time::{Duration, Instant},
};
#[path = "fixture.rs"]
mod fixture;
use fixture::{Reply, Server};
fn publisher(server: &Server) -> Publisher {
    Publisher {
        client: lfs_transfer::Client {
            http: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap(),
            origin: reqwest::Url::parse(&server.endpoint).unwrap(),
            token: "fixture-token".into(),
        },
    }
}
fn reply(body: serde_json::Value) -> Reply {
    Reply {
        status: 200,
        body: serde_json::to_vec(&body).unwrap(),
        hold: false,
        headers: Vec::new(),
    }
}
fn artifact(root: &std::path::Path, unlink: bool) -> Artifact {
    let path = root.join("item.json");
    fs::write(&path, b"{}").unwrap();
    use sha2::{Digest as _, Sha256};
    Artifact {
        file: fs::File::open(&path).unwrap(),
        identity: ArtifactIdentity {
            byte_size: 2,
            sha256: Sha256::digest(b"{}")
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        },
        unlink_path: unlink.then_some(path),
    }
}
fn plan(kind: RepositoryKind, pr: bool) -> Plan {
    Plan {
        repo: "fixture/repo".into(),
        kind,
        revision: if pr { "main" } else { "automation/staging" }.into(),
        path: "items/item.json".into(),
        create_pr: pr,
        maximum_attempts: 1,
        expected_parent: None,
    }
}
#[test]
fn package_upload_actual_regular_commit_preserves_staging_dataset_pr_and_unlinks_only_verified_source()
 {
    for (kind, pr, unlink) in [
        (RepositoryKind::Model, false, cfg!(unix)),
        (RepositoryKind::Dataset, true, false),
    ] {
        let root = tempfile::tempdir().unwrap();
        let canonical = root.path().canonicalize().unwrap();
        let server = Server::start(vec![
            reply(json!({"sha":"a".repeat(40)})),
            reply(
                json!({"files":[{"path":"items/item.json","uploadMode":"regular","shouldIgnore":false}]}),
            ),
            reply(json!({"commitOid":"b".repeat(40)})),
            Reply {
                status: 200,
                body: b"{}".to_vec(),
                hold: false,
                headers: Vec::new(),
            },
        ]);
        let receipt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(publisher(&server).upload_until(
                &plan(kind, pr),
                artifact(&canonical, unlink),
                Instant::now() + Duration::from_secs(5),
                std::future::pending(),
                &mut |_| Ok(()),
            ));
        assert!(receipt.completed && receipt.error.is_none() && receipt.source_custody_verified);
        assert_eq!(receipt.unlinked, unlink);
        assert_eq!(canonical.join("item.json").exists(), !unlink);
        assert_eq!(receipt.attempts.len(), 1);
        assert!(receipt.attempts[0].remote_verified);
        assert_eq!(
            receipt.attempts[0].commit_oid.as_deref(),
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        );
        let requests = server.finish();
        assert_eq!(requests.len(), 4);
        let commit = String::from_utf8_lossy(&requests[2]);
        assert!(commit.contains("\"parentCommit\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\""));
        assert!(commit.contains("\"content\":\"e30=\""));
        if pr {
            assert!(commit.starts_with("POST /api/datasets/fixture/repo/commit/main?create_pr=1 "));
            assert!(String::from_utf8_lossy(&requests[3]).starts_with("GET /datasets/fixture/repo/resolve/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/items/item.json "));
        } else {
            assert!(
                commit.starts_with("POST /api/models/fixture/repo/commit/automation%2Fstaging ")
            );
        }
        root.close().unwrap();
    }
}
#[test]
fn package_upload_refused_remote_bytes_and_held_commit_keep_file_and_known_attempt_evidence() {
    for hold in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let canonical = root.path().canonicalize().unwrap();
        let mut replies = vec![
            reply(json!({"sha":"a".repeat(40)})),
            reply(json!({"files":[{"path":"items/item.json","uploadMode":"regular"}]})),
            Reply {
                status: 200,
                body: serde_json::to_vec(&json!({"commitOid":"b".repeat(40)})).unwrap(),
                hold,
                headers: Vec::new(),
            },
        ];
        if !hold {
            replies.push(Reply {
                status: 200,
                body: b"wrong".to_vec(),
                hold: false,
                headers: Vec::new(),
            });
        }
        let server = Server::start(replies);
        let duration = if hold {
            Duration::from_millis(500)
        } else {
            Duration::from_secs(5)
        };
        let receipt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(publisher(&server).upload_until(
                &plan(RepositoryKind::Model, false),
                artifact(&canonical, cfg!(unix)),
                Instant::now() + duration,
                std::future::pending(),
                &mut |_| Ok(()),
            ));
        assert!(!receipt.completed && receipt.error.is_some() && !receipt.unlinked);
        assert!(canonical.join("item.json").is_file());
        assert!(receipt.attempts[0].commit_attempted);
        assert!(!receipt.attempts[0].remote_verified);
        let requests = server.finish();
        assert_eq!(requests.len(), if hold { 3 } else { 4 });
        assert!(String::from_utf8_lossy(&requests[2]).starts_with("POST /api/models/"));
        root.close().unwrap();
    }
}
#[test]
fn package_upload_policy_limits_attempts_and_disallows_pr_on_staging_or_model() {
    let mut p = plan(RepositoryKind::Model, false);
    assert!(p.validate().is_ok());
    for attempts in [0, 9] {
        p.maximum_attempts = attempts;
        assert!(p.validate().is_err());
    }
    p.maximum_attempts = 8;
    p.create_pr = true;
    assert!(p.validate().is_err());
    p.kind = RepositoryKind::Dataset;
    assert!(p.validate().is_err());
    p.revision = "main".into();
    assert!(p.validate().is_ok());
    p.path = "../foreign".into();
    assert!(p.validate().is_err());
}
#[test]
fn layer_catalog_native_projection_preserves_dict_list_variants_and_replaces_only_target_package() {
    let input = catalog::CatalogInput {
        source_repo: "source/model",
        source_revision: &"a".repeat(40),
        source_file: "Q4/model-Q4-00001-of-00002.gguf",
        target_repo: "fixture/repo",
        model_id: "model",
        layer_count: 4,
    };
    for list in [false, true] {
        let row = json!({"curated":{"name":"model-Q4","keep":"unchanged"},"packages":[{"repo":"other/repo","keep":true},{"repo":"fixture/repo","old":true}],"opaque":{"keep":1}});
        let original = json!({"schema_version":1,"source_repo":"source/model","other":true,"variants":if list{json!([row])}else{json!({"model-Q4":row})}});
        let (path, value) = catalog::project(Some(original), &input).unwrap();
        assert_eq!(path, "entries/source/model.json");
        assert_eq!(value["other"], true);
        let row = if list {
            &value["variants"][0]
        } else {
            &value["variants"]["model-Q4"]
        };
        assert_eq!(row["curated"]["keep"], "unchanged");
        assert_eq!(row["opaque"]["keep"], 1);
        assert_eq!(row["packages"].as_array().unwrap().len(), 2);
        assert_eq!(row["packages"][0]["repo"], "other/repo");
        assert_eq!(row["packages"][1]["source_revision"], "a".repeat(40));
    }
    assert!(
        catalog::project(
            Some(json!({"source_repo":"source/model","variants":false})),
            &input
        )
        .is_err()
    );
    let (_, fresh) = catalog::project(None, &input).unwrap();
    assert_eq!(fresh["variants"]["model-Q4"]["curated"]["size"], "4 layers");
}
#[test]
fn layer_repository_create_exist_ok_still_requires_actual_authenticated_parent() {
    for conflict in [false, true] {
        let server = Server::start(vec![
            Reply {
                status: if conflict { 409 } else { 200 },
                body: b"{}".to_vec(),
                hold: false,
                headers: Vec::new(),
            },
            reply(json!({"sha":"a".repeat(40)})),
        ]);
        let receipt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(publisher(&server).ensure_model_repo_until(
                "fixture/repo",
                true,
                Instant::now() + Duration::from_secs(5),
                std::future::pending(),
            ));
        assert!(receipt.completed && receipt.error.is_none());
        assert!(receipt.mutation_attempted);
        assert_eq!(receipt.existing_conflict, conflict);
        assert_eq!(
            receipt.observed_parent.as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
        let requests = server.finish();
        assert_eq!(requests.len(), 2);
        let create = String::from_utf8_lossy(&requests[0]);
        assert!(create.starts_with("POST /api/repos/create "));
        assert!(create.contains("\"organization\":\"fixture\""));
        assert!(create.contains("\"type\":\"model\""));
    }
}
#[test]
fn package_upload_actual_lfs_artifact_reuses_batch_put_and_commits_one_verified_path() {
    let root = tempfile::tempdir().unwrap();
    let canonical = root.path().canonicalize().unwrap();
    let artifact = artifact(&canonical, cfg!(unix));
    let identity = artifact.identity.clone();
    let mut plan = plan(RepositoryKind::Model, false);
    plan.path = "layers/layer.gguf".into();
    let server = Server::start_with(|origin| {
        vec![
            reply(json!({"sha":"a".repeat(40)})),
            reply(json!({"files":[{"path":"layers/layer.gguf","uploadMode":"lfs"}]})),
            reply(
                json!({"transfer":"basic","objects":[{"oid":identity.sha256,"size":2,"actions":{"upload":{"href":format!("{origin}/object"),"header":{}}}}]}),
            ),
            Reply {
                status: 200,
                body: vec![],
                hold: false,
                headers: Vec::new(),
            },
            reply(json!({"commitOid":"b".repeat(40)})),
            Reply {
                status: 200,
                body: b"{}".to_vec(),
                hold: false,
                headers: Vec::new(),
            },
        ]
    });
    let receipt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(publisher(&server).upload_until(
            &plan,
            artifact,
            Instant::now() + Duration::from_secs(5),
            std::future::pending(),
            &mut |_| Ok(()),
        ));
    assert!(receipt.completed && receipt.source_custody_verified);
    assert_eq!(receipt.unlinked, cfg!(unix));
    let object = receipt.attempts[0].object.as_ref().unwrap();
    assert!(
        object.completed
            && object.object_present
            && object.source_custody_verified
            && object.error.is_none()
    );
    let requests = server.finish();
    assert_eq!(requests.len(), 6);
    assert!(
        String::from_utf8_lossy(&requests[2])
            .starts_with("POST /fixture/repo.git/info/lfs/objects/batch ")
    );
    assert!(String::from_utf8_lossy(&requests[3]).starts_with("PUT /object "));
    assert!(requests[3].ends_with(b"{}"));
    let commit = String::from_utf8_lossy(&requests[4]);
    assert!(commit.contains("\"key\":\"lfsFile\""));
    assert!(commit.contains(&identity.sha256));
    assert!(commit.contains("layers/layer.gguf"));
    assert_eq!(canonical.join("item.json").exists(), !cfg!(unix));
    root.close().unwrap();
}
#[test]
fn package_upload_retry_wait_is_cancelled_after_actual_failed_attempt_without_unlink_or_second_commit()
 {
    let root = tempfile::tempdir().unwrap();
    let canonical = root.path().canonicalize().unwrap();
    let server = Server::start(vec![
        reply(json!({"sha":"a".repeat(40)})),
        Reply {
            status: 503,
            body: b"refused".to_vec(),
            hold: false,
            headers: Vec::new(),
        },
    ]);
    let mut plan = plan(RepositoryKind::Model, false);
    plan.maximum_attempts = 8;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let mut sender = Some(sender);
    let mut observer = |receipt: &Receipt| -> anyhow::Result<()> {
        if receipt.attempts.last().is_some_and(|a| a.error.is_some()) {
            sender.take().unwrap().send(()).unwrap();
        }
        Ok(())
    };
    let receipt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(publisher(&server).upload_until(
            &plan,
            artifact(&canonical, cfg!(unix)),
            Instant::now() + Duration::from_secs(5),
            async {
                receiver.await.unwrap();
            },
            &mut observer,
        ));
    assert!(!receipt.completed && receipt.error.is_some() && !receipt.unlinked);
    assert_eq!(receipt.attempts.len(), 1);
    assert!(receipt.attempts[0].error.is_some());
    assert!(!receipt.attempts[0].commit_attempted);
    assert!(canonical.join("item.json").exists());
    assert_eq!(server.finish().len(), 2);
    root.close().unwrap();
}

#[test]
fn package_upload_uncertain_main_attempt_reconciles_actual_immutable_bytes_without_second_commit() {
    let root = tempfile::tempdir().unwrap();
    let canonical = root.path().canonicalize().unwrap();
    let server = Server::start(vec![
        reply(json!({"sha":"a".repeat(40)})),
        reply(json!({"files":[{"path":"items/item.json","uploadMode":"regular"}]})),
        reply(json!({"commitOid":"malformed"})),
        reply(json!({"sha":"b".repeat(40)})),
        Reply {
            status: 200,
            body: b"{}".to_vec(),
            hold: false,
            headers: Vec::new(),
        },
    ]);
    let publisher = publisher(&server);
    let mut artifact = artifact(&canonical, false);
    let mut plan = plan(RepositoryKind::Model, false);
    plan.revision = "main".into();
    plan.expected_parent = Some("a".repeat(40));
    let mut first = Attempt {
        ordinal: 1,
        parent_commit: None,
        object: None,
        commit_attempted: false,
        commit_oid: None,
        remote_verified: false,
        error: None,
    };
    let mut second = Attempt {
        ordinal: 2,
        parent_commit: None,
        object: None,
        commit_attempted: false,
        commit_oid: None,
        remote_verified: false,
        error: None,
    };
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let until = Instant::now() + Duration::from_secs(5);
            let mut cancellation = Box::pin(std::future::pending::<()>());
            assert!(
                publisher
                    .attempt(
                        &plan,
                        &mut artifact,
                        until,
                        cancellation.as_mut(),
                        &mut first
                    )
                    .await
                    .is_err()
            );
            assert!(first.commit_attempted && first.commit_oid.is_none());
            publisher
                .attempt(
                    &plan,
                    &mut artifact,
                    until,
                    cancellation.as_mut(),
                    &mut second,
                )
                .await
                .unwrap();
        });
    assert!(second.remote_verified && !second.commit_attempted);
    assert_eq!(
        second.commit_oid.as_deref(),
        Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    );
    assert!(canonical.join("item.json").is_file());
    let requests = server.finish();
    assert_eq!(requests.len(), 5);
    assert_eq!(
        requests
            .iter()
            .filter(
                |r| String::from_utf8_lossy(r).starts_with("POST /api/models/fixture/repo/commit/")
            )
            .count(),
        1
    );
    assert!(String::from_utf8_lossy(&requests[4]).starts_with(
        "GET /fixture/repo/resolve/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/items/item.json "
    ));
    root.close().unwrap();
}
#[test]
fn package_upload_uncertain_pr_commit_is_not_repeated_despite_remaining_attempts() {
    let root = tempfile::tempdir().unwrap();
    let canonical = root.path().canonicalize().unwrap();
    let server = Server::start(vec![
        reply(json!({"sha":"a".repeat(40)})),
        reply(json!({"files":[{"path":"items/item.json","uploadMode":"regular"}]})),
        reply(json!({"commitOid":"malformed"})),
    ]);
    let mut plan = plan(RepositoryKind::Dataset, true);
    plan.maximum_attempts = 8;
    let receipt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(publisher(&server).upload_until(
            &plan,
            artifact(&canonical, false),
            Instant::now() + Duration::from_secs(5),
            std::future::pending(),
            &mut |_| Ok(()),
        ));
    assert!(!receipt.completed && receipt.error.is_some());
    assert_eq!(receipt.attempts.len(), 1);
    assert!(
        receipt.attempts[0].commit_attempted
            && receipt.attempts[0].commit_oid.is_none()
            && receipt.attempts[0].error.is_some()
    );
    assert!(!receipt.unlinked && canonical.join("item.json").is_file());
    let requests = server.finish();
    assert_eq!(requests.len(), 3);
    assert!(
        String::from_utf8_lossy(&requests[2])
            .starts_with("POST /api/datasets/fixture/repo/commit/main?create_pr=1 ")
    );
    root.close().unwrap();
}

#[test]
fn layer_catalog_actual_immutable_read_distinguishes_missing_entry_from_auth_and_malformed_failure()
{
    for mode in ["existing", "missing", "ambiguous", "auth", "malformed"] {
        let root = tempfile::tempdir().unwrap();
        let mut response =
            reply(json!({"source_repo":"source/model","variants":{},"preserved":true}));
        match mode {
            "missing" => {
                response.status = 404;
                response
                    .headers
                    .push(("X-Error-Code".into(), "EntryNotFound".into()));
            }
            "ambiguous" => response.status = 404,
            "auth" => response.status = 401,
            "malformed" => response.body = b"invalid".to_vec(),
            _ => (),
        }
        let server = Server::start(vec![reply(json!({"sha":"a".repeat(40)})), response]);
        let source_revision = "b".repeat(40);
        let input = catalog::CatalogInput {
            source_repo: "source/model",
            source_revision: &source_revision,
            source_file: "model-Q4-00001-of-00002.gguf",
            target_repo: "fixture/package",
            model_id: "model",
            layer_count: 4,
        };
        let observed = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(
                publisher(&server)
                    .prepare_catalog_until(&input, Instant::now() + Duration::from_secs(5)),
            );
        if mode == "existing" || mode == "missing" {
            let prepared = observed.unwrap();
            assert_eq!(prepared.observed_parent, "a".repeat(40));
            assert_eq!(prepared.missing_entry, mode == "missing");
            let value: serde_json::Value = serde_json::from_slice(&prepared.bytes).unwrap();
            assert_eq!(
                value["variants"]["model-Q4"]["packages"][0]["repo"],
                "fixture/package"
            );
            if mode == "existing" {
                assert_eq!(value["preserved"], true);
            }
        } else {
            assert!(observed.is_err());
        }
        let requests = server.finish();
        assert_eq!(requests.len(), 2);
        assert!(String::from_utf8_lossy(&requests[1]).starts_with("GET /datasets/meshllm/catalog/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/entries/source/model.json "));
        root.close().unwrap();
    }
}
#[test]
fn layer_catalog_expected_parent_refuses_concurrent_catalog_update_before_any_preupload_or_commit()
{
    let root = tempfile::tempdir().unwrap();
    let canonical = root.path().canonicalize().unwrap();
    let server = Server::start(vec![reply(json!({"sha":"b".repeat(40)}))]);
    let mut plan = plan(RepositoryKind::Dataset, true);
    plan.maximum_attempts = 1;
    plan.expected_parent = Some("a".repeat(40));
    let receipt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(publisher(&server).upload_until(
            &plan,
            artifact(&canonical, false),
            Instant::now() + Duration::from_secs(5),
            std::future::pending(),
            &mut |_| Ok(()),
        ));
    assert!(!receipt.completed && receipt.error.is_some());
    assert_eq!(receipt.attempts.len(), 1);
    assert!(!receipt.attempts[0].commit_attempted);
    assert_eq!(
        receipt.attempts[0].parent_commit.as_deref(),
        Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    );
    assert_eq!(server.finish().len(), 1);
    assert!(canonical.join("item.json").is_file());
    root.close().unwrap();
}

#[test]
fn layer_catalog_actual_held_initial_head_obeys_inherited_deadline() {
    let server = Server::start(vec![Reply {
        status: 200,
        body: b"{}".to_vec(),
        hold: true,
        headers: Vec::new(),
    }]);
    let pin = "b".repeat(40);
    let input = catalog::CatalogInput {
        source_repo: "source/model",
        source_revision: &pin,
        source_file: "model.gguf",
        target_repo: "fixture/package",
        model_id: "model",
        layer_count: 4,
    };
    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(
            publisher(&server)
                .prepare_catalog_until(&input, Instant::now() + Duration::from_millis(500)),
        );
    let error = result.err().unwrap().to_string();
    assert!(error.contains("deadline"));
    let requests = server.finish();
    assert_eq!(requests.len(), 1);
    assert!(
        String::from_utf8_lossy(&requests[0])
            .starts_with("GET /api/datasets/meshllm/catalog/revision/main ")
    );
}
