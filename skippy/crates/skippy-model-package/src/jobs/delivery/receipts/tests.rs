use super::super::{PreparedCertificationDelivery, tests::fixture};
use super::*;
use crate::jobs::{
    TransportLimits,
    transport_tests::{Peer, response, runtime},
};
use crate::snapshot_promotion::regular_publication::fixture_retrieval_publisher;
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
fn prepared() -> PreparedCertificationDelivery {
    let (input, mounts, plan) = fixture();
    PreparedCertificationDelivery::prepare(&serde_json::to_vec(&input).unwrap(), &mounts, &plan)
        .unwrap()
        .with_publication_credential("inert-publication".into())
        .unwrap()
}
fn native(decl: &DeliveryDeclaration) -> Value {
    json!({"schema_version":1,"request_sha256":"a".repeat(64),"transport_input_sha256":decl.transport_input_sha256,"status":"CERTIFIED","error":null,"bootstrap":{"status":"BOOTSTRAP_COMPLETED"},"acquisition":{"certification":{"status":"PASS","source_unchanged":true}}})
}
fn locator(decl: &DeliveryDeclaration, body: &[u8]) -> Locator {
    Locator {
        schema_version: 1,
        transport_input_sha256: decl.transport_input_sha256.clone(),
        receipt_request_sha256: "a".repeat(64),
        repo: decl.evidence_repo.clone(),
        parent_commit: decl.evidence_parent_commit.clone(),
        commit_oid: "b".repeat(40),
        path_in_repo: decl.evidence_path.clone(),
        artifact_sha256: Sha256::digest(body)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        byte_size: body.len() as u64,
        delivery_complete: true,
    }
}
fn status(stage: &str) -> Vec<u8> {
    response(
        200,
        &serde_json::to_vec(&json!({"id":"native-1","status":{"stage":stage,"message":null}}))
            .unwrap(),
    )
}
fn log(value: &Locator) -> String {
    format!(
        "MESH_NATIVE_DELIVERY {}",
        serde_json::to_string(value).unwrap()
    )
}
fn logs(lines: &[String]) -> Vec<u8> {
    response(
        200,
        lines
            .iter()
            .map(|line| {
                format!(
                    "data: {}\n",
                    serde_json::to_string(&json!({"data":line,"timestamp":null})).unwrap()
                )
            })
            .collect::<String>()
            .as_bytes(),
    )
}
#[test]
fn completed_jobs_native_delivery_fetches_final_locator_and_exact_immutable_receipt() {
    let prepared = prepared();
    let body = serde_json::to_vec(&native(prepared.declaration())).unwrap();
    let locator = locator(prepared.declaration(), &body);
    let peer = Peer::many(vec![
        (status("PENDING"), false),
        (status("COMPLETED"), false),
        (logs(&[log(&locator)]), false),
        (response(200, &body), false),
    ]);
    let client = peer.client(TransportLimits::default());
    let deadline = Instant::now() + Duration::from_secs(3);
    let submitted = runtime()
        .block_on(client.submit_certification_until(
            "owner",
            prepared,
            deadline,
            std::future::pending(),
        ))
        .unwrap();
    let publisher = fixture_retrieval_publisher(&client.endpoint);
    let verified = runtime()
        .block_on(client.collect_certification_until(
            "owner",
            &submitted,
            &publisher,
            deadline,
            std::future::pending(),
            MonitorLimits::default(),
        ))
        .unwrap();
    assert_eq!(verified.native_receipt["status"], "CERTIFIED");
    assert_eq!(verified.receipt_log_lines, 1);
    assert_eq!(verified.monitor.end, MonitorEnd::Completed);
    assert!(!verified.image_observed && !verified.cost_observed);
    let requests = peer.requests.try_iter().collect::<Vec<_>>();
    assert_eq!(requests.len(), 4);
    let last = String::from_utf8_lossy(&requests[3]);
    assert!(last.starts_with(&format!(
        "GET /fixture/evidence/resolve/{}/runs/native-job.json ",
        "b".repeat(40)
    )));
}
#[test]
fn completed_jobs_state_cannot_replace_locator_identity_or_native_pass() {
    for mode in [
        "missing",
        "foreign",
        "ambiguous",
        "failed-native",
        "changed-bytes",
    ] {
        let prepared = prepared();
        let mut native = native(prepared.declaration());
        if mode == "failed-native" {
            native["status"] = json!("FAILED");
        }
        let body = serde_json::to_vec(&native).unwrap();
        let mut loc = locator(prepared.declaration(), &body);
        let mut lines = vec![log(&loc)];
        match mode {
            "missing" => lines = vec!["ordinary-log".into()],
            "foreign" => {
                loc.transport_input_sha256 = "c".repeat(64);
                lines = vec![log(&loc)];
            }
            "ambiguous" => {
                loc.commit_oid = "c".repeat(40);
                lines.push(log(&loc));
            }
            _ => (),
        }
        let mut replies = vec![
            (status("PENDING"), false),
            (status("COMPLETED"), false),
            (logs(&lines), false),
        ];
        if matches!(mode, "failed-native" | "changed-bytes") {
            replies.push((
                response(
                    200,
                    if mode == "changed-bytes" {
                        b"wrong"
                    } else {
                        &body
                    },
                ),
                false,
            ));
        }
        let peer = Peer::many(replies);
        let client = peer.client(TransportLimits::default());
        let deadline = Instant::now() + Duration::from_secs(3);
        let submitted = runtime()
            .block_on(client.submit_certification_until(
                "owner",
                prepared,
                deadline,
                std::future::pending(),
            ))
            .unwrap();
        let publisher = fixture_retrieval_publisher(&client.endpoint);
        assert!(
            runtime()
                .block_on(client.collect_certification_until(
                    "owner",
                    &submitted,
                    &publisher,
                    deadline,
                    std::future::pending(),
                    MonitorLimits::default()
                ))
                .is_err()
        );
        assert_eq!(
            peer.requests.try_iter().count(),
            if matches!(mode, "failed-native" | "changed-bytes") {
                4
            } else {
                3
            }
        );
    }
}
#[test]
fn native_delivery_actual_immutable_get_is_cancelled_or_deadlined_under_shared_budget() {
    for cancel in [true, false] {
        let prepared = prepared();
        let body = serde_json::to_vec(&native(prepared.declaration())).unwrap();
        let loc = locator(prepared.declaration(), &body);
        let peer = Peer::many(vec![
            (status("PENDING"), false),
            (status("COMPLETED"), false),
            (logs(&[log(&loc)]), false),
            (
                b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\nConnection: close\r\n\r\n{".to_vec(),
                true,
            ),
        ]);
        let client = peer.client(TransportLimits::default());
        let deadline = Instant::now() + Duration::from_millis(700);
        let submitted = runtime()
            .block_on(client.submit_certification_until(
                "owner",
                prepared,
                deadline,
                std::future::pending(),
            ))
            .unwrap();
        let publisher = fixture_retrieval_publisher(&client.endpoint);
        let observed = Arc::new(AtomicBool::new(false));
        let flag = observed.clone();
        let cancellation = async {
            if !cancel {
                std::future::pending::<()>().await;
            }
            loop {
                while let Ok(request) = peer.requests.try_recv() {
                    if request.starts_with(b"GET /fixture/evidence/resolve/") {
                        flag.store(true, Ordering::SeqCst);
                        return;
                    }
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        };
        let error = runtime()
            .block_on(client.collect_certification_until(
                "owner",
                &submitted,
                &publisher,
                deadline,
                cancellation,
                MonitorLimits::default(),
            ))
            .err()
            .unwrap();
        if cancel {
            assert!(observed.load(Ordering::SeqCst));
            assert!(error.to_string().contains("cancel"));
        } else {
            assert!(!observed.load(Ordering::SeqCst));
            assert!(error.to_string().contains("deadline"));
            assert_eq!(peer.requests.try_iter().count(), 4);
        }
    }
}

#[test]
fn immutable_failed_native_receipt_is_retrievable_without_certification_acceptance() {
    let prepared = prepared();
    let declaration = prepared.declaration();
    let mut value = native(declaration);
    value["status"] = json!("FAILED");
    value["error"] = json!("inert native phase refusal");
    let bytes = serde_json::to_vec(&value).unwrap();
    let mut locator = locator(declaration, &bytes);
    locator.delivery_complete = false;
    let peer = Peer::many(vec![(response(200, &bytes), false)]);
    let client = peer.client(TransportLimits::default());
    let publisher = fixture_retrieval_publisher(&client.endpoint);
    let observed = runtime()
        .block_on(retrieve_native_job_receipt_until(
            &publisher,
            &locator,
            declaration,
            Instant::now() + Duration::from_secs(3),
            std::future::pending(),
        ))
        .unwrap();
    assert_eq!(observed, value);
    assert!(native_receipt(&observed, &locator, declaration).is_err());
}
