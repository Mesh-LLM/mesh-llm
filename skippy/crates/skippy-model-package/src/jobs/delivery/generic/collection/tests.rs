#![cfg(unix)]
use super::*;
use serde_json::json;
fn fixture() -> (Value, Locator, SubmittedConversionDelivery) {
    let (v, mounts, plan) = super::super::tests::fixture();
    let p = PreparedConversionDelivery::prepare(&serde_json::to_vec(&v).unwrap(), &mounts, &plan)
        .unwrap();
    let mut d = p.native.declaration;
    d.submitted = true;
    let submitted = SubmittedConversionDelivery {
        native: SubmittedCertificationDelivery {
            declaration: d,
            job_id: "job-1".into(),
            stage: JobStage::Completed,
        },
        expected_status: p.expected_status,
    };
    let locator = Locator {
        schema_version: 1,
        transport_input_sha256: submitted.native.declaration.transport_input_sha256.clone(),
        receipt_request_sha256: "b".repeat(64),
        repo: "fixture/evidence".into(),
        parent_commit: "a".repeat(40),
        commit_oid: "c".repeat(40),
        path_in_repo: "runs/native-job.json".into(),
        artifact_sha256: "d".repeat(64),
        byte_size: 512,
        delivery_complete: true,
    };
    let value = json!({"schema_version":1,"workflow":"generic-conversion","status":"CONVERSION_COMPLETED","request_sha256":locator.receipt_request_sha256,"transport_input_sha256":locator.transport_input_sha256,"operator_request_sha256":"e".repeat(64),"error":null,"operator":{"request_sha256":"e".repeat(64),"status":"OPERATOR_COMPLETED","error":null,"conversion_request_sha256":"f".repeat(64),"conversion_receipt":{"request_sha256":"f".repeat(64),"status":"LOCAL_ARTIFACT_READY","error":null}}});
    (value, locator, submitted)
}
#[test]
fn generic_collection_distinguishes_completed_conversion_from_certification_and_failed_evidence() {
    let (v, locator, submitted) = fixture();
    assert!(
        observe(v.clone(), &locator, &submitted)
            .unwrap()
            .conversion_admitted
    );
    for (pointer, change) in [
        ("/status", json!("CERTIFIED")),
        ("/workflow", json!("certification")),
        ("/transport_input_sha256", json!("wrong")),
    ] {
        let mut changed = v.clone();
        *changed.pointer_mut(pointer).unwrap() = change;
        assert!(observe(changed, &locator, &submitted).is_err());
    }
    for (pointer, change) in [
        ("/status", json!("FAILED")),
        ("/error", json!("partial")),
        ("/operator/error", json!("partial")),
        ("/operator_request_sha256", json!(null)),
        ("/operator/conversion_request_sha256", json!("wrong")),
        (
            "/operator/conversion_receipt/request_sha256",
            json!("wrong"),
        ),
        ("/operator/conversion_receipt/status", json!("FAILED")),
    ] {
        let mut changed = v.clone();
        *changed.pointer_mut(pointer).unwrap() = change;
        let evidence = observe(changed.clone(), &locator, &submitted).unwrap();
        assert!(!evidence.conversion_admitted);
        assert_eq!(evidence.native_receipt, changed);
    }
    let mut incomplete = locator;
    incomplete.delivery_complete = false;
    assert!(
        !observe(v, &incomplete, &submitted)
            .unwrap()
            .conversion_admitted
    );
}

fn retrieval_fixture() -> (Value, Locator, SubmittedConversionDelivery) {
    let (value, mut locator, submitted) = fixture();
    let declaration = &submitted.native.declaration;
    locator.repo.clone_from(&declaration.evidence_repo);
    locator
        .parent_commit
        .clone_from(&declaration.evidence_parent_commit);
    locator.path_in_repo.clone_from(&declaration.evidence_path);
    (value, locator, submitted)
}
fn pinned_bytes(value: &Value, locator: &mut Locator) -> Vec<u8> {
    use sha2::{Digest as _, Sha256};
    let bytes = serde_json::to_vec(value).unwrap();
    locator.artifact_sha256 = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    locator.byte_size = bytes.len() as u64;
    bytes
}
fn terminal_response(stage: &str) -> Vec<u8> {
    crate::jobs::transport_tests::response(
        200,
        &serde_json::to_vec(&json!({"id":"job-1","status":{"stage":stage,"message":null}}))
            .unwrap(),
    )
}
fn locator_response(locator: &Locator) -> Vec<u8> {
    let line = format!(
        "MESH_NATIVE_DELIVERY {}",
        serde_json::to_string(locator).unwrap()
    );
    let data = format!(
        "data: {}\n",
        serde_json::to_string(&json!({"data":line,"timestamp":null})).unwrap()
    );
    crate::jobs::transport_tests::response(200, data.as_bytes())
}

#[test]
fn generic_collection_actual_monitor_logs_and_immutable_get_preserve_completed_and_failed_evidence()
{
    use crate::jobs::{
        TransportLimits,
        transport_tests::{Peer, response, runtime},
    };
    use crate::snapshot_promotion::regular_publication::fixture_retrieval_publisher;
    use std::time::Duration;
    for mode in ["completed", "remote-failure", "native-failure"] {
        let (mut value, mut locator, submitted) = retrieval_fixture();
        if mode == "native-failure" {
            value["status"] = json!("FAILED");
            value["error"] = json!("inert conversion refusal after observed phase");
            locator.delivery_complete = false;
        }
        let bytes = pinned_bytes(&value, &mut locator);
        let peer = Peer::many(vec![
            (
                terminal_response(if mode == "remote-failure" {
                    "ERROR"
                } else {
                    "COMPLETED"
                }),
                false,
            ),
            (locator_response(&locator), false),
            (response(200, &bytes), false),
        ]);
        let client = peer.client(TransportLimits::default());
        let publisher = fixture_retrieval_publisher(&client.endpoint);
        let collected = runtime()
            .block_on(client.collect_conversion_until(
                "owner",
                &submitted,
                &publisher,
                Instant::now() + Duration::from_secs(3),
                std::future::pending(),
            ))
            .unwrap();
        assert_eq!(collected.job_id, "job-1");
        assert_eq!(collected.evidence.native_receipt, value);
        assert_eq!(collected.evidence.conversion_admitted, mode == "completed");
        assert_eq!(
            collected.monitor.end,
            if mode == "remote-failure" {
                MonitorEnd::TerminalFailure
            } else {
                MonitorEnd::Completed
            }
        );
        assert!(!collected.image_observed && !collected.cost_observed);
        assert_eq!(collected.locator.artifact_sha256, locator.artifact_sha256);
        let requests = peer.requests.try_iter().collect::<Vec<_>>();
        assert_eq!(requests.len(), 3);
        assert!(requests[0].starts_with(b"GET /api/jobs/owner/job-1 "));
        assert!(requests[1].starts_with(b"GET /api/jobs/owner/job-1/logs "));
        assert!(
            requests[2].starts_with(
                format!(
                    "GET /{}/resolve/{}/{} ",
                    locator.repo, locator.commit_oid, locator.path_in_repo,
                )
                .as_bytes()
            )
        );
    }
}

#[test]
fn generic_collection_actual_locator_and_immutable_bytes_cannot_substitute_request_correlation() {
    use crate::jobs::{
        TransportLimits,
        transport_tests::{Peer, response, runtime},
    };
    use crate::snapshot_promotion::regular_publication::fixture_retrieval_publisher;
    use std::time::Duration;
    for mode in ["foreign-transport", "changed-bytes", "foreign-request"] {
        let (mut value, mut locator, submitted) = retrieval_fixture();
        if mode == "foreign-request" {
            value["request_sha256"] = json!("0".repeat(64));
        }
        let bytes = pinned_bytes(&value, &mut locator);
        if mode == "foreign-transport" {
            locator.transport_input_sha256 = "0".repeat(64);
        }
        let mut replies = vec![
            (terminal_response("COMPLETED"), false),
            (locator_response(&locator), false),
        ];
        if mode != "foreign-transport" {
            replies.push((
                response(
                    200,
                    if mode == "changed-bytes" {
                        b"wrong"
                    } else {
                        &bytes
                    },
                ),
                false,
            ));
        }
        let peer = Peer::many(replies);
        let client = peer.client(TransportLimits::default());
        let publisher = fixture_retrieval_publisher(&client.endpoint);
        assert!(
            runtime()
                .block_on(client.collect_conversion_until(
                    "owner",
                    &submitted,
                    &publisher,
                    Instant::now() + Duration::from_secs(3),
                    std::future::pending(),
                ))
                .is_err()
        );
        assert_eq!(
            peer.requests.try_iter().count(),
            if mode == "foreign-transport" { 2 } else { 3 }
        );
    }
}
