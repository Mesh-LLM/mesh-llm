//! Actual shared monitor/log retrieval followed by immutable quant terminal admission.
#![cfg(unix)]
use super::*;
use serde_json::json;
fn submitted(combined: bool) -> SubmittedConversionDelivery {
    let (v, m, p) = super::super::tests::fixture(combined);
    let prepared = super::super::prepare(&serde_json::to_vec(&v).unwrap(), &m, &p).unwrap();
    let mut native = prepared.native.declaration;
    native.submitted = true;
    SubmittedConversionDelivery {
        native: super::super::super::SubmittedCertificationDelivery {
            declaration: native,
            job_id: "job-1".into(),
            stage: JobStage::Completed,
        },
        expected_status: prepared.expected_status,
    }
}
#[test]
fn quant_collection_actual_monitor_locator_immutable_receipt_and_failure_observations() {
    use crate::jobs::{
        TransportLimits,
        transport_tests::{Peer, response, runtime},
    };
    use crate::snapshot_promotion::regular_publication::fixture_retrieval_publisher;
    for mode in [
        "quant",
        "package",
        "failed",
        "incomplete-roster",
        "wrong-request",
    ] {
        let combined = mode == "package";
        let s = submitted(combined);
        let d = &s.native.declaration;
        let workflow = if combined {
            "quantization-and-package"
        } else {
            "quantization"
        };
        let mut native = json!({"schema_version":1,"workflow":workflow,"request_sha256":"b".repeat(64),"transport_input_sha256":d.transport_input_sha256,"status":if mode=="failed"{"FAILED"}else{s.expected_status.as_str()},"error":if mode=="failed"{json!("retained window error")}else{Value::Null},"operator_request_sha256":"e".repeat(64),"operator":{"request_sha256":"e".repeat(64),"status":s.expected_status,"error":null,"completed_job":mode!="failed","full_roster_verified":mode!="incomplete-roster","final_commit":"c".repeat(40),"verify_job":{"completed":true},"package":{"completed":combined,"final_commit":"f".repeat(40)},"windows":[{"ordinal":1,"remote_verified":true}]}});
        if mode == "wrong-request" {
            native["request_sha256"] = json!("0".repeat(64));
        }
        let bytes = serde_json::to_vec(&native).unwrap();
        let locator = Locator {
            schema_version: 1,
            transport_input_sha256: d.transport_input_sha256.clone(),
            receipt_request_sha256: "b".repeat(64),
            repo: d.evidence_repo.clone(),
            parent_commit: d.evidence_parent_commit.clone(),
            commit_oid: "c".repeat(40),
            path_in_repo: d.evidence_path.clone(),
            artifact_sha256: admission::digest(&bytes),
            byte_size: bytes.len() as u64,
            delivery_complete: mode != "failed",
        };
        let logs=format!("data: {}\n",serde_json::to_string(&json!({"data":format!("MESH_NATIVE_DELIVERY {}",serde_json::to_string(&locator).unwrap()),"timestamp":null})).unwrap());
        let peer=Peer::many(vec![(response(200,&serde_json::to_vec(&json!({"id":"job-1","status":{"stage":if mode=="failed"{"ERROR"}else{"COMPLETED"},"message":null}})).unwrap()),false),(response(200,logs.as_bytes()),false),(response(200,&bytes),false)]);
        let client = peer.client(TransportLimits::default());
        let publisher = fixture_retrieval_publisher(&client.endpoint);
        let result = runtime().block_on(client.collect_quantization_until(
            "owner",
            &s,
            &publisher,
            Instant::now() + std::time::Duration::from_secs(3),
            std::future::pending(),
        ));
        if mode == "wrong-request" {
            assert!(result.is_err());
        } else {
            let observed = result.unwrap();
            assert_eq!(
                observed.quantization_admitted,
                matches!(mode, "quant" | "package")
            );
            assert_eq!(observed.package_admitted, combined);
            assert_eq!(observed.native_receipt, native);
            assert!(!observed.image_observed && !observed.cost_observed);
        }
        let requests: Vec<_> = peer.requests.try_iter().collect();
        assert_eq!(requests.len(), 3);
        assert!(String::from_utf8_lossy(&requests[2]).contains(&format!(
            "/resolve/{}/{}",
            "c".repeat(40),
            d.evidence_path
        )));
    }
}
