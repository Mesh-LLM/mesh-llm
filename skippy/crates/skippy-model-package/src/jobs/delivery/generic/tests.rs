#![cfg(unix)]
use super::*;
use crate::jobs::{
    HardwareFlavor, TransportLimits, plan_cpu_job_from_hardware,
    transport_tests::{Peer, response, runtime},
};
use serde_json::json;
use std::time::Duration;
pub(super) fn fixture() -> (Value, Vec<ModelMount>, CpuJobPlan) {
    let (cert, mounts, _) = super::super::tests::fixture();
    let plan = plan_cpu_job_from_hardware(
        &[HardwareFlavor {
            name: "cpu-basic".into(),
            pretty_name: None,
            cpu: Some("2 vCPU".into()),
            ram: Some("16 GB".into()),
            accelerator: None,
            unit_cost_usd: Some(0.01),
            unit_cost_micro_usd: None,
            unit_label: Some("minute".into()),
        }],
        "cpu-basic",
        259200,
        1024,
    )
    .unwrap();
    let mut bootstrap = cert["bootstrap"].clone();
    bootstrap["timeout_seconds"] = json!(plan.timeout_seconds);
    bootstrap["cpu_plan_receipt_sha256"] =
        json!(admission::digest(&serde_json::to_vec(&plan).unwrap()));
    bootstrap["declared_estimate_usd"] = json!(plan.max_cost_usd);
    bootstrap["max_cost_usd"] = json!(plan.max_cost_usd);
    let v = json!({"schema_version":1,"workflow":"generic-conversion","timeout_secs":259200,"runner":cert["runner"],"operator":{"schema_version":1,"bootstrap":bootstrap,"conversion":{"schema_version":1,"source_repo":"fixture/target","target_repo":"fixture/result","mesh_revision":"a".repeat(40),"output_basename":"model","source":"/models/target","source_files":[{"path":"/models/target/config.json","sha256":"1".repeat(64)}],"work_directory":"/work/conversion","expected_splits":1,"upload_only":false,"dry_run":false,"publish_confirmed":false,"credential_file":null,"timeout_seconds":259200}},"receipt_export":cert["receipt_export"]});
    (v, mounts, plan)
}
#[test]
fn generic_delivery_original_72h_immutable_mounts_and_secret_native_command() {
    let (v, mounts, plan) = fixture();
    let p = PreparedConversionDelivery::prepare(&serde_json::to_vec(&v).unwrap(), &mounts, &plan)
        .unwrap();
    assert_eq!(p.native.spec.timeout_seconds, 259200);
    assert_eq!(p.native.spec.arguments[2], "generic-job-worker");
    assert!(p.native.spec.environment.is_empty());
    assert_eq!(p.native.spec.secrets.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(&p.native.spec.secrets[INPUT_KEY]).unwrap(),
        v
    );
    assert!(p.native.spec.volumes.iter().all(
        |m| m.read_only == Some(true) && m.revision.as_deref() == Some("a".repeat(40).as_str())
    ));
    assert_eq!(p.expected_status, "LOCAL_ARTIFACT_READY");
    assert!(
        !p.declaration().native_certification_completed
            && !p.declaration().submitted
            && !p.declaration().image_observed
    );
    assert_eq!(
        collection::monitor_limits().poll_interval,
        Duration::from_secs(10)
    );
    assert!(u64::from(collection::monitor_limits().max_polls) * 10 > 259200);
    assert_eq!(
        crate::jobs::MonitorLimits::default().poll_interval,
        Duration::from_secs(3)
    );
}
#[test]
fn generic_delivery_unknown_workflow_mount_source_budget_and_ambient_secret_refuse() {
    let (v, mounts, plan) = fixture();
    for (pointer, change) in [
        ("/workflow", json!("certification")),
        ("/timeout_secs", json!(259201)),
        (
            "/operator/conversion/source_files/0/path",
            json!("/models/target/../escape"),
        ),
        (
            "/operator/conversion/credential_file",
            json!("/ambient/token"),
        ),
        ("/operator/conversion/timeout_seconds", json!(86400)),
        ("/operator/conversion/upload_only", json!(true)),
        ("/operator/conversion/mesh_revision", json!("main")),
    ] {
        let mut changed = v.clone();
        *changed.pointer_mut(pointer).unwrap() = change;
        assert!(
            PreparedConversionDelivery::prepare(
                &serde_json::to_vec(&changed).unwrap(),
                &mounts,
                &plan
            )
            .is_err()
        );
    }
    let mut changed = mounts;
    changed[1].mount_path = changed[0].mount_path.clone();
    assert!(
        PreparedConversionDelivery::prepare(&serde_json::to_vec(&v).unwrap(), &changed, &plan)
            .is_err()
    );
}
#[test]
fn generic_delivery_actual_submit_keeps_exact_request_and_never_calls_it_certification() {
    let (v, mounts, plan) = fixture();
    let p = PreparedConversionDelivery::prepare(&serde_json::to_vec(&v).unwrap(), &mounts, &plan)
        .unwrap()
        .with_publication_credential("inert-publication-token".into())
        .unwrap();
    let peer = Peer::many(vec![(
        response(
            200,
            b"{\"id\":\"generic-1\",\"status\":{\"stage\":\"PENDING\",\"message\":null}}",
        ),
        false,
    )]);
    let submitted = runtime()
        .block_on(
            peer.client(TransportLimits::default())
                .submit_conversion_until(
                    "owner",
                    p,
                    Instant::now() + Duration::from_secs(3),
                    std::future::pending(),
                ),
        )
        .unwrap();
    assert_eq!(submitted.native.job_id, "generic-1");
    assert!(
        submitted.native.declaration.submitted
            && !submitted.native.declaration.native_certification_completed
    );
    let raw = peer.requests.recv_timeout(Duration::from_secs(1)).unwrap();
    let end = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let body: Value = serde_json::from_slice(&raw[end..]).unwrap();
    assert_eq!(body["arguments"][2], "generic-job-worker");
    assert_eq!(body["timeoutSeconds"], 259200);
    assert_eq!(
        serde_json::from_str::<Value>(body["secrets"][INPUT_KEY].as_str().unwrap()).unwrap(),
        v
    );
}
#[test]
fn generic_submit_expired_and_held_requests_do_not_retry_or_claim_remote_completion() {
    for held in [false, true] {
        let (v, mounts, plan) = fixture();
        let prepared =
            PreparedConversionDelivery::prepare(&serde_json::to_vec(&v).unwrap(), &mounts, &plan)
                .unwrap()
                .with_publication_credential("inert-publication-token".into())
                .unwrap();
        let peer = Peer::many(if held {
            vec![(response(200, b"{}"), true)]
        } else {
            vec![]
        });
        let until = if held {
            Instant::now() + Duration::from_millis(150)
        } else {
            Instant::now()
        };
        assert!(
            runtime()
                .block_on(
                    peer.client(TransportLimits::default())
                        .submit_conversion_until("owner", prepared, until, std::future::pending())
                )
                .is_err()
        );
        if held {
            assert!(peer.requests.recv_timeout(Duration::from_secs(1)).is_ok());
        }
        assert!(peer.requests.try_recv().is_err());
    }
}

#[test]
fn generic_upload_only_delivery_requires_complete_immutable_mounted_artifact() {
    let (mut v, mounts, plan) = fixture();
    let c = &mut v["operator"]["conversion"];
    c["upload_only"] = json!(true);
    c["source_files"] = json!([]);
    c["target_prefix"] = json!("BF16");
    v["upload_artifact"] = json!({"schema_version":1,"repo":"fixture/target","revision":"a".repeat(40),"source_directory":"/models/target","work_directory":"/work/conversion","target_prefix":"BF16","output_basename":"model","requested_splits":1,"timeout_seconds":259200,"files":[{"name":"README.md","sha256":"1".repeat(64),"byte_size":4},{"name":"skippy-convert-manifest.json","sha256":"2".repeat(64),"byte_size":64},{"name":"model.gguf","sha256":"3".repeat(64),"byte_size":16}]});
    let p = PreparedConversionDelivery::prepare(&serde_json::to_vec(&v).unwrap(), &mounts, &plan)
        .unwrap();
    assert_eq!(p.expected_status, "LOCAL_ARTIFACT_READY");
    assert_eq!(
        serde_json::from_str::<Value>(&p.native.spec.secrets[INPUT_KEY]).unwrap()["operator"]["conversion"]
            ["upload_only"],
        true
    );
    for (pointer, change) in [
        ("/upload_artifact/revision", json!("b".repeat(40))),
        ("/upload_artifact/work_directory", json!("/work/other")),
        ("/upload_artifact/files/0/name", json!("../escape")),
        ("/upload_artifact/files/0/sha256", json!("bad")),
    ] {
        let mut changed = v.clone();
        *changed.pointer_mut(pointer).unwrap() = change;
        assert!(
            PreparedConversionDelivery::prepare(
                &serde_json::to_vec(&changed).unwrap(),
                &mounts,
                &plan
            )
            .is_err()
        );
    }
}
