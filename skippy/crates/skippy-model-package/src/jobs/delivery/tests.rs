use super::super::{
    HardwareFlavor, TransportLimits, plan_cpu_job_from_hardware,
    transport_tests::{Peer, response, runtime},
};
use super::*;
use serde_json::json;
use std::time::Duration;
pub(super) fn fixture() -> (Value, Vec<ModelMount>, CpuJobPlan) {
    let hardware = [HardwareFlavor {
        name: "cpu-basic".into(),
        pretty_name: None,
        cpu: Some("2 vCPU".into()),
        ram: Some("16 GB".into()),
        accelerator: None,
        unit_cost_usd: Some(0.01),
        unit_cost_micro_usd: None,
        unit_label: Some("minute".into()),
    }];
    let plan = plan_cpu_job_from_hardware(&hardware, "cpu-basic", 30, 1024).unwrap();
    assert_eq!(plan.timeout_seconds, 7200);
    let pin = |path: &str| json!({"path":path,"sha256":"1".repeat(64)});
    let tools = [
        "git", "just", "cargo", "rustc", "cmake", "c++", "ld.lld", "curl",
    ]
    .map(|name| json!({"name":name,"path":format!("/opt/tools/{name}"),"sha256":"f".repeat(64)}));
    let request = json!({"schema_version":1,"workflow":"certification","timeout_secs":plan.timeout_seconds,
        "runner":pin("/opt/mesh/xtask"),"bootstrap":{"schema_version":1,"mesh_commit":"a".repeat(40),"git_tree":"b".repeat(40),"llama_commit":"c".repeat(40),"upstream_file_sha256":"d".repeat(64),"image":format!("provided/native@sha256:{}","e".repeat(64)),"native_profile":"standalone-static-skippy-quantize-cpu","tools":tools,"path_directories":["/opt/tools"],"timeout_seconds":plan.timeout_seconds,"cpu_plan_receipt_sha256":admission::digest(&serde_json::to_vec(&plan).unwrap()),"declared_estimate_usd":plan.max_cost_usd,"max_cost_usd":plan.max_cost_usd},
        "certification":{"mode":"mtp-attach","projector":pin("/models/projector/mmproj.gguf"),"target_parts":[pin("/models/target/model-00001-of-00002.gguf"),pin("/models/target/model-00002-of-00002.gguf")],"expected_parts":2,"mtp_draft":pin("/models/draft/mtp.gguf"),"layer_count":89,"mtp_layer_count":1,"ctx_size":64},
        "projector":{"kind":"supplied","artifact":pin("/models/projector/mmproj.gguf")}});
    let mut request = request;
    request["receipt_export"] = json!({"helper":pin("/opt/mesh/model-package-publish"),"helper_source":pin("/opt/mesh/publisher-source"),"repo":"fixture/evidence","parent_commit":"a".repeat(40),"credential_file":null,"credential_environment":true,"path_in_repo":"runs/native-job.json"});
    let mounts = ["target", "draft", "projector"]
        .map(|name| ModelMount {
            repo: format!("fixture/{name}"),
            revision: "a".repeat(40),
            mount_path: format!("/models/{name}"),
        })
        .into();
    (request, mounts, plan)
}
#[test]
fn certification_delivery_uses_actual_planner_fixed_native_command_and_secret_transport() {
    let (request, mounts, plan) = fixture();
    let prepared = PreparedCertificationDelivery::prepare(
        &serde_json::to_vec(&request).unwrap(),
        &mounts,
        &plan,
    )
    .unwrap();
    assert_eq!(prepared.spec.command, ["/opt/mesh/xtask"]);
    assert_eq!(
        prepared.spec.arguments,
        [
            "automation",
            "hf-certify",
            "job-worker",
            "--input-environment",
            INPUT_KEY,
            "--output-directory",
            OUTPUT
        ]
    );
    assert!(prepared.spec.environment.is_empty());
    assert_eq!(prepared.spec.secrets.len(), 1);
    let consumed: Value = serde_json::from_str(&prepared.spec.secrets[INPUT_KEY]).unwrap();
    assert_eq!(consumed, request);
    assert_eq!(
        prepared.declaration.transport_input_sha256,
        admission::digest(prepared.spec.secrets[INPUT_KEY].as_bytes())
    );
    assert_eq!(prepared.spec.timeout_seconds, 7200);
    assert!(
        prepared
            .spec
            .volumes
            .iter()
            .all(|v| v.volume_type == "model"
                && v.read_only == Some(true)
                && v.revision.as_deref() == Some("a".repeat(40).as_str()))
    );
    let public = serde_json::to_string(prepared.declaration()).unwrap();
    assert!(!public.contains("/models/target") && !public.contains("tools"));
    assert!(
        !prepared.declaration.image_observed
            && !prepared.declaration.submitted
            && !prepared.declaration.native_certification_completed
    );
}
#[test]
fn certification_delivery_refuses_mutable_mounts_escaping_artifacts_and_plan_drift() {
    let (request, mounts, plan) = fixture();
    for changed in ["revision", "mount_path", "repo"] {
        let mut mounts = mounts.clone();
        match changed {
            "revision" => mounts[0].revision = "main".into(),
            "mount_path" => mounts[0].mount_path = "/work".into(),
            _ => mounts[0].repo = "owner/repo/extra".into(),
        };
        assert!(
            PreparedCertificationDelivery::prepare(
                &serde_json::to_vec(&request).unwrap(),
                &mounts,
                &plan
            )
            .is_err()
        );
    }
    for path in [
        "/models/target/../escape.gguf",
        "/ambient/model.gguf",
        "relative.gguf",
    ] {
        let mut changed = request.clone();
        changed["certification"]["target_parts"][0]["path"] = json!(path);
        assert!(
            PreparedCertificationDelivery::prepare(
                &serde_json::to_vec(&changed).unwrap(),
                &mounts,
                &plan
            )
            .is_err()
        );
    }
    for key in [
        "cpu_plan_receipt_sha256",
        "timeout_seconds",
        "declared_estimate_usd",
        "image",
    ] {
        let mut changed = request.clone();
        changed["bootstrap"][key] = json!("wrong");
        assert!(
            PreparedCertificationDelivery::prepare(
                &serde_json::to_vec(&changed).unwrap(),
                &mounts,
                &plan
            )
            .is_err()
        );
    }
    let mut changed = request.clone();
    changed["workflow"] = json!("nemotron-compose");
    assert!(
        PreparedCertificationDelivery::prepare(
            &serde_json::to_vec(&changed).unwrap(),
            &mounts,
            &plan
        )
        .is_err()
    );
}
#[test]
fn certification_delivery_actual_submit_transports_exact_native_input_without_claiming_pass() {
    let (request, mounts, plan) = fixture();
    let prepared = PreparedCertificationDelivery::prepare(
        &serde_json::to_vec(&request).unwrap(),
        &mounts,
        &plan,
    )
    .unwrap();
    let expected = prepared.declaration.transport_input_sha256.clone();
    let peer = Peer::many(vec![(
        response(
            200,
            b"{\"id\":\"native-1\",\"status\":{\"stage\":\"PENDING\",\"message\":null}}",
        ),
        false,
    )]);
    let prepared = prepared
        .with_publication_credential("inert-explicit-publication".into())
        .unwrap();
    let received = runtime()
        .block_on(
            peer.client(TransportLimits::default())
                .submit_certification_until(
                    "owner",
                    prepared,
                    Instant::now() + Duration::from_secs(3),
                    std::future::pending(),
                ),
        )
        .unwrap();
    assert!(received.declaration.submitted && !received.declaration.native_certification_completed);
    assert_eq!(received.declaration.transport_input_sha256, expected);
    assert_eq!(received.job_id, "native-1");
    let raw = peer.requests.recv_timeout(Duration::from_secs(1)).unwrap();
    let end = raw.windows(4).position(|b| b == b"\r\n\r\n").unwrap() + 4;
    let body: Value = serde_json::from_slice(&raw[end..]).unwrap();
    assert_eq!(body["command"], json!(["/opt/mesh/xtask"]));
    assert_eq!(body["arguments"][3], "--input-environment");
    assert_eq!(
        serde_json::from_str::<Value>(body["secrets"][INPUT_KEY].as_str().unwrap()).unwrap(),
        request
    );
    assert!(
        body["volumes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v["readOnly"] == true)
    );
}
#[test]
fn certification_delivery_precancel_and_expired_deadline_send_nothing() {
    for cancel in [true, false] {
        let (request, mounts, plan) = fixture();
        let prepared = PreparedCertificationDelivery::prepare(
            &serde_json::to_vec(&request).unwrap(),
            &mounts,
            &plan,
        )
        .unwrap();
        let prepared = prepared
            .with_publication_credential("inert-explicit-publication".into())
            .unwrap();
        let peer = Peer::many(vec![]);
        let deadline = if cancel {
            Instant::now() + Duration::from_secs(1)
        } else {
            Instant::now()
        };
        let cancelled = async move {
            if !cancel {
                std::future::pending::<()>().await;
            }
        };
        assert!(
            runtime()
                .block_on(
                    peer.client(TransportLimits::default())
                        .submit_certification_until("owner", prepared, deadline, cancelled)
                )
                .is_err()
        );
        assert!(peer.requests.try_recv().is_err());
    }
}

#[test]
fn certification_delivery_http_failure_and_causal_inflight_cancel_do_not_claim_submission() {
    let (request, mounts, plan) = fixture();
    let make = || {
        PreparedCertificationDelivery::prepare(
            &serde_json::to_vec(&request).unwrap(),
            &mounts,
            &plan,
        )
        .unwrap()
        .with_publication_credential("inert-explicit-publication".into())
        .unwrap()
    };
    let peer = Peer::many(vec![(response(503, b"private-provider-diagnostic"), false)]);
    let error = runtime()
        .block_on(
            peer.client(TransportLimits::default())
                .submit_certification_until(
                    "owner",
                    make(),
                    Instant::now() + Duration::from_secs(2),
                    std::future::pending(),
                ),
        )
        .err()
        .unwrap();
    assert!(!error.to_string().contains("private-provider-diagnostic"));
    for cancel in [true, false] {
        let peer = Peer::many(vec![(
            b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\nConnection: close\r\n\r\n{".to_vec(),
            true,
        )]);
        let observed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = observed.clone();
        let cancellation = async {
            if !cancel {
                std::future::pending::<()>().await;
            }
            loop {
                if peer.requests.try_recv().is_ok() {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    return;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        };
        let error = runtime()
            .block_on(
                peer.client(TransportLimits::default())
                    .submit_certification_until(
                        "owner",
                        make(),
                        Instant::now() + Duration::from_millis(300),
                        cancellation,
                    ),
            )
            .err()
            .unwrap();
        if cancel {
            assert!(observed.load(std::sync::atomic::Ordering::SeqCst));
            assert!(error.to_string().contains("acceptance unknown"));
        } else {
            assert!(!observed.load(std::sync::atomic::Ordering::SeqCst));
            assert!(peer.requests.recv_timeout(Duration::from_secs(1)).is_ok());
        }
    }
}
