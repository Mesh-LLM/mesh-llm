#![cfg(unix)]
use super::*;
fn fixture() -> (Value, Vec<ModelMount>, CpuJobPlan) {
    let (original, mounts, plan) = generic::facade_fixture();
    let pin = |p: &str| json!({"path":p,"sha256":"1".repeat(64)});
    let tool = |p: &str| json!({"name":"helper","path":p,"sha256":"1".repeat(64)});
    let source = json!({"repo":"fixture/checkpoint","revision":"a".repeat(40),"files":{"config.json":"1".repeat(64),"model.safetensors":"2".repeat(64)}});
    let op = json!({"schema_version":1,"bootstrap":original["operator"]["bootstrap"],"staging_helper":tool("/opt/stitch"),"checkpoint":source,"tokenizer_source":{"repo":"fixture/tokenizer","revision":"b".repeat(40),"files":{"tokenizer.json":"3".repeat(64)}},"tokenizer_profile":pin("/opt/profile.json"),"credential_file":null,"maximum_bytes":1048576,"target_parts":[pin("/models/target/part1.gguf"),pin("/models/target/part2.gguf")],"sidecars":[],"repository_helper":tool("/opt/repository"),"repository_helper_source":pin("/opt/repository-source"),"publisher_helper":pin("/opt/publisher"),"publisher_source":pin("/opt/publisher-source"),"overall_seconds":259200,"publication_reserve_seconds":60,"dry_run":false,"confirm_publication":true});
    (
        json!({"schema_version":1,"workflow":"default-mtp-composition","timeout_secs":259200,"runner":original["runner"],"operator":op,"receipt_export":original["receipt_export"]}),
        mounts,
        plan,
    )
}
#[test]
fn default_composition_jobs_prepares_distinct_full_worker_and_refuses_foreign_source_budget() {
    let (v, m, p) = fixture();
    let prepared = prepare(&serde_json::to_vec(&v).unwrap(), &m, &p).unwrap();
    assert_eq!(prepared.native.spec.arguments[2], "composition-job-worker");
    assert_eq!(prepared.expected_status, "COMPOSED_PUBLISHED");
    assert_eq!(prepared.native.spec.timeout_seconds, 259200);
    assert!(!prepared.declaration().native_certification_completed);
    for (pointer, value) in [
        ("/workflow", json!("generic-conversion")),
        ("/operator/credential_file", json!("/ambient/token")),
        ("/operator/overall_seconds", json!(86400)),
        (
            "/operator/target_parts/0/path",
            json!("/unmounted/part.gguf"),
        ),
        ("/operator/checkpoint/revision", json!("main")),
    ] {
        let mut changed = v.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert!(prepare(&serde_json::to_vec(&changed).unwrap(), &m, &p).is_err());
    }
}
#[test]
fn default_composition_collection_refuses_plan_only_and_preserves_failed_publication_rows() {
    let (v, m, p) = fixture();
    let prepared = prepare(&serde_json::to_vec(&v).unwrap(), &m, &p).unwrap();
    let mut declaration = prepared.native.declaration;
    declaration.submitted = true;
    let submitted = SubmittedConversionDelivery {
        native: SubmittedCertificationDelivery {
            declaration,
            job_id: "job1".into(),
            stage: JobStage::Completed,
        },
        expected_status: prepared.expected_status,
    };
    let locator = Locator {
        schema_version: 1,
        transport_input_sha256: submitted.native.declaration.transport_input_sha256.clone(),
        receipt_request_sha256: "a".repeat(64),
        repo: "fixture/evidence".into(),
        parent_commit: "b".repeat(40),
        commit_oid: "c".repeat(40),
        path_in_repo: "runs/native-job.json".into(),
        artifact_sha256: "d".repeat(64),
        byte_size: 512,
        delivery_complete: true,
    };
    let native = json!({"schema_version":1,"workflow":"default-mtp-composition","request_sha256":locator.receipt_request_sha256,"transport_input_sha256":locator.transport_input_sha256,"status":"COMPOSITION_COMPLETED","operator_request_sha256":"e".repeat(64),"error":null,"operator":{"request_sha256":"e".repeat(64),"status":"COMPOSED_PUBLISHED","error":null,"publication_plan":{"entries":[1,2,3]},"ordered_publication":{"status":"PUBLISHED","final_receipt":{"publication":{"completed":true}}}}});
    assert!(observe(&native, &locator, &submitted).unwrap());
    for (pointer, value) in [
        ("/operator/status", json!("COMPOSED")),
        (
            "/operator/ordered_publication/final_receipt/publication/completed",
            json!(false),
        ),
        ("/error", json!("terminal refusal")),
        ("/operator/error", json!("prior failure")),
    ] {
        let mut n = native.clone();
        *n.pointer_mut(pointer).unwrap() = value;
        assert!(!observe(&n, &locator, &submitted).unwrap());
        assert_eq!(
            n["operator"]["publication_plan"]["entries"],
            json!([1, 2, 3])
        );
    }
    let mut n = native;
    n["workflow"] = json!("certification");
    assert!(observe(&n, &locator, &submitted).is_err());
}

#[test]
fn default_composition_actual_monitor_and_immutable_retrieval_preserve_complete_and_failed_observations()
 {
    use crate::jobs::{
        TransportLimits,
        transport_tests::{Peer, response, runtime},
    };
    use crate::snapshot_promotion::regular_publication::fixture_retrieval_publisher;
    for mode in ["complete", "remote-failure", "plan-only"] {
        let (request, mounts, plan) = fixture();
        let prepared = prepare(&serde_json::to_vec(&request).unwrap(), &mounts, &plan).unwrap();
        let mut declaration = prepared.native.declaration;
        declaration.submitted = true;
        let submitted = SubmittedConversionDelivery {
            native: SubmittedCertificationDelivery {
                declaration,
                job_id: "job-1".into(),
                stage: JobStage::Completed,
            },
            expected_status: prepared.expected_status,
        };
        let decl = &submitted.native.declaration;
        let mut locator = Locator {
            schema_version: 1,
            transport_input_sha256: decl.transport_input_sha256.clone(),
            receipt_request_sha256: "a".repeat(64),
            repo: decl.evidence_repo.clone(),
            parent_commit: decl.evidence_parent_commit.clone(),
            commit_oid: "c".repeat(40),
            path_in_repo: decl.evidence_path.clone(),
            artifact_sha256: String::new(),
            byte_size: 0,
            delivery_complete: true,
        };
        let native = json!({"schema_version":1,"workflow":"default-mtp-composition","request_sha256":locator.receipt_request_sha256,"transport_input_sha256":locator.transport_input_sha256,"status":"COMPOSITION_COMPLETED","operator_request_sha256":"e".repeat(64),"error":null,"operator":{"request_sha256":"e".repeat(64),"status":if mode=="plan-only"{"COMPOSED"}else{"COMPOSED_PUBLISHED"},"error":null,"publication_plan":{"entries":[1,2,3]},"ordered_publication":{"status":"PUBLISHED","final_receipt":{"publication":{"completed":true}}}}});
        let bytes = serde_json::to_vec(&native).unwrap();
        locator.artifact_sha256 = admission::digest(&bytes);
        locator.byte_size = bytes.len() as u64;
        let log=format!("data: {}\n",serde_json::to_string(&json!({"data":format!("MESH_NATIVE_DELIVERY {}",serde_json::to_string(&locator).unwrap()),"timestamp":null})).unwrap());
        let peer=Peer::many(vec![(response(200,&serde_json::to_vec(&json!({"id":"job-1","status":{"stage":if mode=="remote-failure"{"ERROR"}else{"COMPLETED"},"message":null}})).unwrap()),false),(response(200,log.as_bytes()),false),(response(200,&bytes),false)]);
        let client = peer.client(TransportLimits::default());
        let publisher = fixture_retrieval_publisher(&client.endpoint);
        let observed = runtime()
            .block_on(client.collect_composition_until(
                "owner",
                &submitted,
                &publisher,
                Instant::now() + std::time::Duration::from_secs(3),
                std::future::pending(),
            ))
            .unwrap();
        assert_eq!(observed.composition_admitted, mode == "complete");
        assert_eq!(observed.native_receipt, native);
        assert_eq!(observed.locator.artifact_sha256, locator.artifact_sha256);
        assert!(!observed.image_observed && !observed.cost_observed);
        let requests = peer.requests.try_iter().collect::<Vec<_>>();
        assert_eq!(requests.len(), 3);
        assert!(
            requests[2].starts_with(
                format!(
                    "GET /{}/resolve/{}/{} ",
                    locator.repo, locator.commit_oid, locator.path_in_repo
                )
                .as_bytes()
            )
        );
    }
}

#[test]
fn default_composition_large_mounted_prepare_preserves_complete_roster_and_refuses_locator_drift() {
    let (mut v, mut mounts, plan) = fixture();
    v["operator"]["checkpoint"]["files"] = Value::Object(
        (0..1024)
            .map(|i| (format!("model-{i:04}.safetensors"), json!("1".repeat(64))))
            .collect(),
    );
    let bytes = serde_json::to_vec(&v).unwrap();
    assert!(bytes.len() > 65536);
    let mount = ModelMount {
        repo: "fixture/request".into(),
        revision: "a".repeat(40),
        mount_path: "/models/request".into(),
    };
    mounts.push(mount);
    let mut locator = super::request_transport::MountedRequest {
        schema_version: 1,
        path: "/models/request/composition.json".into(),
        sha256: admission::digest(&bytes),
        byte_size: bytes.len() as u64,
        repo: "fixture/request".into(),
        revision: "a".repeat(40),
    };
    assert!(prepare(&bytes, &mounts, &plan).is_err());
    let prepared = prepare_mounted(&bytes, &mounts, &plan, &locator).unwrap();
    assert_eq!(prepared.native.spec.arguments[2], "composition-job-worker");
    assert_eq!(
        prepared.native.spec.arguments[3],
        "--mounted-input-environment"
    );
    assert_eq!(
        prepared.declaration().transport_input_sha256,
        admission::digest(&bytes)
    );
    assert!(!prepared.native.spec.secrets.contains_key(INPUT_KEY));
    assert_eq!(prepared.expected_status, "COMPOSED_PUBLISHED");
    assert_eq!(
        serde_json::from_str::<Value>(
            &prepared.native.spec.environment[super::request_transport::LOCATOR_ENVIRONMENT]
        )
        .unwrap()["byte_size"],
        bytes.len() as u64
    );
    locator.sha256 = "0".repeat(64);
    assert!(prepare_mounted(&bytes, &mounts, &plan, &locator).is_err());
    locator.sha256 = admission::digest(&bytes);
    locator.revision = "b".repeat(40);
    assert!(prepare_mounted(&bytes, &mounts, &plan, &locator).is_err());
}
