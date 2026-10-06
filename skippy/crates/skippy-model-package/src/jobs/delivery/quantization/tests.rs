#![cfg(unix)]
use super::*;
use serde_json::json;
pub(in crate::jobs::delivery) fn fixture(combined: bool) -> (Value, Vec<ModelMount>, CpuJobPlan) {
    let (cert, mounts, mut plan) = super::super::tests::fixture();
    plan.timeout_seconds = if combined { 345600 } else { 259200 };
    plan.requested_timeout_seconds = plan.timeout_seconds;
    plan.timeout_bumped_to_minimum = false;
    plan.max_cost_usd =
        estimate_cost_usd(plan.unit_cost_usd, &plan.unit_label, plan.timeout_seconds).unwrap();
    let pin = |p: &str| json!({"path":p,"sha256":"1".repeat(64)});
    let workflow = if combined {
        "quantization-and-package"
    } else {
        "quantization"
    };
    let package = if combined {
        json!({"writer":pin("/opt/mesh/package"),"writer_source":pin("/opt/mesh/package-source"),"generation_defaults":pin("/opt/mesh/defaults"),"target_repo":"fixture/package","max_artifact_bytes":1048576})
    } else {
        Value::Null
    };
    let template = json!({"schema_version":1,"tool_kind":"supplied-window-quantizer","profile_version":"fixture-v1","tool":pin("/opt/mesh/quant"),"tool_source":pin("/opt/mesh/quant-source"),"runtime":pin("/opt/mesh/runtime"),"manifest":pin("/models/target/manifest.json"),"recipe":pin("/models/target/recipe.txt"),"helper":pin("/opt/mesh/publisher"),"helper_source":pin("/opt/mesh/publisher-source"),"source_repo":"fixture/target","source_revision":"a".repeat(40),"source_root":"/models/target","source_prefix":"BF16","source_parts":[pin("/models/target/BF16/model-00001-of-00002.gguf"),pin("/models/target/BF16/model-00002-of-00002.gguf")],"target_repo":"fixture/quant","target_root":"/work/quant-target","target_prefix":"CUSTOM","basename":"model","quant":"CUSTOM","expected_splits":2,"ordinal":1,"work_root":"/work/quant-work","credential_file":null,"publication_confirmed":true,"timeout_seconds":86400,"resume":null});
    let mut export = cert["receipt_export"].clone();
    export["export_budget_secs"] = json!(60);
    let input = json!({"schema_version":1,"workflow":workflow,"timeout_secs":plan.timeout_seconds,"runner":cert["runner"],"authority":{"schema_version":1,"image":format!("supplied/quant@sha256:{}","e".repeat(64)),"mesh_commit":"a".repeat(40),"git_tree":"b".repeat(40),"cpu_plan_receipt_sha256":admission::digest(&serde_json::to_vec(&plan).unwrap()),"declared_estimate_usd":plan.max_cost_usd,"max_cost_usd":plan.max_cost_usd,"tool_kind":"supplied-window-quantizer","profile_version":"fixture-v1"},"operator":{"schema_version":1,"workflow":workflow,"timeout_seconds":plan.timeout_seconds,"window_template":template,"resumes":[],"loader":pin("/opt/mesh/loader"),"package":package},"receipt_export":export});
    (input, mounts, plan)
}
#[test]
fn quant_delivery_preserves_72h_96h_cost_and_supplied_tool_declaration_without_g3_build_claim() {
    for combined in [false, true] {
        let (v, m, p) = fixture(combined);
        let prepared = prepare(&serde_json::to_vec(&v).unwrap(), &m, &p).unwrap();
        assert_eq!(prepared.native.spec.arguments[2], "quant-job-worker");
        assert_eq!(prepared.native.spec.timeout_seconds, p.timeout_seconds);
        assert!(
            prepared
                .native
                .spec
                .volumes
                .iter()
                .all(|v| v.read_only == Some(true) && v.revision.is_some())
        );
        assert!(
            !prepared.declaration().image_observed
                && !prepared.declaration().submitted
                && !prepared.declaration().native_certification_completed
        );
        assert_eq!(
            prepared.expected_status(),
            if combined {
                "QUANTIZATION_PACKAGED"
            } else {
                "QUANTIZATION_PUBLISHED"
            }
        );
        let limits =
            super::super::generic::collection::quant_monitor_limits(p.timeout_seconds).unwrap();
        assert!(u64::from(limits.max_polls) * limits.poll_interval.as_secs() > p.timeout_seconds);
        assert_eq!(
            limits.poll_interval.as_secs(),
            if combined { 15 } else { 10 }
        );
        assert_eq!(
            super::super::generic::monitor_limits()
                .poll_interval
                .as_secs(),
            10
        );
    }
}
#[test]
fn quant_delivery_rejects_partial_source_authority_and_budget_drift() {
    let (v, m, p) = fixture(false);
    for (pointer, value) in [
        (
            "/operator/window_template/source_revision",
            json!("b".repeat(40)),
        ),
        ("/authority/image", json!("mutable:latest")),
        ("/authority/max_cost_usd", json!(0)),
        (
            "/operator/window_template/credential_file",
            json!("/tmp/token"),
        ),
        (
            "/operator/window_template/publication_confirmed",
            json!(false),
        ),
        ("/operator/window_template/expected_splits", json!(3)),
        ("/operator/timeout_seconds", json!(345600)),
        ("/operator/package", json!({})),
    ] {
        let mut bad = v.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            prepare(&serde_json::to_vec(&bad).unwrap(), &m, &p).is_err(),
            "{pointer}"
        );
    }
    assert!(super::super::generic::collection::quant_monitor_limits(345601).is_err());
}
#[test]
fn quant_mounted_request_preserves_exact_large_transport_hash_and_private_inline_compatibility() {
    let (mut v, mut mounts, p) = fixture(false);
    let t = &mut v["operator"]["window_template"];
    t["expected_splits"] = json!(1024);
    t["source_parts"]=json!((1..=1024).map(|i|json!({"path":format!("/models/target/BF16/model-{i:05}-of-01024.gguf"),"sha256":"1".repeat(64)})).collect::<Vec<_>>());
    let bytes = serde_json::to_vec(&v).unwrap();
    assert!(bytes.len() > 65536);
    assert!(prepare(&bytes, &mounts, &p).is_err());
    mounts.push(ModelMount {
        repo: "fixture/request".into(),
        revision: "c".repeat(40),
        mount_path: "/models/request".into(),
    });
    let mut locator = request_transport::MountedRequest {
        schema_version: 1,
        path: "/models/request/quant.json".into(),
        sha256: admission::digest(&bytes),
        byte_size: bytes.len() as u64,
        repo: "fixture/request".into(),
        revision: "c".repeat(40),
    };
    let prepared = prepare_mounted(&bytes, &mounts, &p, &locator).unwrap();
    assert_eq!(
        prepared.declaration().transport_input_sha256,
        locator.sha256
    );
    assert!(prepared.native.spec.secrets.is_empty());
    assert_eq!(
        prepared.native.spec.arguments[3],
        "--mounted-input-environment"
    );
    locator.sha256 = "f".repeat(64);
    assert!(prepare_mounted(&bytes, &mounts, &p, &locator).is_err());
    let (v, m, p) = fixture(false);
    let inline = prepare(&serde_json::to_vec(&v).unwrap(), &m, &p).unwrap();
    assert!(inline.native.spec.secrets.contains_key(INPUT_KEY));
}
