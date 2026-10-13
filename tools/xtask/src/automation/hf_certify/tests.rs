use super::{
    admission::{Artifact, Input, Mode},
    execution,
};
use serde_json::json;
fn input(mode: Mode) -> (Input, tempfile::TempDir) {
    let root = tempfile::tempdir().unwrap();
    let canonical_root = root.path().canonicalize().unwrap();
    let artifact = |name| Artifact {
        path: canonical_root.join(name),
        sha256: "a".repeat(64),
    };
    let input = Input {
        schema_version: 1,
        mode,
        binary: artifact("binary"),
        supplied_mesh_revision: "b".repeat(40),
        native_profile: "standalone-static-skippy-quantize-cpu".into(),
        projector: artifact("projector.gguf"),
        target_parts: if mode == Mode::MtpAttach {
            vec![artifact("target-1.gguf"), artifact("target-2.gguf")]
        } else {
            vec![]
        },
        expected_parts: if mode == Mode::MtpAttach { 2 } else { 0 },
        mtp_draft: if mode == Mode::MtpAttach {
            Some(artifact("draft.gguf"))
        } else {
            None
        },
        layer_count: 2,
        mtp_layer_count: Some(1),
        ctx_size: 64,
        timeout_secs: 15,
    };
    (input, root)
}
#[test]
fn certification_shapes_refuse_missing_short_duplicate_rosters_and_preserve_declared_order() {
    let (mut i, root) = input(Mode::MtpAttach);
    i.validate().unwrap();
    i.expected_parts = 3;
    assert!(i.validate().is_err());
    i.expected_parts = 2;
    i.target_parts.reverse();
    i.validate().unwrap();
    let report = json!({"projector":i.projector.path,"model_parts":i.target_parts.iter().map(|p|&p.path).collect::<Vec<_>>(),"mtp_draft":i.mtp_draft.as_ref().unwrap().path,"layer_count":2,"mtp_layer_count":1,"ctx_size":64,"session_created":true,"native_mtp_multimodal_feature":true});
    execution::correlate(&i, &report).unwrap();
    let mut wrong_order = report;
    wrong_order["model_parts"].as_array_mut().unwrap().reverse();
    assert!(execution::correlate(&i, &wrong_order).is_err());
    let original = i.target_parts[1].clone();
    i.target_parts[1] = i.target_parts[0].clone();
    assert!(i.validate().is_err());
    i.target_parts[1] = original;
    i.target_parts.reverse();
    i.mtp_draft = None;
    assert!(i.validate().is_err());
    root.close().unwrap();
    let (mut i, root) = input(Mode::ProjectorOnly);
    i.validate().unwrap();
    i.target_parts.push(i.projector.clone());
    assert!(i.validate().is_err());
    root.close().unwrap();
}
#[test]
fn certification_native_reports_require_exact_identity_success_feature_and_closed_fields() {
    let (i, root) = input(Mode::ProjectorOnly);
    let mut report = json!({"projector":i.projector.path,"warmup":true,"loaded":true});
    execution::correlate(&i, &report).unwrap();
    report["loaded"] = json!(false);
    assert!(execution::correlate(&i, &report).is_err());
    report["loaded"] = json!(true);
    report["extra"] = json!("unexpected");
    assert!(execution::correlate(&i, &report).is_err());
    root.close().unwrap();
    let (i, root) = input(Mode::MtpAttach);
    let mut report = json!({"projector":i.projector.path,"model_parts":i.target_parts.iter().map(|p|&p.path).collect::<Vec<_>>(),"mtp_draft":i.mtp_draft.as_ref().unwrap().path,"layer_count":2,"mtp_layer_count":1,"ctx_size":64,"session_created":true,"native_mtp_multimodal_feature":true});
    execution::correlate(&i, &report).unwrap();
    for key in ["session_created", "native_mtp_multimodal_feature"] {
        let mut failed = report.clone();
        failed[key] = json!(false);
        assert!(execution::correlate(&i, &failed).is_err());
    }
    report["mtp_layer_count"] = json!(2);
    assert!(execution::correlate(&i, &report).is_err());
    root.close().unwrap();
}
