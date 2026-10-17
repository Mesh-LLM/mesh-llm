//! Opt-in real native planning gate; the fixture stays outside the repository.
use skippy_api::{source, split_certification, stage_admission, stage_load};
use std::path::PathBuf;

#[test]
#[ignore = "requires SKIPPY_TEST_GGUF_PATH and a prepared static native runtime"]
fn real_direct_model_admits_two_stages_with_exact_tensor_and_frontier_bindings() {
    let path =
        PathBuf::from(std::env::var_os("SKIPPY_TEST_GGUF_PATH").expect("model fixture path"));
    let identity = source::synthetic_direct_gguf_package("pinned-model", &path, None).unwrap();
    let (manifest, _) =
        source::planning::direct_gguf_planning_manifest_from_identity("pinned-model", &identity)
            .unwrap();
    let architecture = manifest.model_metadata["general.architecture"]
        .as_str()
        .unwrap();
    split_certification::require_split_certification(&identity, architecture, false).unwrap();
    assert!(identity.layer_count >= 2);
    let middle = identity.layer_count / 2;
    let profiles = [
        stage_admission::StagePlannerProfile {
            profile_id: "decode".into(),
            n_tokens: 1,
            n_sequences: 1,
            n_outputs: 1,
            n_recurrent_rollback_sequences: 0,
        },
        stage_admission::StagePlannerProfile {
            profile_id: "prefill".into(),
            n_tokens: 8,
            n_sequences: 1,
            n_outputs: 8,
            n_recurrent_rollback_sequences: 0,
        },
    ];
    let admissions = stage_admission::realize_direct_gguf_stage_admissions(
        "pinned-model",
        &identity,
        &[(0, middle), (middle, identity.layer_count)],
        &profiles,
        "skippy-graph-configuration:v1:api-test",
        "skippy-backend:cpu:v1",
    )
    .unwrap();
    assert_eq!(admissions.len(), 2);
    assert_ne!(admissions[0].plan_id, admissions[1].plan_id);
    for admission in &admissions {
        assert_eq!(admission.package_id, manifest.package_id);
        assert!(
            !stage_load::admitted_resident_tensor_names(admission, &manifest)
                .unwrap()
                .is_empty()
        );
        stage_load::admitted_activation_frontier(admission).unwrap();
    }
    let first = stage_load::admitted_activation_frontier(&admissions[0]).unwrap();
    let second = stage_load::admitted_activation_frontier(&admissions[1]).unwrap();
    assert!(!first.activation_exports.is_empty());
    assert_eq!(first.activation_exports, second.activation_imports);
    assert_eq!(
        first.activation_export_bindings,
        second.activation_import_bindings
    );
}
