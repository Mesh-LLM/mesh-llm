use super::*;
use std::path::PathBuf;

#[test]
#[ignore = "requires SKIPPY_PACKAGE_V2_TEST_DIR and the prepared native runtime"]
fn realizes_and_admits_a_real_package_v2_chain() {
    let package_dir = std::env::var_os("SKIPPY_PACKAGE_V2_TEST_DIR")
        .map(PathBuf::from)
        .expect("SKIPPY_PACKAGE_V2_TEST_DIR is required");
    let admissions = realize_stage_admissions(
        &package_dir,
        &[(0, 16), (16, 32)],
        &[
            StagePlannerProfile {
                profile_id: "batched".to_string(),
                n_tokens: 8,
                n_sequences: 2,
                n_outputs: 8,
                n_recurrent_rollback_sequences: 0,
            },
            StagePlannerProfile {
                profile_id: "decode".to_string(),
                n_tokens: 1,
                n_sequences: 1,
                n_outputs: 1,
                n_recurrent_rollback_sequences: 0,
            },
            StagePlannerProfile {
                profile_id: "prefill".to_string(),
                n_tokens: 8,
                n_sequences: 1,
                n_outputs: 8,
                n_recurrent_rollback_sequences: 0,
            },
        ],
        "skippy-graph-configuration:v1:real-package-test",
        "skippy-backend:cpu:v1",
    )
    .expect("real package-v2 chain must admit");
    assert_eq!(admissions.len(), 2);
    assert_eq!(admissions[0].package_id, admissions[1].package_id);
    assert_ne!(admissions[0].plan_id, admissions[1].plan_id);
    assert!(
        admissions
            .iter()
            .all(|admission| !admission.resident_tensor_ids.is_empty())
    );
    assert!(
        admissions
            .iter()
            .all(|admission| admission.profiles.len() == 3)
    );
    let package_ref = package_dir.to_string_lossy();
    for admission in &admissions {
        let descriptor = skippy_package_format::stage_admission::StageAdmissionDescriptor {
            package_id: admission.package_id.clone(),
            resident_tensor_ids: admission.resident_tensor_ids.clone(),
            sidecars: Vec::new(),
        };
        let (_, model_parts, projector) =
            crate::inference::skippy::resolve_package_v2_stage_to_local(&package_ref, &descriptor)
                .expect("real package-v2 stage artifacts must resolve");
        assert!(projector.is_none());
        assert!(
            model_parts
                .iter()
                .any(|path| path.ends_with("shared/metadata.gguf"))
        );
        assert!(
            model_parts
                .iter()
                .all(|path| path.starts_with(&package_dir))
        );
        assert!(
            model_parts
                .iter()
                .all(|path| !path.starts_with(package_dir.join("artifacts")))
        );
    }
}
