use super::super::mlock::{TuneMlockLimit, TuneMlockProbe};
use super::helpers::{evaluate_with_probe, sample_gpu, sample_target, survey_with_gpus};
use mesh_llm_config::MeshConfig;

#[test]
fn gpu_tune_reports_mlock_unavailable_reason() {
    let config = MeshConfig::default();
    let target = sample_target(false);
    let survey = survey_with_gpus(vec![sample_gpu(0, "pci:0000:00:00.0", Some("CUDA0"))]);

    let evaluation = evaluate_with_probe(
        &config,
        &target,
        &survey,
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Bytes(64 * 1024),
        },
    )
    .unwrap();

    assert!(!evaluation.mlock.available);
    assert!(
        evaluation
            .mlock
            .reason
            .contains("current lock limit is 64.0 KiB")
    );
    let diagnostics = evaluation.diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].code,
        crate::gpus::tune::TuneDiagnosticCode::MlockUnavailable
    );
}
