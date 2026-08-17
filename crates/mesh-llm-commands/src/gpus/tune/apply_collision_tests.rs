use crate::gpus::tune_apply::{PreparedTunePlan, apply_prepared_tune_plans};
use crate::gpus::tune_resolver::{
    ConfigModelMatch, LocalTargetSource, ResolvedTuneTarget, TuneTargetSelection,
};
use mesh_llm_config::ConfigStore;
use model_hf::store::model_ref_for_path;
use tempfile::tempdir;

use super::*;

#[test]
fn gpu_tune_apply_aborts_on_duplicate_config_collision_without_partial_write() {
    let temp = tempdir().expect("tempdir should be created");
    let duplicate_path = write_local_gguf_file(temp.path(), "duplicate.gguf");
    let appended_path = write_local_gguf_file(temp.path(), "append.gguf");
    let duplicate_canonical = duplicate_path
        .canonicalize()
        .expect("duplicate fixture should canonicalize");
    let appended_canonical = appended_path
        .canonicalize()
        .expect("append fixture should canonicalize");
    let raw_config = format!(
        "# do not change on collision\nversion = 1\n\n[[models]]\nmodel = \"{}\"\n\n[[models]]\nmodel = \"{}\"\n",
        duplicate_canonical.display(),
        duplicate_canonical.display()
    );
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, &raw_config).expect("fixture config should be written");
    let config = mesh_llm_config::MeshConfig {
        models: vec![mesh_llm_config::ModelConfigEntry {
            model: duplicate_canonical.display().to_string(),
            ..mesh_llm_config::ModelConfigEntry::default()
        }],
        ..mesh_llm_config::MeshConfig::default()
    };

    let colliding_target = ResolvedTuneTarget {
        requested_input: duplicate_canonical.display().to_string(),
        canonical_model_ref: model_ref_for_path(&duplicate_canonical),
        resolved_path: duplicate_canonical.clone(),
        local_source: LocalTargetSource::FilesystemPath {
            synthetic_model_ref: model_ref_for_path(&duplicate_canonical),
        },
        config_matches: vec![
            ConfigModelMatch {
                row_index: 0,
                configured_model: duplicate_canonical.display().to_string(),
            },
            ConfigModelMatch {
                row_index: 1,
                configured_model: duplicate_canonical.display().to_string(),
            },
        ],
        selection: TuneTargetSelection::Explicit { configured: true },
    };
    let append_target = appended_target(&appended_canonical);
    let collision_plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &config,
        target: &colliding_target,
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });
    let append_plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &config,
        target: &append_target,
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    let store = ConfigStore::open(&config_path);
    let result = apply_prepared_tune_plans(
        &store,
        &[
            PreparedTunePlan::new(colliding_target, collision_plan),
            PreparedTunePlan::new(append_target, append_plan),
        ],
    );

    assert!(
        result.is_err(),
        "duplicate config collision should abort apply"
    );
    let error = result.expect_err("collision should be reported");
    assert!(
        error
            .to_string()
            .contains("collides with multiple config rows")
    );
    assert_eq!(
        std::fs::read_to_string(&config_path).expect("config should still be readable"),
        raw_config,
        "global safety errors must not partially write the config",
    );
}
