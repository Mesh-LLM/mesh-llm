use super::*;
use crate::plugin::MeshConfig;

fn colliding_selector_config() -> (MeshConfig, tempfile::TempDir, String) {
    let profile_source: MeshConfig = toml::from_str(
        r#"
[defaults.throughput]
threads_batch = 7

[[models]]
model = "profile-source"

[models.throughput]
threads = 11
"#,
    )
    .expect("profile source config parses");
    let colliding_selector = profile_source.models[0]
        .with_profile_defaults(profile_source.defaults.as_ref())
        .derived_profile();
    let temp_dir = tempfile::tempdir().expect("temporary model directory");
    let canonical_path = temp_dir.path().join("canonical.gguf");
    std::fs::write(&canonical_path, b"canonical").expect("canonical model fixture");
    let config = toml::from_str(&format!(
        r#"
[defaults.throughput]
threads_batch = 7

[[models]]
model = "{colliding_selector}"

[models.hardware]
model_path = "{}"

[models.throughput]
threads = 3

[models.advanced.server]
alias = "served-canonical"

[[models]]
model = "profile-source"

[models.throughput]
threads = 11

[models.advanced.server]
alias = "served-profile"
"#,
        canonical_path.display(),
    ))
    .expect("colliding selector config parses");
    (config, temp_dir, colliding_selector)
}

#[test]
fn canonical_selector_wins_over_another_models_derived_profile() {
    let (config, temp_dir, colliding_selector) = colliding_selector_config();
    let canonical_path = temp_dir.path().join("canonical.gguf");

    let resolved = resolve_skippy_config_for_selector(
        SkippyConfigResolveRequest {
            mesh_config: &config,
            model_id: "served-canonical",
            model_path: &canonical_path,
            model_bytes: 1024,
            allocatable_memory_bytes: None,
            request_defaults: None,
            package_generation: None,
            compact_meta: None,
        },
        Some(&colliding_selector),
    )
    .expect("canonical model config resolves");

    assert_eq!(resolved.throughput.threads, Some(3));
    assert_eq!(resolved.throughput.threads_batch, Some(7));
}

#[test]
fn served_alias_and_resolved_path_do_not_select_model_specific_config() {
    let (config, temp_dir, _) = colliding_selector_config();
    let canonical_path = temp_dir.path().join("canonical.gguf");

    let resolved = resolve_skippy_config_for_selector(
        SkippyConfigResolveRequest {
            mesh_config: &config,
            model_id: "served-canonical",
            model_path: &canonical_path,
            model_bytes: 1024,
            allocatable_memory_bytes: None,
            request_defaults: None,
            package_generation: None,
            compact_meta: None,
        },
        None,
    )
    .expect("unconfigured served identity resolves defaults");

    assert_eq!(resolved.throughput.threads, None);
    assert_eq!(resolved.throughput.threads_batch, Some(7));
}
