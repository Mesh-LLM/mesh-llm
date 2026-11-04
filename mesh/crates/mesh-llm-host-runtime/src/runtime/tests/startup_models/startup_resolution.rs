use super::*;
use crate::mesh::StartupResolution;

/// A request waits on a resolving startup model only under a name it can have
/// before it resolves, unless a bare local GGUF, named only once hashed, is
/// among them.
#[test]
fn startup_resolution_waits_only_on_names_known_before_resolving() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bare = dir.path().join("bare.gguf");
    std::fs::write(&bare, b"GGUF").expect("write GGUF");
    let resolution = |specs: &[StartupModelSpec]| {
        let requested: Vec<String> = specs
            .iter()
            .map(|spec| {
                spec.declared_ref
                    .clone()
                    .unwrap_or_else(|| spec.model_ref.to_string_lossy().into_owned())
            })
            .collect();
        crate::runtime::run_auto::startup_resolution(specs, &requested)
    };

    assert_eq!(resolution(&[]), StartupResolution::Done);
    assert_eq!(
        resolution(&[direct_gguf_startup_spec(&bare, None)]),
        StartupResolution::AnyName
    );
    assert_eq!(
        resolution(&[direct_gguf_startup_spec(&bare, Some("team/model"))]),
        StartupResolution::Names(vec!["team/model".into()])
    );
    let mut catalog = direct_gguf_startup_spec(std::path::Path::new("Qwen3-8B-Q4_K_M"), None);
    catalog.config_model_id = Some("qwen-small".into());
    assert_eq!(
        resolution(&[catalog.clone()]),
        StartupResolution::Names(vec!["Qwen3-8B-Q4_K_M".into(), "qwen-small".into()])
    );
    // One bare GGUF among named models still leaves every name possible.
    assert_eq!(
        resolution(&[catalog, direct_gguf_startup_spec(&bare, None)]),
        StartupResolution::AnyName
    );
}
