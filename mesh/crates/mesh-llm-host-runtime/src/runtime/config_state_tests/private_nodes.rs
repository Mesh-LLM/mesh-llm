use super::*;

#[test]
fn a_private_node_list_change_applies_live_and_is_written() {
    let dir = test_dir();
    let path = dir.join("config.toml");
    let mut state = ConfigState::load(&path).expect("load");
    let mut config = state.config().clone();
    config.private.nodes = vec!["ab".repeat(32)];

    match state.apply_with_live_logging(config, 0) {
        ApplyResult::Applied { apply_mode, .. } => {
            assert_eq!(apply_mode, ConfigApplyMode::Live);
        }
        other => panic!("expected a live apply, got {other:?}"),
    }
    let written = mesh_llm_config::load_config(Some(&path)).expect("written config");
    assert_eq!(written.private.nodes, vec!["ab".repeat(32)]);

    std::fs::remove_dir_all(&dir).ok();
}
