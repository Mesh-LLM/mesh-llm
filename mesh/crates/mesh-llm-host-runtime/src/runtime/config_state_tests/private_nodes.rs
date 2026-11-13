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

#[test]
fn an_apply_from_a_stale_snapshot_keeps_the_files_private_list() {
    let dir = test_dir();
    let path = dir.join("config.toml");
    std::fs::write(
        &path,
        format!("[private]\nnodes = [\"{}\"]\n", "ab".repeat(32)),
    )
    .unwrap();
    let mut state = ConfigState::load(&path).expect("load");
    let stale = state.config().clone();
    // `mesh-llm private remove` empties the list behind the state's back.
    std::fs::write(&path, "[private]\nnodes = []\n").unwrap();

    let mut unrelated = stale;
    unrelated.gpu.assignment = GpuAssignment::Pinned;
    assert!(matches!(
        state.apply_with_live_logging(unrelated, 0),
        ApplyResult::Applied { .. }
    ));

    let written = mesh_llm_config::load_config(Some(&path)).expect("written config");
    assert!(written.private.nodes.is_empty(), "{:?}", written.private);
    assert_eq!(written.gpu.assignment, GpuAssignment::Pinned);

    std::fs::remove_dir_all(&dir).ok();
}
