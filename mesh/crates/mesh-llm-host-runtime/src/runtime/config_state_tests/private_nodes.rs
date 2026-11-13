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

#[test]
fn a_cli_edit_waits_for_a_pending_apply_and_is_kept() {
    let dir = test_dir();
    let path = dir.join("config.toml");
    std::fs::write(
        &path,
        format!("[private]\nnodes = [\"{}\"]\n", "ab".repeat(32)),
    )
    .unwrap();
    let mut state = ConfigState::load(&path).expect("load");
    let mut unrelated = state.config().clone();
    unrelated.gpu.assignment = GpuAssignment::Pinned;
    let pending = match state.prepare_apply(unrelated, 0) {
        ConfigApplyPreparation::Pending(pending) => pending,
        other => panic!("expected a pending apply, got {other:?}"),
    };

    // `mesh-llm private remove` empties the list while the apply is pending.
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let cli_path = path.clone();
    let cli = std::thread::spawn(move || {
        mesh_llm_config::ConfigStore::open(&cli_path)
            .edit_preserving(|document| {
                document.remove("private");
                Ok(())
            })
            .expect("cli edit");
        done_tx.send(()).unwrap();
    });
    assert!(
        done_rx
            .recv_timeout(std::time::Duration::from_millis(300))
            .is_err(),
        "the edit must wait for the pending apply's write"
    );

    let persistence = pending.persist();
    assert!(matches!(
        state.finish_apply(*pending, persistence),
        ApplyResult::Applied { .. }
    ));
    cli.join().unwrap();

    let written = mesh_llm_config::load_config(Some(&path)).expect("written config");
    assert!(written.private.nodes.is_empty(), "{:?}", written.private);
    assert_eq!(written.gpu.assignment, GpuAssignment::Pinned);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_apply_is_refused_when_the_file_cannot_be_read() {
    let dir = test_dir();
    let path = dir.join("config.toml");
    std::fs::write(
        &path,
        format!("[private]\nnodes = [\"{}\"]\n", "ab".repeat(32)),
    )
    .unwrap();
    let mut state = ConfigState::load(&path).expect("load");
    let broken = "[private]\nnodes = [\"not-a-node\"]\n";
    std::fs::write(&path, broken).unwrap();

    let mut unrelated = state.config().clone();
    unrelated.gpu.assignment = GpuAssignment::Pinned;
    match state.apply_with_live_logging(unrelated, 0) {
        ApplyResult::PersistError(error) => {
            assert!(
                error.contains("cannot read the current config file"),
                "{error}"
            );
        }
        other => panic!("expected the apply to be refused, got {other:?}"),
    }
    assert_eq!(std::fs::read_to_string(&path).unwrap(), broken);
    assert_eq!(state.revision(), 0);

    std::fs::remove_dir_all(&dir).ok();
}
