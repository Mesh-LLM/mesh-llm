use super::package_metrics;

#[test]
fn package_metrics_counts_direct_cross_target_and_hyphenated_build_files() {
    let temporary = tempfile::tempdir().unwrap();
    for (relative, count) in [
        ("debug/deps/mesh_llm-abc.rcgu.o", 256),
        (
            "x86_64-unknown-linux-gnu/debug/deps/mesh_llm-abc.rcgu.o",
            384,
        ),
        ("debug/build/mesh-llm-abc/output", 512),
    ] {
        let path = temporary.path().join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![b'x'; count]).unwrap();
    }
    let metrics = package_metrics(temporary.path(), &["mesh-llm".to_owned()]);
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].package, "mesh-llm");
    assert_eq!(metrics[0].bytes, 1152);
}
