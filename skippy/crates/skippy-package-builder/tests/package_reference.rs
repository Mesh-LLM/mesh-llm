use std::process::Command;
fn builder() -> &'static str {
    env!("CARGO_BIN_EXE_skippy-package-builder")
}
#[test]
fn actual_cli_preserves_revision_and_refuses_before_cache_preparation() {
    let scratch = tempfile::tempdir().unwrap();
    let cache = scratch.path().join("absent");
    for (input, expected) in [
        ("hf://org/repo", "org/repo\nmain\n"),
        ("hf://org/repo:topic/one", "org/repo\ntopic/one\n"),
        ("hf://org/repo@topic/one", "org/repo\ntopic/one\n"),
        ("hf://org/repo:release=v2", "org/repo\nrelease=v2\n"),
        ("hf://org/repo@release/étape", "org/repo\nrelease/étape\n"),
        ("hf://org/repo@release#v2", "org/repo\nrelease#v2\n"),
    ] {
        let out = Command::new(builder())
            .args(["parse-package-reference", input])
            .env("HF_HOME", &cache)
            .env("MESH_LLM_DATA_DIR", &cache)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(out.stdout, expected.as_bytes());
        assert!(!cache.exists());
    }
    for input in [
        "hf://org/repo:x@y",
        "hf://org/repo@../bad",
        "hf://org/repo@",
    ] {
        let out = Command::new(builder())
            .args(["parse-package-reference", input])
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
    }
    assert!(
        Command::new(builder())
            .args(["parse-package-reference", "--help"])
            .status()
            .unwrap()
            .success()
    );
}
