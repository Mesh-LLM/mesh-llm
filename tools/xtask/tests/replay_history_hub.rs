#[cfg(unix)]
mod unix {
    use std::{
        path::{Path, PathBuf},
        process::{Command, Output},
    };
    fn executable(parent: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let file = parent.join(name);
        std::fs::write(&file, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o700)).unwrap();
        file
    }
    fn invoke(arguments: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix"])
            .args(arguments)
            .env("HF_TOKEN", "fixture-parent-secret")
            .env("HUGGING_FACE_HUB_TOKEN", "fixture-cached-alias")
            .output()
            .unwrap()
    }
    #[test]
    fn fetch_cli_404_bootstraps_anonymously_despite_ambient_tokens() {
        let parent = tempfile::tempdir().unwrap();
        let hf = executable(
            parent.path(),
            "hf",
            "test -z \"${HF_TOKEN:-}\"; test -z \"${HUGGING_FACE_HUB_TOKEN:-}\"; test \"$HF_HUB_DISABLE_IMPLICIT_TOKEN\" = 1; exit 1",
        );
        let curl = executable(
            parent.path(),
            "curl",
            "test -z \"${HF_TOKEN:-}\"; test -z \"${HUGGING_FACE_HUB_TOKEN:-}\"; printf 404",
        );
        let output = parent.path().join("history");
        let result = invoke(&[
            "history-fetch",
            "--dataset-repo",
            "owner/dataset",
            "--output",
            output.to_str().unwrap(),
            "--hf",
            hf.to_str().unwrap(),
            "--curl",
            curl.to_str().unwrap(),
            "--timeout",
            "2",
        ]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!output.exists());
        assert!(String::from_utf8_lossy(&result.stdout).contains("verified HTTP 404"));
        assert!(!String::from_utf8_lossy(&result.stdout).contains("fixture-parent-secret"));
    }
    #[test]
    fn fetch_cli_auth_error_and_failed_transport_never_fail_open() {
        let parent = tempfile::tempdir().unwrap();
        let hf = executable(parent.path(), "hf", "exit 1");
        for (index, body) in [
            "printf 401",
            "printf 403",
            "printf 500",
            "printf 404; exit 28",
        ]
        .iter()
        .enumerate()
        {
            let curl = executable(parent.path(), &format!("curl-{index}"), body);
            let output = parent.path().join(format!("history-{index}"));
            let result = invoke(&[
                "history-fetch",
                "--dataset-repo",
                "owner/dataset",
                "--output",
                output.to_str().unwrap(),
                "--hf",
                hf.to_str().unwrap(),
                "--curl",
                curl.to_str().unwrap(),
                "--timeout",
                "2",
            ]);
            assert!(!result.status.success());
            assert!(!output.exists());
        }
    }
    #[test]
    fn upload_cli_uses_only_explicit_publisher_token_and_fixed_card_source() {
        let parent = tempfile::tempdir().unwrap();
        let output = parent.path().join("output");
        std::fs::create_dir(&output).unwrap();
        std::fs::write(output.join("card.md"), "# fixture\n").unwrap();
        let hf = executable(
            parent.path(),
            "hf",
            "test \"$HF_TOKEN\" = fixture-parent-secret; test -z \"${HUGGING_FACE_HUB_TOKEN:-}\"; test \"$1\" = upload; test \"$4\" = card.md; test \"$8\" = 'card refresh 2026-10-02'; case \"$*\" in *fixture-parent-secret*) exit 8;; esac",
        );
        let result = invoke(&[
            "history-upload",
            "--dataset-repo",
            "owner/dataset",
            "--output-dir",
            output.to_str().unwrap(),
            "--kind",
            "card",
            "--run-date",
            "2026-10-02",
            "--hf",
            hf.to_str().unwrap(),
            "--timeout",
            "2",
        ]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!String::from_utf8_lossy(&result.stdout).contains("fixture-parent-secret"));
    }
}
