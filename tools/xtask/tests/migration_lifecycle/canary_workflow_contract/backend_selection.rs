//! Executes the select-canary-backend action step against stubbed platforms.
use super::{action_shell, document, field, named};
use std::fs;

#[test]
fn canary_backend_selection_requires_a_supported_platform_and_linux_architectures() {
    let action = document(".github/actions/select-canary-backend/action.yml");
    let (_, step) = named(action.get("runs").unwrap(), "name", "Detect stage backend");
    assert_eq!(field(step, "shell"), "bash");
    for (os, architectures, backend) in [
        ("Darwin", None, Some("metal")),
        ("Linux", Some("90"), Some("cuda")),
        ("Linux", Some(""), None),
        ("Linux", None, None),
        ("Windows", Some("90"), None),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().canonicalize().unwrap();
        let architectures = architectures.map_or_else(
            || "unset SKIPPY_CUDA_ARCHITECTURES".to_string(),
            |value| format!("export SKIPPY_CUDA_ARCHITECTURES='{value}'"),
        );
        let setup = format!(
            "export GITHUB_ENV=\"$PWD/action.env\"\n{architectures}\nuname() {{ [[ \"$*\" == -s ]] || return 91; printf '%s\\n' '{os}'; }}"
        );
        let (status, stdout, stderr) = action_shell(&directory, &setup, field(step, "run"));
        if let Some(backend) = backend {
            assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
            assert_eq!(
                fs::read_to_string(directory.join("action.env")).unwrap(),
                format!("LLAMA_STAGE_BACKEND={backend}\n")
            );
            assert!(
                String::from_utf8_lossy(&stdout)
                    .contains(&format!("selected stage backend: {backend}"))
            );
        } else {
            assert_eq!(status, 1, "{os}");
            assert!(!directory.join("action.env").exists());
            if os == "Linux" {
                assert!(
                    String::from_utf8_lossy(&stderr)
                        .contains("SKIPPY_CUDA_ARCHITECTURES is not set")
                );
            }
        }
    }
}
