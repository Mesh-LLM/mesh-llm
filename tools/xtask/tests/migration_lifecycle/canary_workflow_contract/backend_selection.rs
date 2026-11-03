use super::{action_shell, document, field, named};
use std::fs;

#[test]
fn linux_canary_backend_exports_cuda_and_refuses_missing_architectures() {
    let action = document(".github/actions/select-canary-backend/action.yml");
    let (_, step) = named(action.get("runs").unwrap(), "name", "Detect stage backend");
    assert_eq!(field(step, "shell"), "bash");
    for (architectures, succeeds) in [
        ("export SKIPPY_CUDA_ARCHITECTURES=90", true),
        ("unset SKIPPY_CUDA_ARCHITECTURES", false),
        ("export SKIPPY_CUDA_ARCHITECTURES=''", false),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().canonicalize().unwrap();
        let setup = format!(
            r#"
export GITHUB_ENV="$PWD/backend.env"
uname() {{ [[ "$*" == -s ]] || return 91; printf 'Linux\n'; }}
{architectures}
"#
        );
        let (status, stdout, stderr) = action_shell(&directory, &setup, field(step, "run"));
        assert_eq!(
            status == 0,
            succeeds,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
        let environment = directory.join("backend.env");
        if succeeds {
            assert_eq!(
                fs::read_to_string(environment).unwrap(),
                "LLAMA_STAGE_BACKEND=cuda\n"
            );
            assert!(String::from_utf8_lossy(&stdout).contains("selected stage backend: cuda"));
        } else {
            assert!(
                String::from_utf8_lossy(&stderr).contains("SKIPPY_CUDA_ARCHITECTURES is not set")
            );
            assert!(
                !environment.exists(),
                "failed selection must not export a backend"
            );
        }
    }
}
