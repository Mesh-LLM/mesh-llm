//! Execute finite maintained classification fragments, excluding global /tmp writes.
use super::workflows::bash;

const SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.github/actions/compute-changes/derive-outputs.sh"
));

fn fragment(begin: &str, end: &str, output: &str) -> String {
    let (_, remainder) = SOURCE.split_once(begin).expect("classification start");
    let (body, _) = remainder.split_once(end).expect("classification end");
    format!("{begin}{body}\n{output}\n")
}

#[test]
fn backend_windows_and_sdk_classification_accepts_both_layouts() {
    let script = fragment(
        "BACKEND_CHANGED=\"false\"",
        "# Inference artifacts are needed",
        "printf '%s %s %s %s' \"$BACKEND_CHANGED\" \"$WINDOWS_CPU_BUILD_REQUIRED\" \"$WINDOWS_GPU_BUILD_REQUIRED\" \"$SDK_SMOKE_REQUIRED\"",
    );
    let root = tempfile::tempdir().unwrap();
    for (changed, expected) in [
        ("third_party/llama.cpp/upstream.txt", "true true true false"),
        ("skippy/llama_cpp/upstream.txt", "true true true false"),
        (
            "skippy/llama_cpp/patches/test.patch",
            "true true true false",
        ),
        ("sdk/node/index.js", "false false false true"),
        ("mesh/sdk/node/index.js", "false false false true"),
        ("skippy/scripts/build-llama.sh", "true false false true"),
        ("RESEARCH.md", "false false false false"),
    ] {
        let (ok, stdout, error) = bash(
            root.path(),
            &script,
            &[
                ("CHANGED_FILES", changed),
                ("ALL_RUST", "false"),
                ("FORCE_ALL", "false"),
                ("EVENT_NAME", "push"),
                ("AFFECTED_CRATES", "[]"),
                ("BACKEND_RECIPE_CHANGED", "false"),
            ],
        );
        assert!(ok, "{changed}: {error}");
        assert_eq!(stdout, expected, "{changed}");
    }
}

#[test]
fn relocated_runtime_owners_gate_sdk_and_inference_consumers() {
    let script = fragment(
        "SDK_SMOKE_REQUIRED=\"false\"",
        "LINUX_TEST_GROUPS_JSON",
        "printf '%s %s' \"$SDK_SMOKE_REQUIRED\" \"$INFERENCE_ARTIFACT_REQUIRED\"",
    );
    let root = tempfile::tempdir().unwrap();
    for (affected, expected) in [
        ("[\"mesh-llm-native-runtime\"]", "true true"),
        ("[\"skippy-native-runtime\"]", "true true"),
        ("[\"mesh-llm-config\"]", "true true"),
        ("[]", "false false"),
    ] {
        let (ok, stdout, error) = bash(
            root.path(),
            &script,
            &[
                (
                    "CHANGED_FILES",
                    "mesh/crates/skippy-native-runtime/src/lib.rs",
                ),
                ("ALL_RUST", "false"),
                ("FORCE_ALL", "false"),
                ("EVENT_NAME", "push"),
                ("UI_CHANGED", "false"),
                ("BACKEND_CHANGED", "false"),
                ("AFFECTED_CRATES", affected),
            ],
        );
        assert!(ok, "{affected}: {error}");
        assert_eq!(stdout, expected, "{affected}");
    }
}
