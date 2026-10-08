//! Actual setup and verification calls from the OpenCode adapter, without a model.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

const SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/ci-opencode-smoke.sh"
));
const SOLUTION: &str = r#"use std::path::Path;
pub fn parse_codeword(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap().lines().find_map(|line| line.strip_prefix("CODEWORD=")).unwrap().to_owned()
}
pub fn prime_sum_from_matrix(path: &Path) -> i64 {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines().find_map(|line| line.strip_prefix("numbers:")).unwrap().split_whitespace()
        .map(|word| word.parse::<i64>().unwrap()).filter(|&n| n >= 2 && !(2..n).any(|d| n % d == 0)).sum()
}
"#;

fn tool(name: &str) -> PathBuf {
    let selected = std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
        .unwrap();
    if selected.is_absolute() {
        selected
    } else {
        std::env::current_dir().unwrap().join(selected)
    }
}

fn adapter_call(root: &Path, setup: bool) -> (bool, String, String) {
    let selector = SOURCE
        .split_once("# Frozen automation selection ends.")
        .unwrap()
        .0;
    let action = if setup {
        let line = SOURCE
            .lines()
            .find(|line| line.starts_with("INITIAL_IMPL_SHA="))
            .unwrap();
        format!("{line}\nprintf '%s\\n' \"$INITIAL_IMPL_SHA\"\n")
    } else {
        let start = SOURCE
            .find(
                "if ! \"${opencode_automation[@]}\" automation agent-fixture-inputs coding-verify",
            )
            .unwrap();
        let block = SOURCE[start..].split_once("\nfi").unwrap().0;
        format!("{block}\nfi\n")
    };
    let script = root.join("adapter.sh");
    fs::write(
        &script,
        format!("{selector}\nWORK_DIR=\"$OPENCODE_SMOKE_WORK_DIR\"\n{action}"),
    )
    .unwrap();
    let mut environment: BTreeMap<_, _> = [
        "PATH",
        "HOME",
        "USERPROFILE",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
        "CARGO_HOME",
    ]
    .into_iter()
    .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
    .collect();
    // Select absolute invocation paths while retaining rustup's multicall shim name.
    let compiler = tool("rustc");
    let just = tool("just");
    let path = std::env::join_paths([
        compiler.parent().unwrap().to_owned(),
        just.parent().unwrap().to_owned(),
        PathBuf::from("/usr/bin"),
        PathBuf::from("/bin"),
    ])
    .unwrap();
    environment.insert("PATH".into(), Value::Public(path));
    environment.insert(
        "OPENCODE_SMOKE_WORK_DIR".into(),
        Value::Public(root.join("exercise").into_os_string()),
    );
    environment.insert(
        "MESH_LLM_AUTOMATION_BIN".into(),
        Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
    );
    if !setup {
        environment.insert(
            "INITIAL_IMPL_SHA".into(),
            Value::Public(
                fs::read_to_string(root.join("initial"))
                    .unwrap()
                    .trim()
                    .into(),
            ),
        );
    }
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            arguments: vec![Value::Public(script.into_os_string())],
            cwd: root.to_owned(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(90),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 16384,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(16384),
            stderr: NonZeroUsize::new(16384),
        },
    )
    .unwrap();
    assert_eq!(report.process.outcome, Outcome::Exited);
    assert!(
        report.process.failure.is_none() && report.process.cleanup.complete,
        "{report:?}"
    );
    (
        report.process.status.unwrap().success(),
        String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes()).into_owned(),
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes()).into_owned(),
    )
}

fn exercise() -> tempfile::TempDir {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    fs::create_dir(root.join("scripts")).unwrap();
    fs::create_dir(root.join("exercise")).unwrap();
    let (ok, stdout, stderr) = adapter_call(&root, true);
    assert!(ok, "{stderr}");
    assert_eq!(stdout.trim().len(), 64);
    fs::write(root.join("initial"), stdout).unwrap();
    temporary
}

#[test]
fn actual_adapter_setup_and_verifier_accept_file_driven_solution_despite_changed_visible_tests() {
    let fixture = exercise();
    let root = fixture.path().canonicalize().unwrap();
    fs::write(root.join("exercise/src/smoke_calc.rs"), SOLUTION).unwrap();
    fs::write(root.join("exercise/tests/smoke_calc.rs"), "").unwrap();
    let (ok, _, stderr) = adapter_call(&root, false);
    assert!(ok, "{stderr}");
}

#[test]
fn actual_adapter_refuses_unchanged_source_and_visible_answer_constants() {
    let fixture = exercise();
    let root = fixture.path().canonicalize().unwrap();
    let (ok, _, stderr) = adapter_call(&root, false);
    assert!(
        !ok && stderr.contains("left src/smoke_calc.rs unchanged"),
        "{stderr}"
    );
    fs::write(root.join("exercise/src/smoke_calc.rs"), "use std::path::Path;\npub fn parse_codeword(_: &Path) -> String { \"signal-7429\".into() }\npub fn prime_sum_from_matrix(_: &Path) -> i64 { 10 }\n").unwrap();
    let (ok, _, stderr) = adapter_call(&root, false);
    assert!(
        !ok && stderr.contains("fixture implementation verification failed"),
        "{stderr}"
    );
}
