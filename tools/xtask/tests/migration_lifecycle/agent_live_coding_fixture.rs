//! Actual shared helper exercise and guarded verification with isolated client HOME.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap, fs, num::NonZeroUsize, os::unix::fs::PermissionsExt, path::PathBuf,
    time::Duration,
};
const SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/ci-agent-live-fixture-lib.sh"
));
const SOLUTION: &str = r#"use std::path::Path;
pub fn parse_codeword(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap().lines().find_map(|line| line.strip_prefix("CODEWORD=")).unwrap().to_owned()
}
pub fn prime_sum_from_matrix(path: &Path) -> i64 {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines().find_map(|line| line.strip_prefix("numbers:")).unwrap().split_whitespace().map(|word| word.parse::<i64>().unwrap())
        .filter(|&n| n >= 2 && !(2..n).any(|d| n % d == 0)).sum()
}
"#;
const ANSWER: &str = "{\"toolName\":\"read\"}\n{\"toolName\":\"write\"}\n{\"toolName\":\"bash\"}\nCODEWORD=signal-7429\nCHECKSUM=FS-319-DELTA\nPRIME_SUM=10\nQUESTION=facts/signal.md\n";

struct Fixture {
    directory: tempfile::TempDir,
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        for path in ["scripts", "bin", "client-home", "exercise"] {
            fs::create_dir(root.join(path)).unwrap();
        }
        fs::write(root.join("scripts/library.sh"), SOURCE).unwrap();
        let compiler = std::env::split_paths(&std::env::var_os("PATH").unwrap())
            .map(|directory| directory.join("rustc"))
            .find(|path| path.is_file())
            .unwrap();
        let compiler = if compiler.is_absolute() {
            compiler
        } else {
            std::env::current_dir().unwrap().join(compiler)
        };
        let quoted = format!("'{}'", compiler.to_str().unwrap().replace('\'', "'\\''"));
        let shim = root.join("bin/rustc");
        let quote = |value: &str| format!("'{}'", value.replace('\'', "'\\''"));
        let home = quote(std::env::var("HOME").unwrap().as_str());
        let trace = quote(root.join("trace").to_str().unwrap());
        fs::write(&shim, format!("#!/bin/sh\n[ \"$HOME\" = {home} ] || exit 94\nprintf 'compiler-home-ok\\n' >> {trace}\nexec {quoted} \"$@\"\n")).unwrap();
        fs::set_permissions(shim, fs::Permissions::from_mode(0o700)).unwrap();
        let tripwire = root.join("bin/python3");
        fs::write(
            &tripwire,
            "#!/bin/sh\nprintf 'unexpected Python\\n' >> \"$TRACE\"\nexit 92\n",
        )
        .unwrap();
        fs::set_permissions(tripwire, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(root.join("answers.jsonl"), ANSWER).unwrap();
        let fixture = Self { directory, root };
        let (ok, output, stderr) = fixture.run("setup", true, "Pi");
        assert!(ok, "{stderr}");
        assert_eq!(output.trim().len(), 64);
        fs::write(fixture.root.join("initial"), output.trim()).unwrap();
        fixture
    }

    fn run(&self, action: &str, isolate_home: bool, label: &str) -> (bool, String, String) {
        let command = if action == "setup" {
            "agent_smoke_write_fixture \"$WORK_DIR\""
        } else {
            "if agent_smoke_validate_fixture \"$WORK_DIR\" \"$INITIAL\" \"$EVIDENCE\" \"$LABEL\" true; then printf 'accepted\\n'; else exit 17; fi"
        };
        let script = self.root.join("scripts/call.sh");
        fs::write(
            &script,
            format!(
                "set -euo pipefail\nsource \"$LIBRARY\"\n{}\n{command}\n",
                if isolate_home {
                    "export HOME=\"$CLIENT_HOME\""
                } else {
                    ""
                }
            ),
        )
        .unwrap();
        let mut paths = vec![self.root.join("bin")];
        paths.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap()));
        let home = std::env::var_os("HOME").unwrap();
        let mut environment = BTreeMap::from([
            (
                "PATH".into(),
                Value::Public(std::env::join_paths(paths).unwrap()),
            ),
            ("HOME".into(), Value::Public(home.clone())),
            ("TRUSTED_AUTOMATION_HOME".into(), Value::Public(home)),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
            (
                "WORK_DIR".into(),
                Value::Public(self.root.join("exercise").into_os_string()),
            ),
            (
                "CLIENT_HOME".into(),
                Value::Public(self.root.join("client-home").into_os_string()),
            ),
            (
                "LIBRARY".into(),
                Value::Public(self.root.join("scripts/library.sh").into_os_string()),
            ),
            (
                "TRACE".into(),
                Value::Public(self.root.join("trace").into_os_string()),
            ),
            (
                "EVIDENCE".into(),
                Value::Public(self.root.join("answers.jsonl").into_os_string()),
            ),
            ("LABEL".into(), Value::Public(label.into())),
            (
                "INITIAL".into(),
                Value::Public(
                    fs::read_to_string(self.root.join("initial"))
                        .unwrap_or_default()
                        .into(),
                ),
            ),
        ]);
        for key in ["RUSTUP_HOME", "CARGO_HOME", "RUSTUP_TOOLCHAIN"] {
            if let Some(value) = std::env::var_os(key) {
                environment.insert(key.into(), Value::Public(value));
            }
        }
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                arguments: vec![Value::Public(script.into_os_string())],
                cwd: self.root.clone(),
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
            report.process.success(),
            String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes()).into_owned(),
            String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes()).into_owned(),
        )
    }

    fn implementation(&self, source: &str) {
        fs::write(self.root.join("exercise/src/smoke_calc.rs"), source).unwrap();
    }
    fn assert_compiler_and_no_python(&self) {
        assert_eq!(
            fs::read_to_string(self.root.join("trace")).unwrap(),
            "compiler-home-ok\n"
        );
    }
}

#[test]
fn actual_helper_preserves_file_driven_goose_and_isolated_pi_verification() {
    for (label, isolated) in [("Goose", false), ("Pi", true)] {
        let fixture = Fixture::new();
        fixture.implementation(SOLUTION);
        let (ok, stdout, stderr) = fixture.run("verify", isolated, label);
        assert!(ok && stdout.contains("accepted"), "{stderr}");
        fixture.assert_compiler_and_no_python();
        assert!(
            fixture
                .directory
                .path()
                .join("exercise/tests/smoke_calc.rs")
                .exists()
        );
    }
}

#[test]
fn guarded_shared_helper_refuses_unchanged_source_and_constant_visible_answers_before_answer_validation()
 {
    let fixture = Fixture::new();
    let (ok, stdout, _) = fixture.run("verify", true, "Pi");
    assert!(!ok && !stdout.contains("accepted"));
    fixture.implementation("use std::path::Path;\npub fn parse_codeword(_: &Path) -> String { \"signal-7429\".into() }\npub fn prime_sum_from_matrix(_: &Path) -> i64 { 10 }\n");
    let (ok, stdout, stderr) = fixture.run("verify", true, "Pi");
    assert!(
        !ok && !stdout.contains("accepted")
            && stderr.contains("fixture implementation verification failed")
    );
    fixture.assert_compiler_and_no_python();
}

#[test]
fn actual_shared_helper_retains_answer_and_coding_tool_event_requirements() {
    for evidence in [
        ANSWER.replace("{\"toolName\":\"write\"}\n", ""),
        ANSWER.replace("CHECKSUM=FS-319-DELTA", "CHECKSUM=wrong"),
    ] {
        let fixture = Fixture::new();
        fixture.implementation(SOLUTION);
        fs::write(fixture.root.join("answers.jsonl"), evidence).unwrap();
        let (ok, stdout, _) = fixture.run("verify", true, "Pi");
        assert!(!ok && !stdout.contains("accepted"));
        fixture.assert_compiler_and_no_python();
    }
}
