//! Actual Just recipe execution with a finite bootstrap boundary, never Cargo.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn installed_just() -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").expect("Just fixture requires PATH"))
        .map(|directory| directory.join("just"))
        .find(|file| {
            file.is_file() && fs::metadata(file).unwrap().permissions().mode() & 0o111 != 0
        })
        .expect("Just must be installed for actual recipe execution")
        .canonicalize()
        .unwrap()
}
struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    just: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("recipe root with spaces");
        fs::create_dir_all(root.join("bin")).unwrap();
        let source =
            fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../just/ci.just"))
                .unwrap();
        let start = source
            .find("[private]\n[unix]\n[positional-arguments]\nautomation-run *ARGS:\n")
            .expect("recipe-local positional arguments");
        let recipe = source[start..].split("\n\n").next().unwrap();
        fs::write(root.join("Justfile"), format!("{recipe}\n")).unwrap();
        executable(
            &root.join("bin/just"),
            r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 1 && "$1" == automation-bootstrap ]] || exit 95
printf 'bootstrap\n' >> "$FIXTURE_ROOT/bootstrap.calls"
[[ "$BOOTSTRAP_STATUS" == 0 ]] || exit "$BOOTSTRAP_STATUS"
printf 'binary_path=%s\n' "$FIXTURE_ROOT/selected binary"
"#,
        );
        executable(
            &root.join("selected binary"),
            r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$#" > "$FIXTURE_ROOT/argc"
for arg in "$@"; do printf '%s\0' "$arg"; done > "$FIXTURE_ROOT/argv"
exit "$BINARY_STATUS"
"#,
        );
        fs::write(root.join("glob-match.txt"), b"must not glob").unwrap();
        Self {
            _temporary: temporary,
            root,
            just: installed_just(),
        }
    }
    fn run(&self, arguments: &[&str], bootstrap: &str, status: &str) -> process::RawProcessReport {
        let environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            ("HOME", self.root.display().to_string()),
            ("FIXTURE_ROOT", self.root.display().to_string()),
            ("BOOTSTRAP_STATUS", bootstrap.into()),
            ("BINARY_STATUS", status.into()),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value.into())))
        .collect();
        let mut argv = vec![
            "--justfile".into(),
            self.root.join("Justfile").into_os_string(),
            "automation-run".into(),
        ];
        argv.extend(arguments.iter().map(std::ffi::OsString::from));
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: self.just.clone(),
                cwd: self.root.clone(),
                environment,
                arguments: argv.into_iter().map(Value::Public).collect(),
            },
            &Limits {
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(report.process.failure.is_none(), "{:?}", report.process);
        assert!(report.process.cleanup.complete, "{:?}", report.process);
        report
    }
}
#[test]
fn actual_just_automation_recipe_preserves_empty_and_shell_sensitive_arguments() {
    let fixture = Fixture::new();
    let arguments = [
        "automation",
        "",
        "a b",
        "*.txt",
        "$(touch substitution-ran)",
        "; touch semicolon-ran",
        "single'and\"double",
        "line1\nline2",
    ];
    let report = fixture.run(&arguments, "0", "23");
    assert_eq!(report.process.status.unwrap().code(), Some(23));
    assert_eq!(
        fs::read_to_string(fixture.root.join("argc")).unwrap(),
        format!("{}\n", arguments.len())
    );
    let expected: Vec<u8> = arguments
        .iter()
        .flat_map(|arg| arg.as_bytes().iter().copied().chain(std::iter::once(0)))
        .collect();
    assert_eq!(fs::read(fixture.root.join("argv")).unwrap(), expected);
    assert_eq!(
        fs::read_to_string(fixture.root.join("bootstrap.calls")).unwrap(),
        "bootstrap\n"
    );
    for absent in ["substitution-ran", "semicolon-ran"] {
        assert!(!fixture.root.join(absent).exists());
    }
}
#[test]
fn actual_just_automation_recipe_accepts_no_args_and_blocks_binary_on_bootstrap_failure() {
    let empty = Fixture::new();
    assert!(empty.run(&[], "0", "0").process.success());
    assert_eq!(fs::read(empty.root.join("argv")).unwrap(), b"");
    assert_eq!(fs::read_to_string(empty.root.join("argc")).unwrap(), "0\n");
    let failed = Fixture::new();
    let report = failed.run(&["ignored"], "17", "0");
    assert_eq!(report.process.status.unwrap().code(), Some(17));
    assert!(!failed.root.join("argv").exists());
    assert_eq!(
        fs::read_to_string(failed.root.join("bootstrap.calls")).unwrap(),
        "bootstrap\n"
    );
}
