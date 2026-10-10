//! Actual recovery command and failure caller preserve diagnostics without publication.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::Value as Json;
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

fn execute(
    executable: &Path,
    args: &[String],
    cwd: &Path,
    env: &[(&str, String)],
) -> process::RawProcessReport {
    let environment: BTreeMap<_, _> = [("PATH", "/usr/bin:/bin".to_owned())]
        .into_iter()
        .chain(env.iter().cloned())
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
    process::supervise_raw(
        &ProcessSpec {
            executable: executable.into(),
            cwd: cwd.into(),
            environment,
            arguments: args.iter().map(|a| Value::Public(a.into())).collect(),
        },
        &Limits {
            execution: Duration::from_secs(20),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}
fn git(root: &Path, args: &[&str]) -> String {
    let report = execute(
        Path::new("/usr/bin/git"),
        &args.iter().map(|s| (*s).into()).collect::<Vec<_>>(),
        root,
        &[],
    );
    assert!(report.process.success(), "{:?}", report.stderr);
    std::str::from_utf8(report.stdout.unwrap().as_bytes())
        .unwrap()
        .trim()
        .into()
}
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    base: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap().join("source");
        fs::create_dir(&root).unwrap();
        git(&root, &["init", "-q"]);
        fs::write(root.join("tracked"), b"base").unwrap();
        git(&root, &["add", "."]);
        git(
            &root,
            &[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "base",
            ],
        );
        let base = git(&root, &["rev-parse", "HEAD"]);
        fs::write(root.join("tracked"), b"repair").unwrap();
        Self {
            _temp: temp,
            root,
            base,
        }
    }
    fn native(&self, base: &str) -> process::RawProcessReport {
        execute(
            Path::new(env!("CARGO_BIN_EXE_xtask")),
            &[
                "automation".into(),
                "canary-receipts".into(),
                "recover-source".into(),
                "--root".into(),
                self.root.display().to_string(),
                "--output".into(),
                self.root.join("recovery").display().to_string(),
                "--base".into(),
                base.into(),
            ],
            &self.root,
            &[],
        )
    }
}
#[test]
fn actual_recovery_cli_records_failed_source_and_refuses_bad_base_without_complete_output() {
    let fixture = Fixture::new();
    let report = fixture.native(&fixture.base);
    assert!(report.process.success(), "{:?}", report.stderr);
    assert!(report.stdout.unwrap().as_bytes().is_empty());
    let manifest: Json =
        serde_json::from_slice(&fs::read(fixture.root.join("recovery/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["verified"], false);
    assert_eq!(manifest["base"], fixture.base);
    let fixture = Fixture::new();
    let report = fixture.native("HEAD");
    assert!(!report.process.success());
    assert!(!fixture.root.join("recovery").exists());
}
fn caller_source() -> String {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/llama-canary-agent-repair.sh"),
    )
    .unwrap();
    let admission_start = source
        .find("if [[ \"$HARNESS_MODE\" == repair* ]]; then\n  repair_recovery_controller=")
        .unwrap();
    let admission_end = source[admission_start..].find("\nfi\n").unwrap() + admission_start + 4;
    let failure_start = source
        .find("  if repair_candidate_until_green; then")
        .unwrap();
    let failure_end = source[failure_start..]
        .find("\n  snapshot_candidate_tree")
        .unwrap()
        + failure_start;
    format!(
        "{}\n{}",
        &source[admission_start..admission_end],
        &source[failure_start..failure_end]
    )
}
#[test]
fn actual_failed_repair_caller_uses_frozen_native_controller_and_preserves_original_failure() {
    for mode in ["repair", "repair-build"] {
        let fixture = Fixture::new();
        let controller = fixture.root.parent().unwrap().join("trusted-controller");
        fs::write(&controller, "#!/bin/sh\nexec \"$REAL_XTASK\" \"$@\"\n").unwrap();
        fs::set_permissions(&controller, fs::Permissions::from_mode(0o755)).unwrap();
        let body = format!(
            "set -euo pipefail\nrepair_candidate_until_green() {{ return 43; }}\n{}",
            caller_source()
        );
        let report = execute(
            Path::new("/bin/bash"),
            &["-c".into(), body],
            &fixture.root,
            &[
                ("HARNESS_MODE", mode.into()),
                ("MESH_LLM_AUTOMATION_BIN", controller.display().to_string()),
                ("REAL_XTASK", env!("CARGO_BIN_EXE_xtask").into()),
                ("ROOT", fixture.root.display().to_string()),
                ("STATE_DIR", fixture.root.display().to_string()),
                ("BASE_HEAD", fixture.base.clone()),
            ],
        );
        assert_eq!(report.process.status.unwrap().code(), Some(43));
        assert!(report.process.cleanup.complete);
        let manifest: Json =
            serde_json::from_slice(&fs::read(fixture.root.join("recovery/manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["verified"], false);
        assert!(!fixture.root.join("candidate.bundle").exists());
    }
}
#[test]
fn actual_failed_repair_caller_refuses_changed_controller_or_capture_error_without_masking_failure()
{
    for mutation in ["controller", "base"] {
        let fixture = Fixture::new();
        let controller = fixture.root.parent().unwrap().join("trusted-controller");
        fs::write(&controller, "#!/bin/sh\nexec \"$REAL_XTASK\" \"$@\"\n").unwrap();
        fs::set_permissions(&controller, fs::Permissions::from_mode(0o755)).unwrap();
        let change = if mutation == "controller" {
            "printf '# changed after admission\\n' >> \"$MESH_LLM_AUTOMATION_BIN\";"
        } else {
            ""
        };
        let body = format!(
            "set -euo pipefail\nrepair_candidate_until_green() {{ {change} return 43; }}\n{}",
            caller_source()
        );
        let base = if mutation == "base" {
            "0000000000000000000000000000000000000000".into()
        } else {
            fixture.base.clone()
        };
        let report = execute(
            Path::new("/bin/bash"),
            &["-c".into(), body],
            &fixture.root,
            &[
                ("HARNESS_MODE", "repair-build".into()),
                ("MESH_LLM_AUTOMATION_BIN", controller.display().to_string()),
                ("REAL_XTASK", env!("CARGO_BIN_EXE_xtask").into()),
                ("ROOT", fixture.root.display().to_string()),
                ("STATE_DIR", fixture.root.display().to_string()),
                ("BASE_HEAD", base),
            ],
        );
        assert_eq!(report.process.status.unwrap().code(), Some(43));
        assert!(report.process.cleanup.complete);
        assert!(!fixture.root.join("recovery").exists());
        assert!(
            String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
                .contains("could not capture the unverified repair source")
        );
    }
}
