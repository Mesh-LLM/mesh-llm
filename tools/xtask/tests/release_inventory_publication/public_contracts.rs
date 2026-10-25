//! Finite actual public dispatch and native Just façade checks, with no GitHub request or build.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap, ffi::OsString, fs, num::NonZeroUsize, path::Path, time::Duration,
};
fn invoke(
    executable: &Path,
    cwd: &Path,
    args: &[&str],
    values: &[(&str, &str)],
) -> process::RawProcessReport {
    // These finite callees need executable discovery, not ambient credentials or shell state.
    let mut environment = ["PATH", "SystemRoot", "WINDIR", "TMPDIR", "TEMP", "TMP"]
        .into_iter()
        .filter_map(|key| {
            std::env::var_os(key).map(|value| (OsString::from(key), Value::Public(value)))
        })
        .collect::<BTreeMap<_, _>>();
    for (key, value) in values {
        environment.insert(OsString::from(key), Value::Public((*value).into()));
    }
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: executable.to_owned(),
            cwd: cwd.to_owned(),
            environment,
            arguments: args.iter().map(|s| Value::Public((*s).into())).collect(),
        },
        &Limits {
            execution: Duration::from_secs(20),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(4 * 1024 * 1024),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(
        report.process.cleanup.complete && report.process.failure.is_none(),
        "{:?}",
        report.process
    );
    report
}
fn xtask(cwd: &Path, args: &[&str]) -> process::RawProcessReport {
    invoke(Path::new(env!("CARGO_BIN_EXE_xtask")), cwd, args, &[])
}
#[test]
fn release_inventory_publication_public_exact_help_before_git_or_gh() {
    let root = tempfile::tempdir().unwrap();
    // Not a Git repository: successful exact command help cannot depend on provenance or gh.
    for option in ["--help", "-h"] {
        let report = xtask(root.path(), &["release", "inventory", option]);
        assert_eq!(report.process.status.unwrap().code(), Some(0));
        assert!(report.stderr.unwrap().as_bytes().is_empty());
        let stdout = String::from_utf8(report.stdout.unwrap().as_bytes().to_vec()).unwrap();
        for option in [
            "release inventory",
            "--repo",
            "--head",
            "--release-tag",
            "--output",
        ] {
            assert!(stdout.contains(option), "{stdout}");
        }
    }
    let report = xtask(root.path(), &["--help"]);
    assert_eq!(report.process.status.unwrap().code(), Some(0));
    assert!(
        String::from_utf8_lossy(report.stdout.unwrap().as_bytes())
            .contains("cargo xtool release inventory")
    );
}
#[test]
fn release_inventory_publication_public_argument_refusal_preserves_output() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("existing report.json");
    fs::write(&destination, b"prior report").unwrap();
    for tail in [
        vec!["--unknown"],
        vec!["--head"],
        vec!["--repo="],
        vec!["--help", "--head", "HEAD"],
    ] {
        let mut args = vec![
            "release",
            "inventory",
            "--output",
            destination.to_str().unwrap(),
        ];
        args.extend(tail);
        let report = xtask(root.path(), &args);
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        assert!(String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).starts_with("error:"));
        assert_eq!(fs::read(&destination).unwrap(), b"prior report");
    }
}
#[cfg(unix)]
fn native_just() -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    std::env::split_paths(&std::env::var_os("PATH").expect("native Just requires PATH"))
        .map(|p| p.join("just"))
        .find(|p| p.is_file() && fs::metadata(p).unwrap().permissions().mode() & 0o111 != 0)
        .expect("native Just is required for the Unix façade")
        .canonicalize()
        .unwrap()
}
#[cfg(unix)]
#[test]
fn release_inventory_publication_native_just_positional_facade_preserves_argv_and_failure() {
    // Native Just owns recipe parsing. A copied finite callee records argv instead of building or collecting.
    let executable = native_just();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let shown = invoke(
        &executable,
        &repository,
        &["--show", "release-inventory"],
        &[],
    );
    assert_eq!(
        shown.process.status.unwrap().code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(shown.stderr.unwrap().as_bytes())
    );
    let recipe = String::from_utf8(shown.stdout.unwrap().as_bytes().to_vec()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();
    fs::write(root_path.join("Justfile"), format!("{recipe}\n[positional-arguments]\nautomation-run *ARGS:\n    #!/usr/bin/env bash\n    set -euo pipefail\n    printf '%s\\0' \"$@\" > \"$FIXTURE_ARGV\"\n    exit \"$FIXTURE_STATUS\"\n")).unwrap();
    let record = root_path.join("record");
    let output = "evidence directory/quote'\";$(touch forbidden).json";
    let head = "candidate with spaces;$(touch forbidden)";
    let expected = [
        "release",
        "inventory",
        "--repo",
        "Owner/Name",
        "--head",
        head,
        "--output",
        output,
    ];
    for status in ["0", "7"] {
        let report = invoke(
            &executable,
            &root_path,
            &[
                "release-inventory",
                "--repo",
                "Owner/Name",
                "--head",
                head,
                "--output",
                output,
            ],
            &[
                ("FIXTURE_ARGV", record.to_str().unwrap()),
                ("FIXTURE_STATUS", status),
            ],
        );
        assert_eq!(report.process.status.unwrap().success(), status == "0");
        let bytes = fs::read(&record).unwrap();
        let actual = bytes
            .strip_suffix(&[0])
            .unwrap()
            .split(|b| *b == 0)
            .map(|b| std::str::from_utf8(b).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(!root_path.join("forbidden").exists());
    }
}
