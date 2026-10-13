//! Execute only the maintained notes-base metadata tail with finite local inputs.
use super::{document, job, named, support, text};
use crate::process;
use std::{collections::BTreeMap, fs, num::NonZeroUsize, time::Duration};

const TAGS: &str = "v0.75.1\nv0.76.0-rc.1\nv0.76.0\n";
const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const ADAPTERS: &str = r#"
set -euo pipefail
git() {
  [[ $# == 3 && "$1" == tag && "$2" == --list && "$3" == 'v*' ]] || return 91
  if [[ "$TAG_STATUS" != 0 ]]; then return "$TAG_STATUS"; fi
  printf '%s' "$FIXTURE_TAGS"
}
cargo() {
  [[ $# == 4 && "$1" == xtool && "$2" == release && "$3" == notes-base ]] || return 92
  printf '%s\n' "$@" >> "$NATIVE_CALLS"
  "$XTASK" "${@:2}"
}
"#;

fn tail() -> String {
    let doc = document("release.yml");
    let metadata = job(&doc, "metadata");
    let source = named(metadata, "Prepare canonical release source");
    assert_eq!(text(source, "id"), Some("source"));
    assert_eq!(
        text(metadata.get("outputs").unwrap(), "release_notes_base"),
        Some("${{ steps.source.outputs.release_notes_base }}")
    );
    let run = text(source, "run").unwrap();
    let start = run.find("release_notes_base=\"$(").unwrap();
    format!("{ADAPTERS}\n{}", &run[start..])
}

fn run(
    f: &support::Fixture,
    script: &str,
    target: &str,
    tag_status: &str,
) -> process::RawProcessReport {
    let mut environment = BTreeMap::new();
    for (key, value) in [
        ("PATH", std::ffi::OsString::from("/usr/bin:/bin")),
        ("HOME", f.path().join("home").into_os_string()),
        ("GIT_MASTER", "1".into()),
        ("GIT_OPTIONAL_LOCKS", "0".into()),
        ("RELEASE_TAG", target.into()),
        ("FIXTURE_TAGS", TAGS.into()),
        ("TAG_STATUS", tag_status.into()),
        ("source_sha", SHA.into()),
        ("XTASK", env!("CARGO_BIN_EXE_xtask").into()),
        (
            "NATIVE_CALLS",
            f.path().join("native-calls").into_os_string(),
        ),
        (
            "GITHUB_OUTPUT",
            f.path().join("github-output").into_os_string(),
        ),
    ] {
        environment.insert(key.into(), process::Value::Public(value));
    }
    process::supervise_raw(
        &process::ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: f.path().to_owned(),
            arguments: vec![
                process::Value::Public("-c".into()),
                process::Value::Public(script.into()),
            ],
            environment,
        },
        &process::Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 64 * 1024,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(64 * 1024),
            stderr: NonZeroUsize::new(64 * 1024),
        },
    )
    .unwrap()
}

fn clean(raw: &process::RawProcessReport) {
    let report = &raw.process;
    assert_eq!(report.outcome, process::Outcome::Exited, "{report:?}");
    assert!(
        report.failure.is_none() && report.cleanup.failure.is_none(),
        "{report:?}"
    );
    assert!(
        report.cleanup.complete && !report.cleanup.forced && !report.cleanup.graceful_signal_failed,
        "{report:?}"
    );
    for (bytes, stream) in [(&raw.stdout, &report.stdout), (&raw.stderr, &report.stderr)] {
        assert_eq!(
            bytes.as_ref().unwrap().as_bytes().len() as u64,
            stream.bytes_seen
        );
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(stream.oversized_lines, 0);
        assert_eq!(stream.suppressed_lines, 0);
    }
}

#[test]
fn graph_release_notes_base_actual_metadata_slice_publishes_selection_and_refuses_failure() {
    let script = tail();
    for (target, tag_status, expected) in [
        ("v0.76.1", "0", Some("v0.76.0")),
        ("v0.76.1-rc.1", "0", Some("v0.76.0")),
        ("v0.75.1", "0", Some("")),
        ("invalid-target", "0", None),
        ("v0.76.1", "42", None),
    ] {
        let f = support::Fixture::new();
        let raw = run(&f, &script, target, tag_status);
        clean(&raw);
        assert_eq!(
            fs::read_to_string(f.path().join("native-calls")).unwrap(),
            format!("xtool\nrelease\nnotes-base\n{target}\n")
        );
        let output = f.path().join("github-output");
        if let Some(base) = expected {
            assert!(raw.process.status.unwrap().success(), "{:?}", raw.process);
            assert_eq!(
                fs::read(&output).unwrap(),
                format!("sha={SHA}\nrelease_notes_base={base}\n").as_bytes()
            );
        } else {
            assert!(!raw.process.status.unwrap().success());
            assert!(
                !output.exists(),
                "refused selection must not publish metadata"
            );
            if tag_status == "42" {
                assert_eq!(raw.process.status.unwrap().code(), Some(42));
            }
        }
        f.0.close().unwrap();
    }
}
