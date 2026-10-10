//! Actual copied release policy with finite private commands; no release or Git operation.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
const HEAD: &str = "0123456789abcdef0123456789abcdef01234567";
const START: &str = "2026-08-14T12:00:00Z";
const VERSION: &str = "0.1.1-rc1";
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn tool(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|path| path.join(name))
        .find(|path| {
            path.is_file() && fs::metadata(path).unwrap().permissions().mode() & 0o111 != 0
        })
        .unwrap_or_else(|| panic!("finite release caller proof requires component {name}"))
        .canonicalize()
        .unwrap()
}
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
}
struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .canonicalize()
            .unwrap()
            .join("release fixture with spaces");
        for directory in ["bin", "scripts", "scratch"] {
            fs::create_dir_all(root.join(directory)).unwrap();
        }
        fs::copy(
            repo().join("scripts/release.sh"),
            root.join("scripts/release.sh"),
        )
        .unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[workspace.package]\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        for name in ["bash", "perl", "cat", "jq"] {
            std::os::unix::fs::symlink(tool(name), root.join("bin").join(name)).unwrap();
        }
        for name in [
            "gh", "git", "cargo", "date", "sleep", "curl", "cmake", "ninja", "make", "rustc",
            "nvcc", "mesh-llm",
        ] {
            let path = root.join("bin").join(name);
            fs::write(&path, include_str!("release_script/tool.sh")).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
        }
        // API-returned rows are already scoped by gh's request filters; jq selects latest.
        fs::write(
            root.join("runs.json"),
            serde_json::to_vec(&json!([
                {"databaseId":202,"headSha":HEAD,"createdAt":"2026-08-14T12:00:02Z"},
                {"databaseId":201,"headSha":HEAD,"createdAt":"2026-08-14T12:00:01Z"}
            ]))
            .unwrap(),
        )
        .unwrap();
        Self { _temp: temp, root }
    }
    fn run(&self, script: &str, args: &[&str], route: &str, failure: &str) -> Output {
        let environment = [
            ("PATH", self.root.join("bin").into_os_string()),
            ("HOME", self.root.clone().into_os_string()),
            ("TMPDIR", self.root.join("scratch").into_os_string()),
            ("FIXTURE_ROOT", self.root.clone().into_os_string()),
            ("FIXTURE_HEAD", HEAD.into()),
            ("FIXTURE_START", START.into()),
            ("FIXTURE_VERSION", VERSION.into()),
            ("FIXTURE_ROUTE", route.into()),
            ("FAIL_AT", failure.into()),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value)))
        .collect::<BTreeMap<_, _>>();
        let arguments = ["-c", script, "release-caller-fixture"]
            .into_iter()
            .chain(args.iter().copied())
            .map(|arg| Value::Public(arg.into()))
            .collect();
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: tool("bash"),
                cwd: self.root.clone(),
                environment,
                arguments,
            },
            &Limits {
                execution: Duration::from_secs(15),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
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
        .unwrap();
        assert!(result.process.failure.is_none(), "{:?}", result.process);
        assert!(result.process.cleanup.complete);
        Output {
            code: result.process.status.unwrap().code().unwrap(),
            stdout: String::from_utf8_lossy(result.stdout.unwrap().as_bytes()).into_owned(),
            stderr: String::from_utf8_lossy(result.stderr.unwrap().as_bytes()).into_owned(),
        }
    }
    fn main(&self, route: &str, failure: &str, canary: bool) -> Output {
        let mut args = vec!["scripts/release.sh", VERSION, "--skip-gpu-bundles"];
        if canary {
            args.push("--canary");
        }
        // The parent child has closed stdin; only this finite wrapper supplies confirmation.
        self.run("printf 'yes\\n' | bash \"$@\"", &args, route, failure)
    }
    fn calls(&self) -> Vec<Vec<String>> {
        let bytes = fs::read(self.root.join("calls")).unwrap_or_default();
        bytes
            .split(|byte| *byte == 0)
            .fold(vec![Vec::new()], |mut calls, part| {
                if part.is_empty() {
                    if !calls.last().unwrap().is_empty() {
                        calls.push(Vec::new());
                    }
                } else {
                    calls
                        .last_mut()
                        .unwrap()
                        .push(String::from_utf8(part.to_vec()).unwrap());
                }
                calls
            })
            .into_iter()
            .filter(|call| !call.is_empty())
            .collect()
    }
}
fn named<'a>(calls: &'a [Vec<String>], prefix: &[&str]) -> Vec<&'a Vec<String>> {
    calls
        .iter()
        .filter(|call| {
            call.len() >= prefix.len()
                && call
                    .iter()
                    .zip(prefix)
                    .all(|(actual, expected)| actual == expected)
        })
        .collect()
}
#[test]
fn release_script_url_extraction_and_unknown_output_use_actual_helper() {
    let fixture = Fixture::new();
    for (url, expected) in [
        ("https://github.com/fixture/repo/actions/runs/101", "101\n"),
        (
            "https://github.com/fixture/repo/actions/runs/101?check_suite_focus=true",
            "101\n",
        ),
        ("workflow dispatched without a URL", ""),
        (
            "https://github.com/fixture/repo/actions/runs/not-numeric",
            "",
        ),
    ] {
        let output = fixture.run(
            "source \"$1\"; workflow_run_id_from_url \"$2\"",
            &["scripts/release.sh", url],
            "url",
            "",
        );
        assert_eq!(output.code, 0, "{}", output.stderr);
        assert_eq!(output.stdout, expected);
    }
    assert!(
        fixture.calls().is_empty(),
        "pure extraction must not dispatch or query"
    );
}
#[test]
fn release_script_fallback_preserves_filter_arguments_and_selects_latest_returned_row() {
    let fixture = Fixture::new();
    let sha = "literal commit argument with spaces; unexecuted";
    let output = fixture.run(
        "source \"$1\"; find_dispatched_release_run_id \"$2\" \"$3\"",
        &["scripts/release.sh", sha, START],
        "fallback",
        "",
    );
    assert_eq!(output.code, 0, "{}", output.stderr);
    assert_eq!(output.stdout, "202\n");
    let calls = fixture.calls();
    let query = named(&calls, &["gh", "run", "list"]);
    assert_eq!(query.len(), 1);
    assert_eq!(
        &query[0][..],
        [
            "gh",
            "run",
            "list",
            "--workflow",
            "release.yml",
            "--branch",
            "main",
            "--commit",
            sha,
            "--event",
            "workflow_dispatch",
            "--created",
            &format!(">={START}"),
            "--json",
            "databaseId,headSha,createdAt",
            "--jq",
            "sort_by(.createdAt) | reverse | .[0].databaseId // empty"
        ]
    );
}
#[test]
fn release_script_actual_main_routes_url_or_correlated_fallback_to_numeric_watch() {
    for (route, expected) in [("url", "101"), ("fallback", "202")] {
        let fixture = Fixture::new();
        let output = fixture.main(route, "", true);
        assert_eq!(output.code, 0, "{}", output.stderr);
        assert!(output.stdout.contains("Canary release workflow succeeded"));
        let calls = fixture.calls();
        let dispatch = named(&calls, &["gh", "workflow", "run"]);
        assert_eq!(dispatch.len(), 1);
        assert_eq!(
            &dispatch[0][..],
            [
                "gh",
                "workflow",
                "run",
                "release.yml",
                "--ref",
                "main",
                "--raw-field",
                "version=v0.1.1-rc1",
                "--raw-field",
                "skip_gpu_bundles=true",
                "--raw-field",
                "canary=true"
            ]
        );
        let watch = named(&calls, &["gh", "run", "watch"]);
        assert_eq!(watch.len(), 1);
        assert_eq!(
            &watch[0][..],
            ["gh", "run", "watch", expected, "--compact", "--exit-status"]
        );
        let query = named(&calls, &["gh", "run", "list"]);
        assert_eq!(query.len(), usize::from(route == "fallback"));
        if route == "fallback" {
            assert!(query[0].windows(2).any(|pair| pair == ["--commit", HEAD]));
            assert!(
                query[0]
                    .windows(2)
                    .any(|pair| pair == ["--created", &format!(">={START}")])
            );
        }
        let dispatch_index = calls.iter().position(|call| call == dispatch[0]).unwrap();
        assert!(
            calls[..dispatch_index]
                .iter()
                .any(|call| call == &["git", "rev-parse", "origin/main"])
        );
        assert!(
            calls[..dispatch_index]
                .iter()
                .any(|call| call == &["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"])
        );
        assert_eq!(
            named(&calls, &["cargo"])[0].as_slice(),
            [
                "cargo",
                "xtool",
                "release",
                "version-at-least",
                "0.1.0",
                VERSION
            ]
        );
    }
}
#[test]
fn release_script_dispatch_lookup_and_watch_failure_never_report_release_success() {
    for (failure, expected_code) in [
        ("dispatch", Some(17)),
        ("lookup", None),
        ("watch", Some(23)),
    ] {
        let fixture = Fixture::new();
        let output = fixture.main("fallback", failure, false);
        assert_ne!(output.code, 0, "{failure}: {}", output.stderr);
        if let Some(expected) = expected_code {
            assert_eq!(output.code, expected);
        }
        assert!(!output.stdout.contains("Release complete"));
        assert!(!output.stdout.contains("Canary release workflow succeeded"));
        let calls = fixture.calls();
        let watches = named(&calls, &["gh", "run", "watch"]);
        assert_eq!(watches.len(), usize::from(failure == "watch"));
        assert!(
            named(&calls, &["git", "show"]).is_empty(),
            "failed workflow cannot reach publication/version verification"
        );
        if failure == "dispatch" {
            assert!(named(&calls, &["gh", "run", "list"]).is_empty());
        }
    }
}
