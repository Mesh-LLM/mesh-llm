//! `ci-ops sccache-stats` against
//! `.github/actions/capture-sccache-stats/capture.py`. Each case runs in a
//! temporary directory whose `bin/` (the whole `PATH`) holds a stub
//! `sccache` printing `payload.json`, or failing with a secret on stderr when
//! `error` exists. Exit status, both streams, the written evidence file and
//! the `--github-output` file are compared with a golden in
//! `fixtures/ci_operations/sccache/` (`{root}` is the temp dir) and, when
//! configured, a side-by-side legacy run. `GITHUB_OUTPUT` and
//! `GITHUB_STEP_SUMMARY` point at temp files the tool must never touch.

use crate::support::{CAPTURE_ENV, LEGACY_ENV, Stage, TestResult, fixture_dir, repo_root};
use serde_json::{Value, json};
use std::error::Error;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const SCRIPT: &str = ".github/actions/capture-sccache-stats/capture.py";
const STATS: &str = "evidence/sccache-stats.json";
const SECRET: &str = "https://cache-user:stderr-secret@cache.example/private";
const FORBIDDEN: [&str; 7] = [
    "stderr-secret",
    "count-key-secret",
    "private-source",
    "location-secret",
    "raw-version-secret",
    "payload-secret",
    "/Users/private",
];

#[derive(Debug, PartialEq, Eq)]
struct Observed {
    code: i32,
    stdout: String,
    stderr: String,
    stats: Option<String>,
    github_output: Option<String>,
}

struct Sandbox {
    stage: Stage,
}

impl Sandbox {
    /// `payload: None` leaves `bin/` empty, so `sccache` is not installed.
    fn new(label: &str, payload: Option<&[u8]>, fail: bool) -> Result<Self, Box<dyn Error>> {
        let stage = Stage::empty(&format!("sccache-{label}"))?;
        fs::create_dir_all(stage.path().join("bin"))?;
        let root = stage.root_arg();
        if let Some(payload) = payload {
            stage.write("payload.json", payload)?;
            stage.write(
                "bin/sccache",
                format!(
                    "#!/bin/sh\nif [ \"$#\" -eq 3 ] && [ \"$1\" = --show-stats ] && \
                     [ \"$2\" = --stats-format ] && [ \"$3\" = json ]; then\n  \
                     if [ -f '{root}/error' ]; then /bin/cat '{root}/error' >&2; exit 23; fi\n  \
                     /bin/cat '{root}/payload.json'\nelse\n  exit 2\nfi\n"
                )
                .as_bytes(),
            )?;
            fs::set_permissions(
                stage.path().join("bin/sccache"),
                fs::Permissions::from_mode(0o755),
            )?;
        }
        if fail {
            stage.write("error", format!("{SECRET}\n").as_bytes())?;
        }
        Ok(Self { stage })
    }

    fn reset(&self) {
        for name in ["evidence", "github-output", "env-output", "env-summary"] {
            let path = self.stage.path().join(name);
            let _missing = fs::remove_dir_all(&path).or_else(|_| fs::remove_file(&path));
        }
    }

    fn optional(&self, relative: &str) -> Option<String> {
        let root = self.stage.root_arg();
        fs::read_to_string(self.stage.path().join(relative))
            .ok()
            .map(|text| text.replace(&root, "{root}"))
    }

    fn run(
        &self,
        program: &Path,
        prefix: &[PathBuf],
        args: &[&str],
    ) -> Result<Observed, Box<dyn Error>> {
        let root = self.stage.root_arg();
        let output = Command::new(program)
            .current_dir(self.stage.path())
            .args(prefix)
            .args(args.iter().map(|arg| arg.replace("{root}", &root)))
            .env("PATH", self.stage.path().join("bin"))
            .env("GITHUB_OUTPUT", self.stage.path().join("env-output"))
            .env("GITHUB_STEP_SUMMARY", self.stage.path().join("env-summary"))
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env_remove("COLUMNS")
            .stdin(Stdio::null())
            .output()?;
        assert!(
            self.optional("env-output").is_none(),
            "GITHUB_OUTPUT written"
        );
        assert!(
            self.optional("env-summary").is_none(),
            "GITHUB_STEP_SUMMARY written"
        );
        Ok(Observed {
            code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).replace(&root, "{root}"),
            stderr: String::from_utf8_lossy(&output.stderr).replace(&root, "{root}"),
            stats: self.optional(STATS),
            github_output: self.optional("github-output"),
        })
    }
}

fn golden_path(name: &str) -> PathBuf {
    fixture_dir().join("sccache").join(format!("{name}.json"))
}

fn text(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

/// Runs the port, then (if configured) the legacy script on a reset copy;
/// both must match each other and the captured golden.
fn case(name: &str, sandbox: &Sandbox, args: &[&str]) -> Result<Observed, Box<dyn Error>> {
    sandbox.reset();
    let prefix = ["ci-ops", "sccache-stats"].map(PathBuf::from);
    let actual = sandbox.run(Path::new(env!("CARGO_BIN_EXE_xtask")), &prefix, args)?;
    if let Some(python) = std::env::var_os(LEGACY_ENV).map(PathBuf::from) {
        sandbox.reset();
        let legacy = sandbox.run(&python, &[repo_root().join(SCRIPT)], args)?;
        if std::env::var_os(CAPTURE_ENV).is_some() {
            let golden = json!({
                "code": legacy.code,
                "stdout": legacy.stdout,
                "stderr": legacy.stderr,
                "stats": legacy.stats,
                "github_output": legacy.github_output,
            });
            fs::create_dir_all(fixture_dir().join("sccache"))?;
            fs::write(
                golden_path(name),
                serde_json::to_string_pretty(&golden)? + "\n",
            )?;
        }
        assert_eq!(
            actual,
            legacy,
            "{name}: Rust port differs from legacy {}",
            python.display()
        );
    }
    let golden: Value = serde_json::from_slice(&fs::read(golden_path(name))?)?;
    let expected = Observed {
        code: i32::try_from(golden["code"].as_i64().ok_or("golden code")?)?,
        stdout: text(&golden["stdout"]).ok_or("golden stdout")?,
        stderr: text(&golden["stderr"]).ok_or("golden stderr")?,
        stats: text(&golden["stats"]),
        github_output: text(&golden["github_output"]),
    };
    assert_eq!(
        actual, expected,
        "{name}: Rust port differs from captured golden"
    );
    for forbidden in FORBIDDEN {
        let surface = format!("{actual:?}");
        assert!(!surface.contains(forbidden), "{name}: leaked {forbidden}");
    }
    Ok(actual)
}

fn valid_payload() -> Value {
    json!({
        "stats": {
            "compile_requests": 12,
            "requests_executed": 10,
            "compilations": 4,
            "cache_writes": 3,
            "cache_read_errors": 0,
            "cache_write_errors": 0,
            "cache_hits": {"counts": {"Rust": 6}, "adv_counts": {}},
            "cache_misses": {"counts": {"Rust": 4}, "adv_counts": {}},
            "cache_errors": {"counts": {}, "adv_counts": {}},
        },
        "version": "test",
    })
}

fn bytes(payload: &Value) -> Vec<u8> {
    let mut raw = serde_json::to_vec(payload).unwrap_or_default();
    raw.push(b'\n');
    raw
}

fn full_args<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut args = vec![
        "--artifact-name",
        "sccache-test-1",
        "--output",
        "{root}/evidence/sccache-stats.json",
        "--github-output",
        "{root}/github-output",
    ];
    args.extend_from_slice(extra);
    args
}

fn payload_case(name: &str, payload: &[u8], extra: &[&str]) -> Result<Observed, Box<dyn Error>> {
    let sandbox = Sandbox::new(name, Some(payload), false)?;
    case(name, &sandbox, &full_args(extra))
}

fn edited(edit: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut payload = valid_payload();
    edit(&mut payload);
    bytes(&payload)
}

#[test]
fn sanitized_evidence_and_outputs_match_legacy_for_each_expectation() -> TestResult {
    let valid = bytes(&valid_payload());
    let cases: [(&str, &[&str]); 6] = [
        ("opportunistic", &[]),
        (
            "warm_pass",
            &["--cache-expectation", "warm", "--minimum-hit-rate", "0.5"],
        ),
        (
            "warm_failure",
            &["--cache-expectation", "warm", "--minimum-hit-rate", "0.80"],
        ),
        (
            "cold",
            &["--cache-expectation", "cold", "--minimum-hit-rate", "0.80"],
        ),
        (
            "warm_tiny_floor",
            &["--cache-expectation=warm", "--minimum-hit-rate=1e-5"],
        ),
        ("negative_zero_floor", &["--c", "warm", "--m", "-0.0"]),
    ];
    for (name, extra) in cases {
        let outcome = payload_case(name, &valid, extra)?;
        assert_eq!(outcome.code, 0, "{name}: {}", outcome.stderr);
        assert!(outcome.stats.is_some(), "{name}: no evidence");
    }
    Ok(())
}

#[test]
fn raw_secrets_urls_and_paths_never_reach_logs_or_evidence() -> TestResult {
    let payload = edited(|payload| {
        payload["stats"]["cache_hits"]["counts"] =
            json!({"/Users/private/cache/path?token=count-key-secret": 6});
        payload["stats"]["not_cached"] =
            json!({"/home/runner/private-source": 1, "https://stats-secret.example/cache": 2});
        payload["cache_location"] =
            json!("WebDAV: https://cache-user:location-secret@cache.example");
        payload["version"] = json!("raw-version-secret");
        payload["url"] = json!("https://payload-secret.example/cache");
        payload["absolute_path"] = json!("/Users/private/sccache");
    });
    let outcome = payload_case("secrets", &payload, &[])?;
    assert_eq!(outcome.code, 0, "{}", outcome.stderr);
    Ok(())
}

#[test]
fn zero_request_observations_warn_and_classify_by_floor() -> TestResult {
    let zero = |executed_zero: bool| {
        edited(|payload| {
            let stats = &mut payload["stats"];
            stats["compile_requests"] = json!(0);
            if executed_zero {
                for name in ["requests_executed", "compilations", "cache_writes"] {
                    stats[name] = json!(0);
                }
            }
            stats["cache_hits"]["counts"] = json!({});
            stats["cache_misses"]["counts"] = json!({});
        })
    };
    let pass = payload_case(
        "zero_requests_zero_floor",
        &zero(true),
        &["--cache-expectation", "warm", "--minimum-hit-rate", "0"],
    )?;
    assert!(
        pass.stdout
            .contains("::warning title=sccache reported zero compile requests")
    );
    let fail = payload_case(
        "zero_requests_positive_floor",
        &zero(false),
        &["--cache-expectation", "warm", "--minimum-hit-rate", "0.01"],
    )?;
    assert!(
        fail.github_output
            .unwrap_or_default()
            .contains("cache_passed=false")
    );
    let nested = edited(|payload| {
        payload["stats"]["cache_hits"]["counts"] = json!({"Rust": {"a": 2, "b": {"c": 3}}, "C": 1});
    });
    payload_case("nested_counts", &nested, &[])?;
    let duplicate = b"{\"stats\": {\"compile_requests\": -1, \"compile_requests\": 5, \
        \"requests_executed\": 1, \"compilations\": 1, \"cache_writes\": 1, \
        \"cache_read_errors\": 0, \"cache_write_errors\": 0, \
        \"cache_hits\": {\"counts\": {\"a\": 1, \"a\": 7}}, \
        \"cache_misses\": {\"counts\": {\"b\": 2}}, \"cache_errors\": {\"counts\": {}}}}\r\n";
    payload_case("duplicate_keys", duplicate, &[])?;
    Ok(())
}

#[test]
fn untrustworthy_sccache_output_is_rejected_without_evidence() -> TestResult {
    let invalid: Vec<(&str, Vec<u8>)> = vec![
        (
            "missing_count_map",
            edited(|p| drop(p["stats"].as_object_mut().map(|s| s.remove("cache_misses")))),
        ),
        (
            "missing_counter",
            edited(|p| drop(p["stats"].as_object_mut().map(|s| s.remove("compilations")))),
        ),
        (
            "boolean_counter",
            edited(|p| p["stats"]["cache_writes"] = json!(true)),
        ),
        (
            "negative_counter",
            edited(|p| p["stats"]["compile_requests"] = json!(-3)),
        ),
        (
            "float_counter",
            edited(|p| p["stats"]["compile_requests"] = json!(1.0)),
        ),
        (
            "counts_not_object",
            edited(|p| p["stats"]["cache_errors"]["counts"] = json!([1])),
        ),
        (
            "count_map_not_object",
            edited(|p| p["stats"]["cache_hits"] = json!(6)),
        ),
        (
            "count_boolean",
            edited(|p| p["stats"]["cache_hits"]["counts"] = json!({"x": false})),
        ),
        (
            "count_negative",
            edited(|p| p["stats"]["cache_misses"]["counts"] = json!({"x": {"y": -1}})),
        ),
        (
            "count_string",
            edited(|p| p["stats"]["cache_misses"]["counts"] = json!({"x": "1"})),
        ),
        ("stats_not_object", edited(|p| p["stats"] = json!(null))),
        ("root_not_object", b"[1, 2]\n".to_vec()),
        (
            "corrupt_json",
            b"{\"stats\": {\"compile_requests\": 1,\n  oops}\n".to_vec(),
        ),
        ("empty_output", Vec::new()),
        ("extra_data", b"{} {}\n".to_vec()),
        ("bom", b"\xef\xbb\xbf{}\n".to_vec()),
        (
            "nan_count",
            b"{\"stats\": {\"compile_requests\": NaN}}\n".to_vec(),
        ),
        (
            "huge_counter",
            b"{\"stats\": {\"compile_requests\": 1e400}}\n".to_vec(),
        ),
    ];
    for (name, payload) in invalid {
        let outcome = payload_case(name, &payload, &[])?;
        assert_eq!(outcome.code, 1, "{name}");
        assert!(outcome.stats.is_none(), "{name}: evidence written");
        assert!(
            outcome.stderr.starts_with("ERROR: "),
            "{name}: {}",
            outcome.stderr
        );
    }
    Ok(())
}

#[test]
fn infinity_counters_are_floats_not_decode_errors() -> TestResult {
    let raw =
        b"{\"stats\": {\"compile_requests\": 1, \"requests_executed\": 1, \"compilations\": 1, \
        \"cache_writes\": 1, \"cache_read_errors\": 0, \"cache_write_errors\": 0, \
        \"cache_hits\": {\"counts\": {\"x\": -Infinity}}}}\n";
    let outcome = payload_case("infinity_in_counts", raw, &[])?;
    assert_eq!(outcome.code, 1);
    Ok(())
}

#[test]
fn sccache_failures_and_absence_fail_closed() -> TestResult {
    let valid = bytes(&valid_payload());
    let failing = Sandbox::new("sccache_error", Some(&valid), true)?;
    let outcome = case("sccache_error", &failing, &full_args(&[]))?;
    assert!(outcome.stderr.contains("exit code 23"));
    let missing = Sandbox::new("missing_sccache", None, false)?;
    case("missing_sccache", &missing, &full_args(&[]))?;
    let escape = Sandbox::new("artifact_escape", Some(&valid), false)?;
    let args = [
        "--artifact-name",
        "../sccache-test",
        "--output",
        "{root}/evidence/sccache-stats.json",
    ];
    case("artifact_escape", &escape, &args)?;
    for (name, rate) in [
        ("rate_above_one", "1.5"),
        ("rate_nan", "nan"),
        ("rate_underscore", "0_5"),
    ] {
        payload_case(name, &valid, &["--minimum-hit-rate", rate])?;
    }
    Ok(())
}

#[test]
fn relative_output_without_github_output_matches_legacy() -> TestResult {
    let valid = bytes(&valid_payload());
    let sandbox = Sandbox::new("relative", Some(&valid), false)?;
    let outcome = case(
        "relative_output",
        &sandbox,
        &[
            "--artifact-name",
            "a.b_c-1",
            "--output",
            "evidence//./sccache-stats.json",
        ],
    )?;
    assert_eq!(outcome.code, 0, "{}", outcome.stderr);
    assert!(outcome.github_output.is_none());
    Ok(())
}

#[test]
fn argparse_surface_matches_legacy() -> TestResult {
    let valid = bytes(&valid_payload());
    let sandbox = Sandbox::new("argv", Some(&valid), false)?;
    let cases: [(&str, &[&str]); 16] = [
        ("argv_help", &["-h"]),
        ("argv_help_bundled", &["--artifact-name", "a", "-hx"]),
        ("argv_help_explicit", &["--help=x"]),
        (
            "argv_help_after_unknown",
            &["-x", "--artifact-name", "a", "-h"],
        ),
        ("argv_empty", &[]),
        ("argv_missing_output", &["--artifact-name", "a"]),
        (
            "argv_unrecognized",
            &["x", "--artifact-name", "a", "--output", "o", "--x=1", "y"],
        ),
        (
            "argv_double_dash",
            &["--artifact-name", "a", "--output", "o", "--", "x"],
        ),
        ("argv_only_double_dash", &["--", "--artifact-name", "a"]),
        ("argv_invalid_choice", &["--cache-expectation", "x"]),
        (
            "argv_invalid_float",
            &["--minimum-hit-rate", "x", "--artifact-name", "a"],
        ),
        ("argv_float_attached_empty", &["--minimum-hit-rate="]),
        ("argv_expected_value", &["--artifact-name"]),
        (
            "argv_value_is_option",
            &["--output", "--", "--artifact-name", "a"],
        ),
        (
            "argv_negative_exponent",
            &["--artifact-name", "a", "--output", "o", "--m", "-1e3"],
        ),
        (
            "argv_negative_value",
            &[
                "--minimum-hit-rate",
                "-0.5",
                "--artifact-name",
                "../a",
                "--output",
                "o",
            ],
        ),
    ];
    for (name, args) in cases {
        let outcome = case(name, &sandbox, args)?;
        assert!(outcome.stats.is_none(), "{name}: evidence written");
    }
    Ok(())
}
