use crate::process::{self, Cancellation, Completion, Limits, ProcessSpec, Readiness, Value};
use std::{collections::BTreeMap, fs, os::unix::fs::PermissionsExt, path::Path, time::Duration};

const SCRIPT: &str = include_str!("../../../../scripts/publish-crates.sh");
const SELECTOR: &str = include_str!("../../../../scripts/lib/automation.sh");

fn executable(root: &Path, name: &str, source: &str) {
    let path = root.join("bin").join(name);
    fs::write(&path, source).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn declared_crates() -> Vec<String> {
    let body = SCRIPT
        .split_once("\npublish_crates=(\n")
        .expect("actual declared publish chain")
        .1
        .split_once("\n)")
        .unwrap()
        .0;
    body.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.trim_matches('"').to_owned())
        .collect()
}

struct Fixture {
    temporary: tempfile::TempDir,
    order: Vec<String>,
}

impl Fixture {
    fn new(consumer: &str, provider: &str) -> Self {
        let temporary = tempfile::Builder::new()
            .prefix("publish dry run ")
            .tempdir()
            .unwrap();
        let root = temporary.path();
        fs::create_dir_all(root.join("scripts/lib")).unwrap();
        fs::create_dir(root.join("bin")).unwrap();
        fs::write(root.join("scripts/publish-crates.sh"), SCRIPT).unwrap();
        fs::write(root.join("scripts/lib/automation.sh"), SELECTOR).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[workspace.package]\nversion = \"0.68.0\"\n",
        )
        .unwrap();
        let order = declared_crates();
        assert!(order.iter().any(|name| name == consumer));
        assert!(order.iter().any(|name| name == provider));
        let packages: Vec<_> = order.iter().map(|name| {
            let dependencies = if name == consumer {
                vec![serde_json::json!({"name":provider,"kind":null,"path":root.join("crates").join(provider),"optional":true})]
            } else { Vec::new() };
            serde_json::json!({"id":name,"name":name,"manifest_path":root.join("crates").join(name).join("Cargo.toml"),"publish":null,"dependencies":dependencies})
        }).collect();
        fs::write(
            root.join("metadata.json"),
            serde_json::to_vec(&serde_json::json!({"packages":packages,"workspace_members":order}))
                .unwrap(),
        )
        .unwrap();
        executable(
            root,
            "cargo",
            r#"#!/bin/sh
set -eu
case "$1" in
  metadata)
    printf '%s\n' "$*" >> "$FIXTURE/metadata.log"
    cat "$FIXTURE/metadata.json"
    ;;
  publish)
    printf '%s\n' "$*" >> "$FIXTURE/cargo.log"
    printf '%s\n' "${LLAMA_STAGE_LINK_MODE:-}" >> "$FIXTURE/link.log"
    printf '%s\n' "${LLAMA_STAGE_BUILD_DIR:-}" >> "$FIXTURE/native.log"
    ;;
  *) exit 91 ;;
esac
"#,
        );
        executable(
            root,
            "curl",
            r#"#!/bin/sh
set -eu
for argument do url="$argument"; done
printf '%s\n' "$url" >> "$FIXTURE/curl.log"
case "$url" in
  "https://crates.io/api/v1/crates/$PROVIDER/0.68.0") printf '404' ;;
  *) exit 92 ;;
esac
"#,
        );
        executable(
            root,
            "sleep",
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$FIXTURE/sleep.log\"\nexit 93\n",
        );
        executable(
            root,
            "date",
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$FIXTURE/date.log\"\nexit 94\n",
        );
        Self { temporary, order }
    }

    fn run(&self, provider: &str) -> process::RawProcessReport {
        self.invoke(&["--dry-run", "--allow-dirty"], &[("PROVIDER", provider)])
    }

    fn invoke(&self, args: &[&str], extra: &[(&str, &str)]) -> process::RawProcessReport {
        let root = self.temporary.path();
        let mut spec = ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.into(),
            arguments: std::iter::once("scripts/publish-crates.sh")
                .chain(args.iter().copied())
                .map(|value| Value::Public(value.into()))
                .collect(),
            environment: BTreeMap::from([
                (
                    "PATH".into(),
                    Value::Public(format!("{}:/usr/bin:/bin", root.join("bin").display()).into()),
                ),
                ("HOME".into(), Value::Public(root.into())),
                ("FIXTURE".into(), Value::Public(root.into())),
                ("PROVIDER".into(), Value::Public("model-ref".into())),
                (
                    "MESH_LLM_AUTOMATION_BIN".into(),
                    Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
                ),
            ]),
        };
        for (key, value) in extra {
            spec.environment
                .insert((*key).into(), Value::Public((*value).into()));
        }
        let report = process::supervise_raw(
            &spec,
            &Limits {
                execution: Duration::from_secs(8),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 256 * 1024,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            process::RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(256 * 1024),
                stderr: std::num::NonZeroUsize::new(256 * 1024),
            },
        )
        .unwrap();
        assert!(
            report.process.cleanup.complete
                && report.process.failure.is_none()
                && report.stdout.is_some()
                && report.stderr.is_some(),
            "{report:?}"
        );
        report
    }

    fn assert_skip(&self, consumer: &str, provider: &str) {
        let report = self.run(provider);
        assert!(report.process.success(), "{report:?}");
        let stdout = String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes());
        assert!(
            stdout.contains(&format!(
                "dry-run cannot verify {consumer} until {provider}@0.68.0 exists in crates.io"
            )),
            "expected skip diagnostic absent from raw capture"
        );
        let root = self.temporary.path();
        let calls = fs::read_to_string(root.join("cargo.log")).unwrap();
        let expected: Vec<_> = self
            .order
            .iter()
            .filter(|name| *name != consumer)
            .cloned()
            .collect();
        assert_eq!(self.published(), expected);
        assert!(expected.iter().any(|name| name == provider));
        for call in calls.lines() {
            let words: Vec<_> = call.split_whitespace().collect();
            for flag in ["--locked", "--dry-run", "--allow-dirty"] {
                assert!(words.contains(&flag), "missing {flag}: {call}");
            }
        }
        assert_eq!(
            fs::read_to_string(root.join("metadata.log")).unwrap(),
            "metadata --format-version 1 --no-deps\n"
        );
        assert_eq!(
            fs::read_to_string(root.join("curl.log")).unwrap(),
            format!("https://crates.io/api/v1/crates/{provider}/0.68.0\n")
        );
        assert!(!root.join("sleep.log").exists());
        assert!(!root.join("date.log").exists());
        assert!(
            fs::read_to_string(root.join("link.log"))
                .unwrap()
                .lines()
                .all(|mode| mode == "dynamic")
        );
        let native = fs::read_to_string(root.join("native.log")).unwrap();
        assert!(native.lines().all(|directory| {
            Path::new(directory).canonicalize().unwrap()
                == root
                    .join("target/publish-crates-native-verify")
                    .canonicalize()
                    .unwrap()
        }));
        assert!(root.join("target/publish-crates-native-verify").is_dir());
    }
}

#[test]
fn actual_publish_dry_run_derives_new_workspace_dependency_and_skips_consumer() {
    Fixture::new("skippy-protocol", "skippy-tokenizer")
        .assert_skip("skippy-protocol", "skippy-tokenizer");
}

#[test]
fn actual_publish_dry_run_retains_provider_order_and_skips_missing_registry_dependency_without_sleep()
 {
    Fixture::new("model-artifact", "model-ref").assert_skip("model-artifact", "model-ref");
}

impl Fixture {
    fn publishing() -> Self {
        let fixture = Self::new("model-artifact", "model-ref");
        let root = fixture.temporary.path();
        executable(
            root,
            "cargo",
            r#"#!/bin/sh
set -eu
case "$1" in
  metadata) printf '%s\n' "$*" >> "$FIXTURE/metadata.log"; cat "$FIXTURE/metadata.json"; exit 0 ;;
  publish) ;;
  *) exit 91 ;;
esac
printf '%s\n' "$*" >> "$FIXTURE/cargo.log"
printf '%s\n' "${LLAMA_STAGE_LINK_MODE:-}" >> "$FIXTURE/link.log"
printf '%s\n' "${LLAMA_STAGE_BUILD_DIR:-}" >> "$FIXTURE/native.log"
previous=''
for argument do
  if [ "$previous" = '-p' ]; then crate="$argument"; break; fi
  previous="$argument"
done
count=0
if [ -f "$FIXTURE/count-$crate" ]; then count=$(cat "$FIXTURE/count-$crate"); fi
count=$((count + 1))
printf '%s\n' "$count" > "$FIXTURE/count-$crate"
if [ "$crate" = "${FAIL_CRATE:-}" ] && [ "$count" -le "${FAIL_COUNT:-0}" ]; then
  case "${FAIL_MODE:-}" in
    rate) printf '%s\n' 'status 429 Too Many Requests:' 'Please try again after Fri, 22 May 2026 09:58:23 GMT' >&2 ;;
    uploaded) printf '%s\n' 'error: crate version is already uploaded' >&2 ;;
    fatal) printf 'fatal: registry token %s leaked in diagnostic\n' "${CARGO_REGISTRY_TOKEN:-}" >&2 ;;
    *) exit 95 ;;
  esac
  exit 101
fi
"#,
        );
        executable(
            root,
            "curl",
            r#"#!/bin/sh
set -eu
for argument do url="$argument"; done
printf '%s\n' "$url" >> "$FIXTURE/curl.log"
printf '%s\n' "$*" >> "$FIXTURE/curl-args.log"
case "$url" in
  "https://crates.io/api/v1/crates/model-ref/0.68.0") printf '%s' "${PROVIDER_STATUS:-404}" ;;
  https://crates.io/api/v1/crates/*/0.68.0) printf '404' ;;
  *) exit 92 ;;
esac
"#,
        );
        executable(
            root,
            "sleep",
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$FIXTURE/sleep.log\"\n",
        );
        executable(
            root,
            "date",
            r#"#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FIXTURE/date.log"
case "$*" in
  *'Fri, 22 May 2026 09:58:23 GMT'*) printf '1779443903\n' ;;
  '-u +%s') printf '1779443600\n' ;;
  *) exit 94 ;;
esac
"#,
        );
        fixture
    }
    fn log(&self, name: &str) -> String {
        let path = self.temporary.path().join(name);
        if !path.exists() {
            return String::new();
        }
        fs::read_to_string(path).unwrap()
    }
    fn published(&self) -> Vec<String> {
        self.log("cargo.log")
            .lines()
            .map(|line| {
                let words: Vec<_> = line.split_whitespace().collect();
                assert_eq!(words.first(), Some(&"publish"));
                assert!(words.contains(&"--locked"));
                let index = words.iter().position(|word| *word == "-p").unwrap();
                words[index + 1].to_owned()
            })
            .collect()
    }
    fn native_environment(&self, expected: &Path) {
        let modes = self.log("link.log");
        assert!(!modes.is_empty());
        assert!(modes.lines().all(|mode| mode == "dynamic"));
        let directories = self.log("native.log");
        assert!(!directories.is_empty());
        assert!(
            directories
                .lines()
                .all(|directory| Path::new(directory).canonicalize().unwrap()
                    == expected.canonicalize().unwrap())
        );
        assert!(expected.is_dir());
    }
    fn real(&self, args: &[&str], extra: &[(&str, &str)]) -> process::RawProcessReport {
        // Public fake token: only actual shell redaction may satisfy this proof.
        let mut environment = vec![
            ("CARGO_REGISTRY_TOKEN", "fixture-secret-token"),
            ("CRATES_IO_PUBLISH_SETTLE_SECONDS", "0"),
        ];
        environment.extend_from_slice(extra);
        self.invoke(args, &environment)
    }
}
#[test]
fn actual_publish_native_verification_is_dynamic_and_honors_explicit_directory() {
    for explicit in [false, true] {
        let fixture = Fixture::publishing();
        let directory = fixture.temporary.path().join(if explicit {
            "prepared native"
        } else {
            "target/publish-crates-native-verify"
        });
        let extra = if explicit {
            vec![("LLAMA_STAGE_BUILD_DIR", directory.to_str().unwrap())]
        } else {
            Vec::new()
        };
        let report = fixture.invoke(&["--dry-run", "--allow-dirty"], &extra);
        assert!(report.process.success(), "{report:?}");
        fixture.native_environment(&directory);
    }
}
#[test]
fn actual_publish_rate_limit_retries_then_continues_in_declared_order() {
    let fixture = Fixture::publishing();
    let report = fixture.real(
        &[],
        &[
            ("FAIL_CRATE", "model-artifact"),
            ("FAIL_COUNT", "1"),
            ("FAIL_MODE", "rate"),
            ("CRATES_IO_PUBLISH_MAX_ATTEMPTS", "3"),
        ],
    );
    assert!(report.process.success(), "{report:?}");
    let mut expected = fixture.order.clone();
    let index = expected
        .iter()
        .position(|name| name == "model-artifact")
        .unwrap();
    expected.insert(index, "model-artifact".into());
    assert_eq!(fixture.published(), expected);
    let delays = fixture.log("sleep.log");
    assert_eq!(delays.lines().count(), 1);
    assert!(delays.trim().parse::<u64>().unwrap() > 0);
    assert!(
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
            .contains("crates.io rate limit hit for model-artifact@0.68.0")
    );
    assert!(fixture.log("curl.log").is_empty());
}
#[test]
fn actual_publish_exhausted_retry_stops_before_later_crates() {
    let fixture = Fixture::publishing();
    let report = fixture.real(
        &[],
        &[
            ("FAIL_CRATE", "model-artifact"),
            ("FAIL_COUNT", "5"),
            ("FAIL_MODE", "rate"),
            ("CRATES_IO_PUBLISH_MAX_ATTEMPTS", "2"),
        ],
    );
    assert!(!report.process.success(), "{report:?}");
    assert_eq!(
        report
            .process
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code),
        Some(101)
    );
    let index = fixture
        .order
        .iter()
        .position(|name| name == "model-artifact")
        .unwrap();
    let mut expected = fixture.order[..=index].to_vec();
    expected.push("model-artifact".into());
    assert_eq!(fixture.published(), expected);
    assert_eq!(fixture.log("sleep.log").lines().count(), 1);
    assert!(
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
            .contains("retry limit exceeded for model-artifact@0.68.0 after 2 attempts")
    );
}
#[test]
fn actual_publish_requires_token_and_rejects_conflicting_resume_before_cargo() {
    for args in [vec![], vec!["--dry-run", "--resume"]] {
        let fixture = Fixture::publishing();
        let report = fixture.invoke(&args, &[]);
        assert!(!report.process.success(), "{report:?}");
        assert_eq!(
            report
                .process
                .status
                .as_ref()
                .and_then(std::process::ExitStatus::code),
            Some(1)
        );
        assert!(fixture.log("cargo.log").is_empty());
        assert!(fixture.log("metadata.log").is_empty());
        assert!(fixture.log("curl.log").is_empty());
        let diagnostic = String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes());
        assert!(
            diagnostic.contains(if args.is_empty() {
                "CARGO_REGISTRY_TOKEN is required"
            } else {
                "--resume is only supported for real publishing"
            }),
            "{diagnostic}"
        );
    }
}
#[test]
fn actual_publish_uses_cargo_without_speculative_registry_probe_or_duplicate_retry() {
    for uploaded in [false, true] {
        let fixture = Fixture::publishing();
        let extra = if uploaded {
            vec![
                ("FAIL_CRATE", "model-ref"),
                ("FAIL_COUNT", "1"),
                ("FAIL_MODE", "uploaded"),
                ("PROVIDER_STATUS", "500"),
            ]
        } else {
            Vec::new()
        };
        let report = fixture.real(&[], &extra);
        assert!(report.process.success(), "{report:?}");
        assert_eq!(fixture.published(), fixture.order);
        assert!(fixture.log("curl.log").is_empty());
        assert!(fixture.log("sleep.log").is_empty());
        if uploaded {
            assert!(
                String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes())
                    .contains("model-ref@0.68.0 already published according to cargo; continuing")
            );
        }
    }
}
#[test]
fn actual_publish_resume_skips_only_confirmed_versions_and_defers_unknown_to_cargo() {
    for status in ["200", "500"] {
        let fixture = Fixture::publishing();
        let report = fixture.real(&["--resume"], &[("PROVIDER_STATUS", status)]);
        assert!(report.process.success(), "{report:?}");
        let expected: Vec<_> = fixture
            .order
            .iter()
            .filter(|name| status != "200" || *name != "model-ref")
            .cloned()
            .collect();
        assert_eq!(fixture.published(), expected);
        let probes: Vec<_> = fixture
            .order
            .iter()
            .map(|name| format!("https://crates.io/api/v1/crates/{name}/0.68.0"))
            .collect();
        assert_eq!(fixture.log("curl.log").lines().collect::<Vec<_>>(), probes);
        assert!(fixture.log("curl-args.log").lines().all(|line| line.contains("--user-agent mesh-llm-publish-crates/0.68.0 (https://github.com/Mesh-LLM/mesh-llm)")));
        if status == "200" {
            assert!(
                String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes())
                    .contains("model-ref@0.68.0 already published; skipping")
            );
        }
        assert!(fixture.log("sleep.log").is_empty());
    }
}
#[test]
fn actual_publish_fatal_cargo_output_redacts_token_and_does_not_continue() {
    let fixture = Fixture::publishing();
    let report = fixture.real(
        &[],
        &[
            ("FAIL_CRATE", "model-ref"),
            ("FAIL_COUNT", "1"),
            ("FAIL_MODE", "fatal"),
        ],
    );
    assert!(!report.process.success(), "{report:?}");
    assert_eq!(
        report
            .process
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code),
        Some(101)
    );
    let stderr = String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes());
    assert!(
        stderr.contains("<redacted>"),
        "shell diagnostic was not redacted"
    );
    assert!(!stderr.contains("fixture-secret-token"));
    assert!(
        !String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes())
            .contains("fixture-secret-token")
    );
    let index = fixture
        .order
        .iter()
        .position(|name| name == "model-ref")
        .unwrap();
    assert_eq!(fixture.published(), fixture.order[..=index]);
    assert!(fixture.log("sleep.log").is_empty());
}
