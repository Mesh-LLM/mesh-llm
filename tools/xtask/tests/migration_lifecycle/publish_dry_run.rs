use crate::process::{self, Cancellation, Completion, Limits, ProcessSpec, Readiness, Value};
use std::{collections::BTreeMap, fs, os::unix::fs::PermissionsExt, path::Path, time::Duration};

const SCRIPT: &str = include_str!("../../../../scripts/publish-crates.sh");
const SELECTOR: &str = include_str!("../../../../scripts/lib/automation.sh");

fn historical_fixture() -> (Fixture, std::path::PathBuf, Vec<String>) {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/crates_recovery/v0761-roster.json"
    ))
    .unwrap();
    assert_eq!(
        data["source_sha"],
        "ff18c0b5a74c0317d943bb6611a1ce4836b35d61"
    );
    let roster: Vec<String> = data["roster"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| name.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(roster.len(), 50);
    let fixture = Fixture::publication_tools(Fixture::new_with_order(
        "model-artifact",
        "model-ref",
        roster.clone(),
    ));
    let root = fixture.temporary.path();
    let historical = root.join("release source");
    fs::create_dir_all(historical.join("scripts")).unwrap();
    assert_eq!(fixture.order, roster);
    fs::write(
        historical.join("Cargo.toml"),
        "[workspace.package]\nversion = \"0.76.1\"\n",
    )
    .unwrap();
    let script = format!(
        "printf executed > \"$PWD/historical-executed\"\nexit 97\npublish_crates=(\n{}\n)\n",
        roster.join("\n")
    );
    fs::write(historical.join("scripts/publish-crates.sh"), script).unwrap();

    // Package metadata is a finite fixture, relocated to the historical source.
    // It proves caller behavior, not Cargo execution against the real release.
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("metadata.json")).unwrap()).unwrap();
    let packages = metadata["packages"].as_array_mut().unwrap();
    packages.retain(|package| roster.iter().any(|name| package["name"] == *name));
    for package in packages {
        let name = package["name"].as_str().unwrap();
        // The historical Cargo package name differs from its source directory.
        let directory = if name == "mesh-llm-client" {
            "mesh-client"
        } else {
            name
        };
        let package_dir = historical.join("crates").join(directory);
        fs::create_dir_all(&package_dir).unwrap();
        fs::write(package_dir.join("Cargo.toml"), "[package]\n").unwrap();
        fs::write(
            package_dir.join("README.md"),
            "historical package documentation\n",
        )
        .unwrap();
        package["manifest_path"] = serde_json::json!(package_dir.join("Cargo.toml"));
        package["version"] = serde_json::json!("0.76.1");
        for dependency in package["dependencies"].as_array_mut().unwrap() {
            if let Some(path) = dependency["path"].as_str() {
                let name = Path::new(path).file_name().unwrap();
                dependency["path"] = serde_json::json!(historical.join("crates").join(name));
                dependency["req"] = serde_json::json!("^0.76.1");
            }
        }
    }
    fs::create_dir_all(historical.join("tools/xtask")).unwrap();
    fs::write(
        historical.join("tools/xtask/Cargo.toml"),
        "never build historical automation\n",
    )
    .unwrap();
    for relative in [
        "crates/mesh-client/src/models/catalog.json",
        "crates/mesh-llm-node/src/catalog.json",
    ] {
        let path = historical.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "{}\n").unwrap();
    }
    metadata["workspace_members"] = serde_json::json!(roster);
    fs::write(
        root.join("metadata.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    let cargo = root.join("bin/cargo");
    let source = fs::read_to_string(&cargo)
        .unwrap()
        .replace("set -eu\n", "set -eu\npwd >> \"$FIXTURE/cargo-cwd.log\"\n");
    fs::write(cargo, source).unwrap();
    let curl = root.join("bin/curl");
    let source = fs::read_to_string(&curl)
        .unwrap()
        .replace("/0.68.0", "/0.76.1");
    fs::write(curl, source).unwrap();
    (fixture, historical, roster)
}

#[test]
fn actual_controller_resume_uses_historical_roster_version_and_cargo_directory() {
    let (fixture, historical, roster) = historical_fixture();
    let report = fixture.invoke_from(
        &historical,
        &["--resume"],
        &[
            ("CARGO_REGISTRY_TOKEN", "fixture-secret-token"),
            ("CRATES_IO_PUBLISH_SETTLE_SECONDS", "0"),
            ("PROVIDER_STATUS", "200"),
        ],
    );
    assert!(report.process.success(), "{report:?}");
    let expected: Vec<_> = roster
        .iter()
        .filter(|name| name.as_str() != "model-ref")
        .cloned()
        .collect();
    assert_eq!(fixture.published(), expected);
    assert!(!historical.join("historical-executed").exists());
    let stdout = String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes());
    assert!(stdout.contains("model-ref@0.76.1 already published; skipping"));
    assert!(!stdout.contains("@0.68.0"));
    assert_eq!(
        fixture.log("metadata.log"),
        "metadata --format-version 1 --no-deps --locked\n"
    );
    assert_eq!(
        fixture.log("cargo-cwd.log").lines().count(),
        expected.len() + 1
    );
    assert!(fixture
        .log("cargo-cwd.log")
        .lines()
        .all(|cwd| Path::new(cwd).canonicalize().unwrap() == historical.canonicalize().unwrap()));
    assert_eq!(fixture.log("curl.log").lines().count(), roster.len());
    assert!(
        fixture
            .log("curl.log")
            .lines()
            .all(|url| url.ends_with("/0.76.1"))
    );
    assert!(fixture.log("sleep.log").is_empty());
}

#[test]
fn actual_controller_rejects_invalid_historical_roster_before_registry_or_publish() {
    for invalid in [
        "publish_crates=(\n controller-only-unknown\n)",
        "publish_crates=(\n model-artifact\n model-ref\n)",
        "publish_crates=(\n model-ref\n model-ref\n)",
        "publish_crates=(\n $(touch historical-executed)\n)",
    ] {
        let (fixture, historical, _) = historical_fixture();
        fs::write(historical.join("scripts/publish-crates.sh"), invalid).unwrap();
        let report = fixture.invoke_from(
            &historical,
            &["--resume"],
            &[
                ("CARGO_REGISTRY_TOKEN", "fixture-secret-token"),
                ("CRATES_IO_PUBLISH_SETTLE_SECONDS", "0"),
            ],
        );
        assert!(!report.process.success(), "{report:?}");
        assert!(fixture.published().is_empty());
        assert!(fixture.log("curl.log").is_empty());
        assert!(!historical.join("historical-executed").exists());
    }
}

#[test]
fn actual_controller_metadata_failure_prevents_registry_and_publication() {
    let (fixture, historical, _) = historical_fixture();
    executable(fixture.temporary.path(), "cargo", "#!/bin/sh\nexit 91\n");
    let report = fixture.invoke_from(
        &historical,
        &["--resume"],
        &[
            ("CARGO_REGISTRY_TOKEN", "fixture-secret-token"),
            ("CRATES_IO_PUBLISH_SETTLE_SECONDS", "0"),
        ],
    );
    assert!(!report.process.success(), "{report:?}");
    assert!(fixture.published().is_empty());
    assert!(fixture.log("curl.log").is_empty());
    assert!(!historical.join("historical-executed").exists());
}

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
    provider: String,
}

impl Fixture {
    fn new(consumer: &str, provider: &str) -> Self {
        Self::new_with_order(consumer, provider, declared_crates())
    }
    fn new_with_order(consumer: &str, provider: &str, order: Vec<String>) -> Self {
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

        assert!(order.iter().any(|name| name == consumer));
        assert!(order.iter().any(|name| name == provider));
        fs::create_dir_all(root.join("tools/xtask")).unwrap();
        fs::write(
            root.join("tools/xtask/Cargo.toml"),
            "historical automation marker\n",
        )
        .unwrap();
        let current_layout = order.iter().any(|name| name == "skippy-model-ref");
        let package_directory = |name: &str| {
            if !current_layout {
                return root.join("crates").join(name);
            }
            let product = if name.starts_with("skippy-") {
                "skippy"
            } else {
                "mesh"
            };
            let directory = if name == "mesh-llm-client" {
                "mesh-client"
            } else {
                name
            };
            root.join(product).join("crates").join(directory)
        };
        for name in &order {
            let package = package_directory(name);
            fs::create_dir_all(&package).unwrap();
            fs::write(package.join("Cargo.toml"), "[package]\n").unwrap();
            fs::write(package.join("README.md"), "publication fixture\n").unwrap();
        }
        let catalog_root = if current_layout {
            root.join("mesh")
        } else {
            root.to_path_buf()
        };
        for relative in [
            "crates/mesh-client/src/models/catalog.json",
            "crates/mesh-llm-node/src/catalog.json",
        ] {
            let path = catalog_root.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "{}\n").unwrap();
        }
        let packages: Vec<_> = order.iter().map(|name| {
            let dependencies = if name == consumer {
                vec![serde_json::json!({"name":provider,"req":"^0.68.0","kind":null,"path":package_directory(provider),"optional":true})]
            } else { Vec::new() };
            serde_json::json!({"id":name,"name":name,"version":"0.68.0","description":"publication fixture","license":"MIT","license_file":null,"repository":"https://example.invalid/repository","readme":"README.md","manifest_path":package_directory(name).join("Cargo.toml"),"publish":null,"dependencies":dependencies})
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
        Self {
            temporary,
            order,
            provider: provider.to_owned(),
        }
    }

    fn run(&self, provider: &str) -> process::RawProcessReport {
        self.invoke(&["--dry-run", "--allow-dirty"], &[("PROVIDER", provider)])
    }

    fn invoke(&self, args: &[&str], extra: &[(&str, &str)]) -> process::RawProcessReport {
        self.invoke_from(self.temporary.path(), args, extra)
    }

    fn invoke_from(
        &self,
        cwd: &Path,
        args: &[&str],
        extra: &[(&str, &str)],
    ) -> process::RawProcessReport {
        let root = self.temporary.path();
        let script = root.join("scripts/publish-crates.sh");
        let mut spec = ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: cwd.into(),
            arguments: std::iter::once(script.into_os_string())
                .chain(args.iter().map(std::ffi::OsString::from))
                .map(Value::Public)
                .collect(),
            environment: BTreeMap::from([
                (
                    "PATH".into(),
                    Value::Public(format!("{}:/usr/bin:/bin", root.join("bin").display()).into()),
                ),
                ("HOME".into(), Value::Public(root.into())),
                ("FIXTURE".into(), Value::Public(root.into())),
                (
                    "PROVIDER".into(),
                    Value::Public(self.provider.clone().into()),
                ),
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
            "metadata --format-version 1 --no-deps --locked\n"
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
    Fixture::new("skippy-model-artifact", "skippy-model-ref")
        .assert_skip("skippy-model-artifact", "skippy-model-ref");
}

impl Fixture {
    fn publishing() -> Self {
        Self::publication_tools(Self::new("skippy-model-artifact", "skippy-model-ref"))
    }
    fn publication_tools(fixture: Self) -> Self {
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
  "https://crates.io/api/v1/crates/$PROVIDER/0.68.0") printf '%s' "${PROVIDER_STATUS:-404}" ;;
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
            ("FAIL_CRATE", "skippy-model-artifact"),
            ("FAIL_COUNT", "1"),
            ("FAIL_MODE", "rate"),
            ("CRATES_IO_PUBLISH_MAX_ATTEMPTS", "3"),
        ],
    );
    assert!(report.process.success(), "{report:?}");
    let mut expected = fixture.order.clone();
    let index = expected
        .iter()
        .position(|name| name == "skippy-model-artifact")
        .unwrap();
    expected.insert(index, "skippy-model-artifact".into());
    assert_eq!(fixture.published(), expected);
    let delays = fixture.log("sleep.log");
    assert_eq!(delays.lines().count(), 1);
    assert!(delays.trim().parse::<u64>().unwrap() > 0);
    assert!(
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
            .contains("crates.io rate limit hit for skippy-model-artifact@0.68.0")
    );
    assert!(fixture.log("curl.log").is_empty());
}
#[test]
fn actual_publish_exhausted_retry_stops_before_later_crates() {
    let fixture = Fixture::publishing();
    let report = fixture.real(
        &[],
        &[
            ("FAIL_CRATE", "skippy-model-artifact"),
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
        .position(|name| name == "skippy-model-artifact")
        .unwrap();
    let mut expected = fixture.order[..=index].to_vec();
    expected.push("skippy-model-artifact".into());
    assert_eq!(fixture.published(), expected);
    assert_eq!(fixture.log("sleep.log").lines().count(), 1);
    assert!(
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
            .contains("retry limit exceeded for skippy-model-artifact@0.68.0 after 2 attempts")
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
                ("FAIL_CRATE", "skippy-model-ref"),
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
                String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes()).contains(
                    "skippy-model-ref@0.68.0 already published according to cargo; continuing"
                )
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
            .filter(|name| status != "200" || *name != "skippy-model-ref")
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
                    .contains("skippy-model-ref@0.68.0 already published; skipping")
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
            ("FAIL_CRATE", "skippy-model-ref"),
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
        .position(|name| name == "skippy-model-ref")
        .unwrap();
    assert_eq!(fixture.published(), fixture.order[..=index]);
    assert!(fixture.log("sleep.log").is_empty());
}
