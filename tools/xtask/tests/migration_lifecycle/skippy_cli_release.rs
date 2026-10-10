//! Actual standalone CLI producers and release adapter, with inert inspection/build peers.
use crate::{
    process,
    workflow_yaml::{self, Node},
};
use flate2::read::GzDecoder;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

const PAYLOAD: &[u8] = b"\x7fELFinert standalone CLI";
fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn executable(path: &Path, source: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, source).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn document(path: &str) -> Node {
    workflow_yaml::parse(&fs::read_to_string(repository().join(path)).unwrap()).unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn steps(node: &Node) -> &[Node] {
    let Node::Seq(steps) = node.get("steps").unwrap() else {
        panic!("steps required")
    };
    steps
}
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .canonicalize()
            .unwrap()
            .join("standalone CLI with spaces");
        for path in [
            "bin",
            "home",
            "tmp",
            "input",
            "target/release",
            "target/debug",
        ] {
            fs::create_dir_all(root.join(path)).unwrap();
        }
        for path in [
            "skippy/scripts/package-cli-release.sh",
            "scripts/lib/automation.sh",
        ] {
            let destination = root.join(path);
            fs::create_dir_all(destination.parent().unwrap()).unwrap();
            fs::copy(repository().join(path), destination).unwrap();
        }
        executable(
            &root.join("bin/cargo"),
            "#!/bin/bash\nset -euo pipefail\n[[ $1 == xtool ]] || exit 92\nshift\nprintf 'verify\\n' >> \"$FIXTURE_ROOT/events\"\nexec \"$FIXTURE_XTASK\" \"$@\"\n",
        );
        executable(
            &root.join("bin/just"),
            "#!/bin/bash\nset -euo pipefail\n[[ $# == 1 && ( $1 == skippy-cli-build || $1 == skippy-cli-release-build ) ]] || exit 93\nprintf 'build|%s\\n' \"$1\" >> \"$FIXTURE_ROOT/events\"\nexit \"${BUILD_STATUS:-0}\"\n",
        );
        executable(
            &root.join("bin/readelf"),
            "#!/bin/bash\nset -euo pipefail\n[[ $LC_ALL == C && $# == 2 ]] || exit 94\nprintf 'inspect|%s\\n' \"$1\" >> \"$FIXTURE_ROOT/events\"\ncase $1 in -d) printf ' 0x1 (NEEDED) Shared library: [%s]\\n' \"$INSPECT_IMPORT\";; -V) printf \"Version needs section '.gnu.version_r'\\nName: GLIBC_2.17\\n\";; *) exit 95;; esac\n",
        );
        executable(
            &root.join("bin/python3"),
            "#!/bin/sh\nprintf forbidden-python > \"$FIXTURE_ROOT/forbidden\"\nexit 96\n",
        );
        executable(
            &root.join("bin/sha256sum"),
            "#!/bin/bash\nset -euo pipefail\nprintf 'checksum|%s\\n' \"$*\" >> \"$FIXTURE_ROOT/events\"\nexec /usr/bin/shasum -a 256 \"$@\"\n",
        );
        fs::write(root.join("sentinel"), b"keep unrelated bytes").unwrap();
        Self { _temp: temp, root }
    }
    fn run(&self, arguments: Vec<String>, overrides: &[(&str, &str)]) -> process::RawProcessReport {
        let mut environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            ("HOME", self.root.join("home").display().to_string()),
            ("TMPDIR", self.root.join("tmp").display().to_string()),
            ("FIXTURE_ROOT", self.root.display().to_string()),
            ("FIXTURE_XTASK", env!("CARGO_BIN_EXE_xtask").to_owned()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                env!("CARGO_BIN_EXE_xtask").to_owned(),
            ),
            ("INSPECT_IMPORT", "libc.so.6".to_owned()),
            ("INPUT_PROFILE", "release".to_owned()),
            ("INPUT_OUTPUT_DIR", "prepared".to_owned()),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), process::Value::Public(v.into())))
        .collect();
        for (key, value) in overrides {
            environment.insert((*key).into(), process::Value::Public((*value).into()));
        }
        let report = process::supervise_raw(
            &process::ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                environment,
                arguments: arguments
                    .into_iter()
                    .map(|v| process::Value::Public(v.into()))
                    .collect(),
            },
            &process::Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
            &process::Cancellation::default(),
            process::RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert_eq!(report.process.outcome, process::Outcome::Exited);
        assert!(report.process.failure.is_none() && report.process.cleanup.complete);
        assert!(
            !self.root.join("forbidden").exists(),
            "retired Python cannot run"
        );
        assert_eq!(
            fs::read(self.root.join("sentinel")).unwrap(),
            b"keep unrelated bytes"
        );
        report
    }
    fn input(&self, target: &str) -> Value {
        let binary = if target.starts_with("windows") {
            "skippy.exe"
        } else {
            "skippy"
        };
        let format = if target.starts_with("windows") {
            "pe"
        } else if target.starts_with("darwin") {
            "macho"
        } else {
            "elf"
        };
        fs::write(self.root.join("input").join(binary), PAYLOAD).unwrap();
        fs::write(
            self.root.join("input").join(format!("{binary}.sha256")),
            format!("{}  {binary}\n", digest(PAYLOAD)),
        )
        .unwrap();
        let report = json!({"binary":binary,"binary_sha256":digest(PAYLOAD),"format":format,
            "imports":["libc.so.6"],"policy":"mesh-llm-dynamic-host-v2","rejected_imports":[]});
        self.report(&report);
        report
    }
    fn report(&self, report: &Value) {
        fs::write(
            self.root.join("input/host-imports.json"),
            serde_json::to_vec(report).unwrap(),
        )
        .unwrap();
    }
    fn package(&self, target: &str) -> process::RawProcessReport {
        self.run(
            vec![
                "skippy/scripts/package-cli-release.sh".into(),
                "v1.2.3".into(),
                target.into(),
                "input".into(),
                "release".into(),
            ],
            &[],
        )
    }
    fn events(&self) -> String {
        fs::read_to_string(self.root.join("events")).unwrap_or_default()
    }
}
fn rejected(report: &process::RawProcessReport) {
    assert!(
        !report.process.status.unwrap().success(),
        "unexpected acceptance"
    );
}

#[test]
fn standalone_cli_release_packages_four_exact_checksum_bound_archives_without_python() {
    for target in [
        "darwin-aarch64",
        "linux-x86_64",
        "linux-aarch64",
        "windows-x86_64",
    ] {
        let fixture = Fixture::new();
        let imports = fixture.input(target);
        let report = fixture.package(target);
        assert!(
            report.process.status.unwrap().success(),
            "{:?}",
            report.stderr
        );
        let name = format!("skippy-v1.2.3-{target}-cli.tar.gz");
        let archive = fixture.root.join("release").join(&name);
        let archive_bytes = fs::read(&archive).unwrap();
        assert_eq!(
            fs::read_to_string(archive.with_file_name(format!("{name}.sha256"))).unwrap(),
            format!("{}  {name}\n", digest(&archive_bytes))
        );
        let mut members = BTreeMap::new();
        let mut bundle = tar::Archive::new(GzDecoder::new(archive_bytes.as_slice()));
        for entry in bundle.entries().unwrap() {
            let mut entry = entry.unwrap();
            let path = entry.path().unwrap().into_owned();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).unwrap();
            assert!(members.insert(path.display().to_string(), bytes).is_none());
        }
        let binary = imports["binary"].as_str().unwrap();
        assert_eq!(members.len(), 3);
        assert_eq!(members[binary], PAYLOAD);
        assert_eq!(
            members[&format!("{binary}.sha256")],
            format!("{}  {binary}\n", digest(PAYLOAD)).as_bytes()
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&members["host-imports.json"]).unwrap(),
            imports
        );
        assert!(
            report
                .stdout
                .as_ref()
                .unwrap()
                .as_bytes()
                .ends_with(format!("release/{name}\n").as_bytes())
        );
    }
}

#[test]
fn standalone_cli_release_refuses_invalid_import_reports_before_checksum_or_output() {
    for mutation in [
        "missing",
        "malformed",
        "policy",
        "rejected",
        "forged-imports",
        "binary",
        "format",
        "digest",
        "no-digest",
        "no-imports",
        "imports-type",
    ] {
        let fixture = Fixture::new();
        let mut report = fixture.input("linux-x86_64");
        match mutation {
            "missing" => fs::remove_file(fixture.root.join("input/host-imports.json")).unwrap(),
            "malformed" => fs::write(
                fixture.root.join("input/host-imports.json"),
                b"{invalid-json",
            )
            .unwrap(),
            _ => {
                match mutation {
                    "policy" => report["policy"] = json!("none"),
                    "rejected" => report["rejected_imports"] = json!(["libllama.so"]),
                    "forged-imports" => report["imports"] = json!(["libllama.so"]),
                    "binary" => report["binary"] = json!("mesh-llm"),
                    "format" => report["format"] = json!("pe"),
                    "digest" => report["binary_sha256"] = json!(digest(b"previous CLI")),
                    "no-digest" => {
                        report.as_object_mut().unwrap().remove("binary_sha256");
                    }
                    "no-imports" => {
                        report.as_object_mut().unwrap().remove("imports");
                    }
                    "imports-type" => report["imports"] = json!([false]),
                    _ => unreachable!(),
                }
                fixture.report(&report);
            }
        }
        rejected(&fixture.package("linux-x86_64"));
        assert!(!fixture.root.join("release").exists(), "{mutation}");
        assert!(!fixture.events().contains("checksum|"), "{mutation}");
    }
}

#[test]
fn standalone_cli_native_admission_rejects_unknown_target_before_input_io() {
    let fixture = Fixture::new();
    let report = fixture.run(vec!["-c".into(), "exec \"$FIXTURE_XTASK\" product skippy-cli-input missing-report missing-binary linux-riscv64".into()], &[]);
    assert_eq!(report.process.status.unwrap().code(), Some(2));
    assert!(report.stdout.unwrap().as_bytes().is_empty());
    assert!(
        String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
            .contains("unsupported Skippy CLI target")
    );
    assert!(fixture.events().is_empty());
}

#[test]
fn standalone_cli_release_refuses_changed_binary_checksum_and_unbounded_or_linked_report() {
    for mutation in [
        "binary-changed",
        "checksum-changed",
        "large-report",
        "linked-report",
    ] {
        let fixture = Fixture::new();
        fixture.input("linux-x86_64");
        match mutation {
            "binary-changed" => {
                fs::write(fixture.root.join("input/skippy"), b"different binary").unwrap()
            }
            "checksum-changed" => fs::write(
                fixture.root.join("input/skippy.sha256"),
                format!("{}  skippy\n", digest(b"old")),
            )
            .unwrap(),
            "large-report" => fs::write(
                fixture.root.join("input/host-imports.json"),
                vec![b' '; 1024 * 1024 + 1],
            )
            .unwrap(),
            "linked-report" => {
                fs::rename(
                    fixture.root.join("input/host-imports.json"),
                    fixture.root.join("imports-original.json"),
                )
                .unwrap();
                std::os::unix::fs::symlink(
                    "../imports-original.json",
                    fixture.root.join("input/host-imports.json"),
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
        rejected(&fixture.package("linux-x86_64"));
        assert!(!fixture.root.join("release").exists(), "{mutation}");
        if mutation != "checksum-changed" {
            assert!(!fixture.events().contains("checksum|"), "{mutation}");
        }
    }
}

#[test]
fn standalone_cli_release_refuses_unsupported_targets_without_reading_or_publishing_inputs() {
    for target in [
        "bogus-x86_64",
        "linux-amd64",
        "linux-riscv64",
        "windows-aarch64",
    ] {
        let fixture = Fixture::new();
        let report = fixture.package(target);
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        assert!(
            String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
                .contains("invalid target")
        );
        assert!(fixture.events().is_empty());
        assert!(!fixture.root.join("release").exists());
    }
}

fn unix_producer() -> String {
    let action = document(".github/actions/prepare-skippy-cli-input/action.yml");
    let producer = steps(action.get("runs").unwrap())
        .iter()
        .find(|s| text(s, "shell") == "bash")
        .unwrap();
    assert_eq!(text(producer, "if"), "runner.os != 'Windows'");
    text(producer, "run").to_owned()
}
#[test]
fn standalone_cli_actual_producer_verifies_imports_before_checksumming_and_binds_report_bytes() {
    for (profile, directory, recipe) in [
        ("release", "release", "skippy-cli-release-build"),
        ("debug", "debug", "skippy-cli-build"),
        ("dev", "debug", "skippy-cli-build"),
    ] {
        let fixture = Fixture::new();
        fs::write(
            fixture.root.join(format!("target/{directory}/skippy")),
            PAYLOAD,
        )
        .unwrap();
        let report = fixture.run(
            vec!["-c".into(), unix_producer()],
            &[("INPUT_PROFILE", profile)],
        );
        assert!(
            report.process.status.unwrap().success(),
            "{:?}",
            report.stderr
        );
        let imports: Value = serde_json::from_slice(
            &fs::read(fixture.root.join("prepared/host-imports.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(imports["binary_sha256"], digest(PAYLOAD));
        assert_eq!(imports["rejected_imports"], json!([]));
        assert_eq!(
            fs::read_to_string(fixture.root.join("prepared/skippy.sha256")).unwrap(),
            format!("{}  skippy\n", digest(PAYLOAD))
        );
        let events = fixture.events();
        assert!(events.starts_with(&format!("build|{recipe}\nverify\n")));
        assert!(events.find("inspect|-d").unwrap() < events.find("checksum|").unwrap());
    }
}

#[test]
fn standalone_cli_actual_producer_backend_import_refusal_cannot_publish_checksum() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("target/release/skippy"), PAYLOAD).unwrap();
    let report = fixture.run(
        vec!["-c".into(), unix_producer()],
        &[("INSPECT_IMPORT", "libllama.so")],
    );
    rejected(&report);
    assert!(
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
            .contains("host dependency policy rejected")
    );
    let imports: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("prepared/host-imports.json")).unwrap())
            .unwrap();
    assert_eq!(imports["rejected_imports"], json!(["libllama.so"]));
    assert!(!fixture.root.join("prepared/skippy.sha256").exists());
    assert!(!fixture.events().contains("checksum|"));
}

#[test]
fn standalone_cli_actual_producer_build_failure_or_unknown_profile_stops_before_inspection() {
    for env in [[("BUILD_STATUS", "43")], [("INPUT_PROFILE", "unbounded")]] {
        let fixture = Fixture::new();
        rejected(&fixture.run(vec!["-c".into(), unix_producer()], &env));
        assert!(!fixture.root.join("prepared").exists());
        assert!(!fixture.events().contains("verify"));
    }
}

#[test]
fn standalone_cli_platform_graph_orders_separate_producer_upload_before_mesh_host() {
    for (platform, job) in [
        ("linux", "linux_host"),
        ("macos", "macos_host"),
        ("windows", "windows_host"),
    ] {
        let workflow = document(&format!(".github/workflows/ci-{platform}-host-slice.yml"));
        let row = workflow.get("jobs").unwrap().get(job).unwrap();
        let items = steps(row);
        let producer = items
            .iter()
            .position(|s| text(s, "name") == "Prepare standalone Skippy CLI")
            .unwrap();
        let upload = &items[producer + 1];
        assert_eq!(
            text(&items[producer], "uses"),
            "./.github/actions/prepare-skippy-cli-input"
        );
        assert!(text(upload, "uses").starts_with("actions/upload-artifact@"));
        assert_eq!(
            text(upload.get("with").unwrap(), "name"),
            format!("ci-skippy-cli-{platform}-${{{{ matrix.host.architecture }}}}")
        );
        assert!(text(&items[producer + 2], "name").starts_with("Prepare immutable"));
        assert_eq!(
            text(upload.get("with").unwrap(), "if-no-files-found"),
            "error"
        );
    }
    let action = document(".github/actions/prepare-skippy-cli-input/action.yml");
    let windows = steps(action.get("runs").unwrap())
        .iter()
        .find(|s| text(s, "shell") == "pwsh")
        .unwrap();
    assert_eq!(text(windows, "if"), "runner.os == 'Windows'");
    let run = text(windows, "run");
    assert!(run.find("verify-host-dependencies").unwrap() < run.find("Get-FileHash").unwrap());
    assert!(
        run.contains("if ($LASTEXITCODE -ne 0) { throw 'Skippy host import verification failed' }")
    );
    assert!(run.contains("--format pe"));
}
