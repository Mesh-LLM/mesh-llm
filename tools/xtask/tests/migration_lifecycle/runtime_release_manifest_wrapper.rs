//! Actual release producer composition: portable admission precedes publication.
//! Finite inert runtime bytes require neither native inspectors nor a build.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use flate2::{Compression, write::GzEncoder};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

const ID: &str = "meshllm-native-runtime-linux-aarch64-cpu";
fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn executable(path: &Path, body: &str) {
    fs::write(path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn octal(field: &mut [u8], value: u64) {
    let digits = format!("{value:0width$o}", width = field.len() - 1);
    field[..digits.len()].copy_from_slice(digits.as_bytes());
}
fn archive(path: &Path, members: &[(String, Vec<u8>)]) {
    let mut raw = Vec::new();
    for (name, bytes) in members {
        let mut header = [0_u8; 512];
        assert!(name.len() < 100);
        header[..name.len()].copy_from_slice(name.as_bytes());
        octal(&mut header[100..108], 0o644);
        octal(&mut header[108..116], 0);
        octal(&mut header[116..124], 0);
        octal(&mut header[124..136], bytes.len() as u64);
        octal(&mut header[136..148], 0);
        header[148..156].fill(b' ');
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
        header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
        raw.extend(header);
        raw.extend(bytes);
        raw.resize(raw.len().div_ceil(512) * 512, 0);
    }
    raw.resize(raw.len() + 1024, 0);
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw).unwrap();
    fs::write(path, encoder.finish().unwrap()).unwrap();
}
struct Fixture {
    directory: tempfile::TempDir,
    root: PathBuf,
    archive: PathBuf,
}
impl Fixture {
    fn new(attack: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory
            .path()
            .canonicalize()
            .unwrap()
            .join("selected source with spaces");
        for relative in ["scripts/lib", "bin", "tmp"] {
            fs::create_dir_all(root.join(relative)).unwrap();
        }
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        for relative in [
            "scripts/generate-native-runtime-release-manifest.sh",
            "skippy/scripts/generate-native-runtime-release-manifest.sh",
            "skippy/crates/skippy-native-runtime/RUNTIME_VERSION",
            "scripts/lib/automation.sh",
        ] {
            fs::create_dir_all(root.join(relative).parent().unwrap()).unwrap();
            fs::copy(source.join(relative), root.join(relative)).unwrap();
        }
        fs::write(
            root.join("Cargo.toml"),
            "[workspace.package]\nversion = \"99.0.0\"\n",
        )
        .unwrap();
        executable(
            &root.join("bin/owner"),
            "printf '%s\\t' \"$@\" >> \"$OWNER_LOG\"\nprintf '\\n' >> \"$OWNER_LOG\"\nexec \"$REAL_XTASK\" \"$@\"",
        );
        for tool in ["cargo", "just", "cmake", "readelf", "otool"] {
            executable(
                &root.join("bin").join(tool),
                "printf '%s\\n' forbidden >> \"$FORBIDDEN_LOG\"\nexit 97",
            );
        }
        let archive_path = root.join(format!("{ID}.tar.gz"));
        let fixture = Self {
            directory,
            root,
            archive: archive_path,
        };
        fixture.package(attack, "0.68.0");
        fixture
    }
    fn package(&self, attack: &str, version: &str) {
        let library = b"inert portable runtime library".to_vec();
        let digest = sha(&library);
        let mut manifest = json!({"schema_version":2,"runtime":{"id":ID,"release_version":version,"skippy_abi":"0.1.25","platform":{"os":"linux","arch":"aarch64","target":"aarch64-unknown-linux-gnu"},"backend":{"kind":"cpu"},"rank":0,"libraries":["lib/runtime.bin"],"files":{"lib/runtime.bin":digest}},"build":{"primary_library":"lib/runtime.bin","library_sha256":digest}});
        if attack == "malformed-manifest" {
            manifest["runtime"]["files"] = json!([]);
        }
        if attack == "library-digest" {
            manifest["runtime"]["files"]["lib/runtime.bin"] = json!("0".repeat(64));
        }
        let mut members = vec![
            (
                format!("{ID}/manifest.json"),
                serde_json::to_vec(&manifest).unwrap(),
            ),
            (format!("{ID}/lib/runtime.bin"), library),
        ];
        if attack == "traversal" {
            members.push(("../escaped".into(), b"must not escape".to_vec()));
        }
        if attack == "sibling" {
            members.push(("unexpected.txt".into(), b"outside artifact".to_vec()));
        }
        archive(&self.archive, &members);
        if attack == "missing-sidecar" {
            return;
        }
        let digest = if attack == "corrupt-sidecar" {
            "0".repeat(64)
        } else {
            sha(&fs::read(&self.archive).unwrap())
        };
        let name = self.archive.file_name().unwrap().to_str().unwrap();
        let text = match attack {
            "empty-sidecar" => String::new(),
            "wrong-sidecar-name" => format!("{digest}  other.tar.gz\n"),
            "noncanonical-sidecar" => format!("{digest} {name}\n"),
            "multiple-sidecars" => format!("{digest}  {name}\n{digest}  {name}\n"),
            _ => format!("{digest}  {name}\n"),
        };
        fs::write(self.root.join(format!("{name}.sha256")), text).unwrap();
    }
    fn invoke(&self, tag: &str) -> process::RawProcessReport {
        self.invoke_version(tag, Some("0.68.0"))
    }
    fn invoke_version(&self, tag: &str, version: Option<&str>) -> process::RawProcessReport {
        let environment: BTreeMap<_, _> = [
            (
                "PATH",
                self.root.join("bin").display().to_string() + ":/usr/bin:/bin",
            ),
            ("HOME", self.root.display().to_string()),
            ("TMPDIR", self.root.join("tmp").display().to_string()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                self.root.join("bin/owner").display().to_string(),
            ),
            ("REAL_XTASK", env!("CARGO_BIN_EXE_xtask").into()),
            (
                "OWNER_LOG",
                self.root.join("owner.calls").display().to_string(),
            ),
            (
                "FORBIDDEN_LOG",
                self.root.join("forbidden.calls").display().to_string(),
            ),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value.into())))
        .collect();
        let mut arguments = vec![
            self.root
                .join("scripts/generate-native-runtime-release-manifest.sh")
                .into_os_string(),
            "--tag".into(),
            tag.into(),
            "--out".into(),
            self.root.join("native-runtimes.json").into_os_string(),
            "--repo".into(),
            "Fixture/runtime".into(),
        ];
        if let Some(version) = version {
            arguments.extend(["--runtime-version".into(), version.into()]);
        }
        arguments.push(self.archive.clone().into_os_string());
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                environment,
                arguments: arguments.into_iter().map(Value::Public).collect(),
            },
            &Limits {
                execution: Duration::from_secs(8),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(1048576),
                stderr: std::num::NonZeroUsize::new(1048576),
            },
        )
        .unwrap();
        assert!(result.process.cleanup.complete, "{:?}", result.process);
        assert!(result.process.failure.is_none(), "{:?}", result.process);
        assert!(
            !self.root.join("forbidden.calls").exists(),
            "portable manifest production must not build or inspect native bytes"
        );
        result
    }
    fn calls(&self) -> Vec<String> {
        fs::read_to_string(self.root.join("owner.calls"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }
}
#[test]
fn release_wrapper_publishes_one_document_after_actual_portable_admission() {
    let fixture = Fixture::new("");
    let result = fixture.invoke("v0.68.0");
    assert!(result.process.success(), "{:?}", result.process);
    let document: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.root.join("native-runtimes.json")).unwrap())
            .unwrap();
    assert_eq!(document["release_version"], "0.68.0");
    assert!(document.get("mesh_version").is_none());
    assert_eq!(document["artifacts"].as_array().unwrap().len(), 1);
    assert_eq!(document["artifacts"][0]["id"], ID);
    assert_eq!(
        document["artifacts"][0]["sha256"],
        sha(&fs::read(&fixture.archive).unwrap())
    );
    assert_eq!(
        document["artifacts"][0]["url"],
        format!("https://github.com/Fixture/runtime/releases/download/v0.68.0/{ID}.tar.gz")
    );
    let calls = fixture.calls();
    assert_eq!(calls.len(), 2);
    assert!(calls[0].starts_with("native\tverify-runtime-package\t--portable\t"));
    assert!(calls[1].starts_with("product\truntime-release-manifest\t"));
    let stdout = result.stdout.unwrap();
    assert!(
        String::from_utf8_lossy(stdout.as_bytes())
            .contains("generated native runtime release manifest: ")
    );
    assert!(!fixture.directory.path().join("escaped").exists());
}
#[test]
fn release_wrapper_rejects_invalid_sidecars_before_publication() {
    for attack in [
        "missing-sidecar",
        "empty-sidecar",
        "corrupt-sidecar",
        "wrong-sidecar-name",
        "noncanonical-sidecar",
        "multiple-sidecars",
    ] {
        let fixture = Fixture::new(attack);
        let result = fixture.invoke("v0.68.0");
        assert!(!result.process.success(), "{attack}");
        assert!(!fixture.root.join("native-runtimes.json").exists());
        assert_eq!(
            fixture.calls().len(),
            1,
            "product must not execute after failed checksum admission"
        );
    }
}
#[test]
fn release_wrapper_rejects_escape_sibling_and_malformed_runtime_before_publication() {
    for attack in [
        "traversal",
        "sibling",
        "malformed-manifest",
        "library-digest",
    ] {
        let fixture = Fixture::new(attack);
        let result = fixture.invoke("v0.68.0");
        assert!(!result.process.success(), "{attack}");
        assert!(!fixture.root.join("native-runtimes.json").exists());
        assert!(!fixture.root.join("escaped").exists());
        assert!(!fixture.directory.path().join("escaped").exists());
        assert_eq!(fixture.calls().len(), 1);
    }
}
#[test]
fn release_wrapper_accepts_independent_tag_with_explicit_or_default_runtime_version() {
    for explicit in [false, true] {
        let fixture = Fixture::new("");
        let owned = fs::read_to_string(
            fixture
                .root
                .join("skippy/crates/skippy-native-runtime/RUNTIME_VERSION"),
        )
        .unwrap();
        let version = if explicit { "0.68.0" } else { owned.trim() };
        fixture.package("", version);
        let result = fixture.invoke_version("product-test-v99.0.0", explicit.then_some(version));
        assert!(result.process.success(), "{:?}", result.process);
        let document: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture.root.join("native-runtimes.json")).unwrap())
                .unwrap();
        assert_eq!(document["release_version"], version);
        assert_eq!(document["artifacts"][0]["release_version"], version);
        assert_eq!(
            document["artifacts"][0]["url"],
            format!(
                "https://github.com/Fixture/runtime/releases/download/product-test-v99.0.0/{ID}.tar.gz"
            )
        );
        assert_eq!(fixture.calls().len(), 2);
    }
}
#[test]
fn release_wrapper_rejects_requested_runtime_mismatch_without_publication() {
    for previous in [false, true] {
        let fixture = Fixture::new("");
        let out = fixture.root.join("native-runtimes.json");
        if previous {
            fs::write(&out, b"previous immutable publication").unwrap();
        }
        let result = fixture.invoke_version("product-test-v99.0.0", Some("0.69.0-rc1"));
        assert!(!result.process.success());
        assert!(
            String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes())
                .contains("does not match requested runtime release")
        );
        if previous {
            assert_eq!(fs::read(out).unwrap(), b"previous immutable publication");
        } else {
            assert!(!out.exists());
        }
        assert_eq!(
            fixture.calls().len(),
            2,
            "valid portable input reaches actual runtime admission"
        );
    }
}
