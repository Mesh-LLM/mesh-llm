//! Actual copied product composer, real typed policy owners, inert producer bytes.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    io::{Read, Write},
    num::NonZeroUsize,
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
    time::Duration,
};

type Snapshot = BTreeMap<PathBuf, (Vec<u8>, u32)>;
const OLD_RECEIPT: &str = "previous_receipt=keep\n";
const OLD_ARCHIVE: &[u8] = b"previous published product: keep";

struct Fixture {
    _scratch: tempfile::TempDir,
    root: PathBuf,
    outside: PathBuf,
}

fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn snapshot(root: &Path) -> Snapshot {
    fn visit(root: &Path, dir: &Path, result: &mut Snapshot) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let metadata = fs::symlink_metadata(&path).unwrap();
            if metadata.is_dir() {
                visit(root, &path, result);
            } else {
                assert!(metadata.is_file(), "fixture producer must be regular");
                result.insert(
                    path.strip_prefix(root).unwrap().into(),
                    (
                        fs::read(&path).unwrap(),
                        metadata.permissions().mode() & 0o777,
                    ),
                );
            }
        }
    }
    let mut result = BTreeMap::new();
    visit(root, root, &mut result);
    result
}

fn invoke(root: &Path, environment: BTreeMap<OsString, Value>) -> process::RawProcessReport {
    invoke_arguments(
        root,
        vec![Value::Public("scripts/ci-compose-product-input.sh".into())],
        environment,
    )
}

fn invoke_arguments(
    root: &Path,
    arguments: Vec<Value>,
    environment: BTreeMap<OsString, Value>,
) -> process::RawProcessReport {
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            arguments,
            cwd: root.into(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(15),
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
    .unwrap();
    assert_eq!(
        result.process.outcome,
        Outcome::Exited,
        "{:?}",
        result.process
    );
    assert!(result.process.cleanup.complete && result.process.failure.is_none());
    result
}

impl Fixture {
    fn new(version: &str) -> Self {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path().join("workspace with spaces");
        let outside = scratch.path().join("outside");
        for path in [
            "scripts/lib",
            "bin",
            "inputs/host",
            "inputs/runtime",
            "product",
        ] {
            fs::create_dir_all(root.join(path)).unwrap();
        }
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("sentinel"), b"outside keep").unwrap();
        fs::write(root.join("sentinel"), b"workspace keep").unwrap();
        fs::write(root.join("product/previous-staging"), b"staging keep").unwrap();
        fs::write(root.join("github-output"), OLD_RECEIPT).unwrap();
        fs::write(root.join("product.tar.gz"), OLD_ARCHIVE).unwrap();
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for path in [
            "scripts/ci-compose-product-input.sh",
            "scripts/lib/automation.sh",
        ] {
            fs::copy(repository.join(path), root.join(path)).unwrap();
        }
        executable(
            &root.join("bin/automation"),
            "#!/bin/sh\nprintf '%s %s\\n' \"$1\" \"$2\" >> \"$PRODUCT_EVENTS\"\nexec \"$PRODUCT_REAL_XTASK\" \"$@\"\n",
        );
        for name in [
            "cargo", "rustc", "cc", "c++", "gcc", "g++", "clang", "clang++", "cmake", "ninja",
            "git", "python", "python3", "readelf", "otool",
        ] {
            executable(
                &root.join("bin").join(name),
                "#!/bin/sh\nprintf 'forbidden %s\\n' \"$0\" >> \"$PRODUCT_EVENTS\"\nexit 98\n",
            );
        }
        let host = format!(
            "#!/bin/sh\n[ \"$#\" -eq 1 ] && [ \"$1\" = --version ] || exit 98\nprintf 'host-version\\n' >> \"$PRODUCT_EVENTS\"\nprintf 'mesh-llm {version}\\n'\n"
        );
        executable(&root.join("inputs/host/mesh-llm"), &host);
        fs::write(root.join("inputs/host/host-imports.json"), b"{}").unwrap();
        fs::write(
            root.join("inputs/host/mesh-llm.sha256"),
            format!("{}  mesh-llm\n", digest(host.as_bytes())),
        )
        .unwrap();
        write_runtime(&root.join("inputs/runtime/runtime"));
        Self {
            _scratch: scratch,
            root: root.canonicalize().unwrap(),
            outside,
        }
    }

    fn run(&self, output: &str) -> process::RawProcessReport {
        self.run_version(output, "")
    }

    fn run_version(&self, output: &str, requested: &str) -> process::RawProcessReport {
        self.run_inputs(output, requested, &[])
    }

    fn run_inputs(
        &self,
        output: &str,
        requested: &str,
        overrides: &[(&str, String)],
    ) -> process::RawProcessReport {
        invoke(&self.root, self.environment(output, requested, overrides))
    }

    fn environment(
        &self,
        output: &str,
        requested: &str,
        overrides: &[(&str, String)],
    ) -> BTreeMap<OsString, Value> {
        let mut environment = BTreeMap::new();
        for (key, value) in [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            (
                "MESH_LLM_AUTOMATION_BIN",
                self.root.join("bin/automation").display().to_string(),
            ),
            ("PRODUCT_REAL_XTASK", env!("CARGO_BIN_EXE_xtask").into()),
            (
                "PRODUCT_EVENTS",
                self.root.join("events").display().to_string(),
            ),
            ("GITHUB_WORKSPACE", self.root.display().to_string()),
            (
                "GITHUB_OUTPUT",
                self.root.join("github-output").display().to_string(),
            ),
            ("INPUT_HOST_INPUT_DIR", "inputs/host".into()),
            ("INPUT_RUNTIME_INPUT_DIR", "inputs/runtime".into()),
            ("INPUT_OUTPUT_DIR", output.into()),
            ("INPUT_BACKEND", "cpu".into()),
            ("INPUT_BINARY_NAME", "mesh-llm".into()),
            ("INPUT_READINESS_SMOKE", "false".into()),
            ("INPUT_VERSION", requested.into()),
            ("COPYFILE_DISABLE", "1".into()),
            ("LC_ALL", "C".into()),
        ] {
            environment.insert(key.into(), Value::Public(value.into()));
        }
        for (key, value) in overrides {
            environment.insert((*key).into(), Value::Public(value.clone().into()));
        }
        environment
    }

    fn events(&self) -> Vec<String> {
        fs::read_to_string(self.root.join("events"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn refusal(&self, result: &process::RawProcessReport, before: &Snapshot) {
        assert!(
            !result.process.status.unwrap().success(),
            "unexpected acceptance"
        );
        assert_eq!(snapshot(&self.root.join("inputs")), *before);
        assert_eq!(
            fs::read(self.root.join("product.tar.gz")).unwrap(),
            OLD_ARCHIVE
        );
        assert_eq!(
            fs::read_to_string(self.root.join("github-output")).unwrap(),
            OLD_RECEIPT
        );
        assert_eq!(
            fs::read(self.outside.join("sentinel")).unwrap(),
            b"outside keep"
        );
        assert_eq!(
            fs::read(self.root.join("sentinel")).unwrap(),
            b"workspace keep"
        );
        assert!(
            !self
                .events()
                .iter()
                .any(|line| line == "product compose" || line.starts_with("forbidden"))
        );
    }

    fn accepted(&self, result: &process::RawProcessReport, before: &Snapshot) {
        assert!(
            result.process.status.unwrap().success(),
            "{}",
            String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes())
        );
        assert_eq!(snapshot(&self.root.join("inputs")), *before);
        let events = self.events();
        assert!(!events.iter().any(|line| line.starts_with("forbidden")));
        let compose = events
            .iter()
            .position(|line| line == "product compose")
            .unwrap();
        assert!(
            events
                .iter()
                .position(|line| line == "native verify-runtime-package")
                .unwrap()
                < compose
        );
        assert!(
            events
                .iter()
                .position(|line| line == "host-version")
                .unwrap()
                < compose
        );
        let manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(self.root.join("product/product-manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["mesh_version"], "1.2.3");
        assert_eq!(manifest["backend"], "cpu");
        assert_eq!(
            manifest["host"]["sha256"],
            digest(&fs::read(self.root.join("inputs/host/mesh-llm")).unwrap())
        );
        assert_eq!(manifest["runtime"]["id"], "runtime");
        assert_eq!(
            fs::read(
                self.root
                    .join("product/native-runtimes/runtime/lib/runtime.bin")
            )
            .unwrap(),
            b"inert runtime library"
        );
    }

    fn archive(&self, mode: &str) {
        let directory = self.root.join("inputs/runtime/runtime");
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        for (relative, (bytes, permissions)) in snapshot(&directory) {
            tar_member(
                &mut encoder,
                &format!("runtime/{}", relative.display()),
                &bytes,
                permissions,
            );
        }
        encoder.write_all(&[0; 1024]).unwrap();
        let bytes = encoder.finish().unwrap();
        let parent = self.root.join("inputs/runtime");
        fs::remove_dir_all(directory).unwrap();
        let archive = parent.join("runtime.tar.gz");
        fs::write(&archive, &bytes).unwrap();
        if mode != "missing" {
            let sidecar = match mode {
                "corrupt" => format!("{}  runtime.tar.gz\n", "0".repeat(64)),
                "wrong-name" => format!("{}  other.tar.gz\n", digest(&bytes)),
                "multiline" => format!(
                    "{}  runtime.tar.gz\n{}  runtime.tar.gz\n",
                    digest(&bytes),
                    digest(&bytes)
                ),
                _ => format!("{}  runtime.tar.gz\n", digest(&bytes)),
            };
            fs::write(parent.join("runtime.tar.gz.sha256"), sidecar).unwrap();
        }
        match mode {
            "duplicate" => {
                fs::write(parent.join("extra.tar.gz.sha256"), b"stray").unwrap();
            }
            "two-archives" => {
                fs::copy(&archive, parent.join("other.tar.gz")).unwrap();
            }
            "orphan" => {
                fs::remove_file(archive).unwrap();
            }
            _ => {}
        }
    }
}

fn write_runtime(directory: &Path) {
    fs::create_dir_all(directory.join("lib")).unwrap();
    fs::create_dir_all(directory.join("tools")).unwrap();
    let library = b"inert runtime library";
    let tool = "#!/bin/sh\nprintf 'forbidden runtime tool\\n' >> \"$PRODUCT_EVENTS\"\nexit 98\n";
    fs::write(directory.join("lib/runtime.bin"), library).unwrap();
    executable(&directory.join("tools/mesh-runtime-bench"), tool);
    let manifest = serde_json::json!({
        "runtime": {
            "id":"runtime", "mesh_version":"1.2.3", "skippy_abi":"0.1.0",
            "platform":{"os":"macos","arch":"x86_64","target":"x86_64-apple-darwin"},
            "backend":{"kind":"cpu"}, "libraries":["lib/runtime.bin"],
            "files":{"lib/runtime.bin":digest(library)},
            "tools":{"tools/mesh-runtime-bench":digest(tool.as_bytes())}
        },
        "build":{"backend":"cpu","primary_library":"lib/runtime.bin","library_sha256":digest(library)}
    });
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

fn tar_member(writer: &mut impl Write, name: &str, bytes: &[u8], mode: u32) {
    assert!(name.len() < 100);
    let mut header = [0_u8; 512];
    header[..name.len()].copy_from_slice(name.as_bytes());
    header[100..108].copy_from_slice(format!("{mode:07o}\0").as_bytes());
    header[108..116].copy_from_slice(b"0000000\0");
    header[116..124].copy_from_slice(b"0000000\0");
    header[124..136].copy_from_slice(format!("{:011o}\0", bytes.len()).as_bytes());
    header[136..148].copy_from_slice(b"00000000000\0");
    header[148..156].fill(b' ');
    header[156] = b'0';
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    let sum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    header[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    writer.write_all(&header).unwrap();
    writer.write_all(bytes).unwrap();
    writer
        .write_all(&vec![0; (512 - bytes.len() % 512) % 512])
        .unwrap();
}

fn archive_modes(path: &Path) -> BTreeMap<String, u32> {
    let mut decoded = Vec::new();
    GzDecoder::new(fs::File::open(path).unwrap())
        .read_to_end(&mut decoded)
        .unwrap();
    let field = |bytes: &[u8]| {
        std::str::from_utf8(bytes)
            .unwrap()
            .trim_matches('\0')
            .trim()
            .to_owned()
    };
    let mut offset = 0;
    let mut entries = BTreeMap::new();
    while offset + 512 <= decoded.len() {
        let header = &decoded[offset..offset + 512];
        if header.iter().all(|byte| *byte == 0) {
            break;
        }
        let size = usize::from_str_radix(&field(&header[124..136]), 8).unwrap();
        let name = field(&header[..100]);
        let prefix = field(&header[345..500]);
        if header[156] == b'0' || header[156] == 0 {
            let full = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            entries.insert(
                full.trim_start_matches("./").to_owned(),
                u32::from_str_radix(&field(&header[100..108]), 8).unwrap(),
            );
        }
        offset += 512 + size.div_ceil(512) * 512;
        assert!(offset <= decoded.len(), "truncated fixture product archive");
    }
    entries
}

#[test]
fn product_composition_adapter_compares_release_version_without_discarding_drift() {
    for (version, requested, accepted) in [
        ("1.2.3", "", true),
        ("1.2.3+gABC123", "1.2.3", true),
        ("1.2.3+gABC123.dirty", "v1.2.3", true),
        ("9.9.9", "", false),
        ("9.9.9+gABC123", "1.2.3", false),
    ] {
        let fixture = Fixture::new(version);
        let before = snapshot(&fixture.root.join("inputs"));
        let result = fixture.run_version("product", requested);
        if accepted {
            fixture.accepted(&result, &before);
        } else {
            fixture.refusal(&result, &before);
            assert!(fixture.events().iter().any(|line| line == "host-version"));
            assert!(!fixture.root.join("product/product-manifest.json").exists());
        }
    }
}

#[test]
fn product_composition_adapter_selects_exact_runtime_archive_and_sidecar_before_publication() {
    for mode in [
        "valid",
        "missing",
        "duplicate",
        "corrupt",
        "wrong-name",
        "multiline",
        "two-archives",
        "orphan",
    ] {
        let fixture = Fixture::new("1.2.3");
        fixture.archive(mode);
        let before = snapshot(&fixture.root.join("inputs"));
        let result = fixture.run("product");
        if mode == "valid" {
            fixture.accepted(&result, &before);
            let events = fixture.events();
            let extraction = events
                .iter()
                .position(|line| line == "artifact extract-tar")
                .unwrap();
            assert!(
                events
                    .iter()
                    .filter(|line| *line == "native verify-runtime-package")
                    .count()
                    >= 2
            );
            assert!(
                events
                    .iter()
                    .position(|line| line == "native verify-runtime-package")
                    .unwrap()
                    < extraction
            );
            let directory_verification = events
                .iter()
                .rposition(|line| line == "native verify-runtime-package")
                .unwrap();
            assert!(extraction < directory_verification);
            assert!(
                directory_verification
                    < events
                        .iter()
                        .position(|line| line == "host-version")
                        .unwrap()
            );
        } else {
            fixture.refusal(&result, &before);
            assert!(
                !fixture
                    .events()
                    .iter()
                    .any(|line| line == "artifact extract-tar" || line == "host-version")
            );
            assert!(!fixture.root.join("product/product-manifest.json").exists());
        }
    }
}

#[test]
fn product_composition_adapter_publishes_executable_host_and_tool_archive_modes() {
    for archived in [false, true] {
        let fixture = Fixture::new("1.2.3+gABC123.dirty");
        if archived {
            fixture.archive("valid");
        }
        let before = snapshot(&fixture.root.join("inputs"));
        let result = fixture.run("product");
        fixture.accepted(&result, &before);
        let modes = archive_modes(&fixture.root.join("product.tar.gz"));
        for path in [
            "mesh-llm",
            "native-runtimes/runtime/tools/mesh-runtime-bench",
        ] {
            assert_ne!(modes.get(path).unwrap() & 0o111, 0, "{path}");
        }
        let receipt = fs::read_to_string(fixture.root.join("github-output")).unwrap();
        let fields: BTreeMap<_, _> = receipt
            .lines()
            .skip(1)
            .map(|line| line.split_once('=').unwrap())
            .collect();
        for (key, relative) in [
            ("product_dir", "product"),
            ("binary_path", "product/mesh-llm"),
            ("runtime_root", "product/native-runtimes"),
            ("runtime_dir", "product/native-runtimes/runtime"),
            ("archive_path", "product.tar.gz"),
        ] {
            assert_eq!(
                fields.get(key).copied(),
                Some(fixture.root.join(relative).to_str().unwrap())
            );
        }
        assert_eq!(fields.len(), 5);
    }
}

#[test]
fn product_composition_adapter_refuses_destructive_and_escaped_destinations_before_mutation() {
    for case in 0..11 {
        let fixture = Fixture::new("1.2.3");
        symlink(&fixture.outside, fixture.root.join("escaped-output")).unwrap();
        let output = match case {
            0 => ".".into(),
            1 => "./".into(),
            2 => "product/..".into(),
            3 => fixture.root.display().to_string(),
            4 => fixture.outside.display().to_string(),
            5 => "inputs/host".into(),
            6 => "inputs/host/product".into(),
            7 => "inputs".into(),
            8 => "inputs/runtime".into(),
            9 => "inputs/runtime/product".into(),
            _ => "escaped-output/product".into(),
        };
        let before = snapshot(&fixture.root.join("inputs"));
        let result = fixture.run(&output);
        fixture.refusal(&result, &before);
        assert_eq!(fixture.events(), ["product canonical-inputs"]);
        assert_eq!(
            fs::read(fixture.root.join("product/previous-staging")).unwrap(),
            b"staging keep"
        );
        assert!(!fixture.root.join("product/product-manifest.json").exists());
    }
}

#[test]
fn product_composition_adapter_rejects_noncanonical_host_checksum_before_staging_mutation() {
    for multiline in [false, true] {
        let fixture = Fixture::new("1.2.3");
        let host = fs::read(fixture.root.join("inputs/host/mesh-llm")).unwrap();
        let checksum = digest(&host);
        let text = if multiline {
            format!("{checksum}  mesh-llm\n{checksum}  mesh-llm\n")
        } else {
            format!("{checksum}  other-host\n")
        };
        fs::write(fixture.root.join("inputs/host/mesh-llm.sha256"), text).unwrap();
        let before = snapshot(&fixture.root.join("inputs"));
        let result = fixture.run("product");
        fixture.refusal(&result, &before);
        assert_eq!(
            fixture.events(),
            ["product canonical-inputs", "artifact verify-checksum"]
        );
        assert_eq!(
            fs::read(fixture.root.join("product/previous-staging")).unwrap(),
            b"staging keep"
        );
    }
}

#[path = "product_composition_adapter/attestation.rs"]
mod attestation;
#[path = "product_composition_adapter/windows_paths.rs"]
mod windows_paths;

#[path = "product_composition_adapter/host_attestation.rs"]
mod host_attestation;
#[path = "product_composition_adapter/model_restore.rs"]
mod model_restore;
#[path = "product_composition_adapter/readiness.rs"]
mod readiness;
#[path = "product_composition_adapter/restore_product.rs"]
mod restore_product;
