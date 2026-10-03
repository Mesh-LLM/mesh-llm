use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use flate2::{Compression, write::GzEncoder};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
}
impl Fixture {
    fn new(extra_upload: bool) -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("SDK restore with spaces");
        for name in [
            "scripts/lib",
            "bin",
            "tmp",
            "home",
            "source/meshllm-native-linux-x86_64-cpu/lib",
            "download",
        ] {
            fs::create_dir_all(root.join(name)).unwrap();
        }
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for name in [
            "scripts/restore-native-sdk-input.sh",
            "scripts/verify-native-sdk-package.sh",
            "scripts/verify-native-runtime-package.sh",
            "scripts/lib/automation.sh",
        ] {
            fs::copy(repository.join(name), root.join(name)).unwrap();
        }
        let uname = root.join("bin/uname");
        fs::write(&uname, "#!/bin/sh\nprintf 'x86_64\\n'\n").unwrap();
        fs::set_permissions(&uname, fs::Permissions::from_mode(0o755)).unwrap();
        let artifact = root.join("source/meshllm-native-linux-x86_64-cpu");
        let bytes = b"inert native SDK bytes";
        fs::write(artifact.join("lib/libmesh_llm_ffi.so"), bytes).unwrap();
        fs::write(artifact.join("lib/libmesh_llm_uniffi.so"), bytes).unwrap();
        let manifest = json!({
            "schema_version":1,"artifact_id":"meshllm-native-linux-x86_64-cpu",
            "native_runtime_id":"meshllm-native-linux-x86_64-cpu",
            "sdk_version":"0.75.0","mesh_version":"0.75.0",
            "target_triple":"x86_64-unknown-linux-gnu","platform":"linux-x86_64",
            "os":"linux","arch":"x86_64","backend":"cpu","flavor":"cpu","cargo_profile":"debug",
            "library":"lib/libmesh_llm_ffi.so","library_paths":["lib/libmesh_llm_ffi.so"],
            "uniffi_library":"lib/libmesh_llm_uniffi.so",
            "library_sha256":hex::encode(Sha256::digest(bytes)),"requirements":[],
            "features":["mesh-inference","model-management","local-serving","chat","responses"]
        });
        fs::write(
            artifact.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let archive = root.join("download/sdk.tar.gz");
        let encoder = GzEncoder::new(fs::File::create(&archive).unwrap(), Compression::default());
        let mut builder = tar::Builder::new(encoder);
        builder
            .append_dir_all("meshllm-native-linux-x86_64-cpu", &artifact)
            .unwrap();
        builder.into_inner().unwrap().finish().unwrap();
        let digest = hex::encode(Sha256::digest(fs::read(&archive).unwrap()));
        fs::write(
            root.join("download/sdk.tar.gz.sha256"),
            format!("{digest}  sdk.tar.gz\n"),
        )
        .unwrap();
        if extra_upload {
            fs::write(root.join("download/unexpected"), b"not admitted").unwrap();
        }
        fs::write(root.join("outside-sentinel"), b"keep").unwrap();
        Self {
            _temporary: temporary,
            root,
        }
    }
    fn run(&self, target: &str, profile: &str) -> process::RawProcessReport {
        let arguments = [
            self.root
                .join("scripts/restore-native-sdk-input.sh")
                .display()
                .to_string(),
            self.root.join("download").display().to_string(),
            self.root.join("restored").display().to_string(),
            target.into(),
            "cpu".into(),
            profile.into(),
        ]
        .into_iter()
        .map(|value| Value::Public(value.into()))
        .collect();
        self.command(arguments)
    }
    fn verify(&self, sdk: bool, artifact: &Path) -> process::RawProcessReport {
        let script = if sdk {
            "verify-native-sdk-package.sh"
        } else {
            "verify-native-runtime-package.sh"
        };
        let mut words = vec![self.root.join("scripts").join(script).display().to_string()];
        if !sdk {
            words.push("--portable".into());
        }
        words.push(artifact.display().to_string());
        self.command(
            words
                .into_iter()
                .map(|word| Value::Public(word.into()))
                .collect(),
        )
    }
    fn command(&self, arguments: Vec<Value>) -> process::RawProcessReport {
        let environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            ("TMPDIR", self.root.join("tmp").display().to_string()),
            ("HOME", self.root.join("home").display().to_string()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                env!("CARGO_BIN_EXE_xtask").to_owned(),
            ),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value.into())))
        .collect();
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                arguments,
                environment,
            },
            &Limits {
                execution: Duration::from_secs(10),
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
        assert_eq!(report.process.outcome, Outcome::Exited);
        assert!(report.process.failure.is_none());
        assert!(report.process.cleanup.complete);
        assert_eq!(
            fs::read(self.root.join("outside-sentinel")).unwrap(),
            b"keep"
        );
        report
    }
}

#[test]
fn native_sdk_restore_actual_wrapper_accepts_verified_typed_archive() {
    let fixture = Fixture::new(false);
    let report = fixture.run("x86_64-unknown-linux-gnu", "debug");
    assert!(
        report.process.status.unwrap().success(),
        "{}",
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
    );
    let expected = fixture
        .root
        .join("restored/meshllm-native-linux-x86_64-cpu");
    assert_eq!(
        report.stdout.unwrap().as_bytes(),
        format!("{}\n", expected.display()).as_bytes()
    );
    assert!(expected.join("manifest.json").is_file());
    assert_eq!(
        fs::read(expected.join("lib/libmesh_llm_ffi.so")).unwrap(),
        b"inert native SDK bytes"
    );
}

#[test]
fn native_sdk_restore_actual_wrapper_refuses_runner_profile_and_extra_upload() {
    for (target, profile, extra, reason) in [
        (
            "aarch64-unknown-linux-gnu",
            "debug",
            false,
            "target/runner architecture mismatch",
        ),
        (
            "x86_64-unknown-linux-gnu",
            "release",
            false,
            "cargo_profile mismatch",
        ),
        (
            "x86_64-unknown-linux-gnu",
            "debug",
            true,
            "exactly one archive and checksum",
        ),
    ] {
        let fixture = Fixture::new(extra);
        let report = fixture.run(target, profile);
        assert!(!report.process.status.unwrap().success());
        assert!(
            report.stdout.unwrap().as_bytes().is_empty(),
            "refusal must not publish an accepted artifact path"
        );
        assert!(String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains(reason));
    }
}

fn runtime(fixture: &Fixture) -> (PathBuf, serde_json::Value) {
    let artifact = fixture
        .root
        .join("source/meshllm-native-runtime-darwin-x86_64-cpu");
    fs::create_dir_all(artifact.join("lib")).unwrap();
    fs::create_dir_all(artifact.join("tools")).unwrap();
    fs::write(artifact.parent().unwrap().join("outside-sentinel"), b"keep").unwrap();
    let library = b"inert runtime library";
    let tool = b"inert runtime tool";
    fs::write(artifact.join("lib/llama.bin"), library).unwrap();
    fs::write(artifact.join("tools/probe"), tool).unwrap();
    fs::set_permissions(
        artifact.join("tools/probe"),
        fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    let document = json!({"runtime":{
        "id":"meshllm-native-runtime-darwin-x86_64-cpu","mesh_version":"0.75.0","skippy_abi":"0.1.32",
        "platform":{"os":"macos","arch":"x86_64","target":"x86_64-apple-darwin"},"backend":{"kind":"cpu"},
        "libraries":["lib/llama.bin"],"files":{"lib/llama.bin":hex::encode(Sha256::digest(library))},
        "tools":{"tools/probe":hex::encode(Sha256::digest(tool))}},
        "build":{"primary_library":"lib/llama.bin","library_sha256":hex::encode(Sha256::digest(library))}});
    (artifact, document)
}
fn write_document(artifact: &Path, document: &serde_json::Value) -> Vec<u8> {
    let bytes = serde_json::to_vec(document).unwrap();
    fs::write(artifact.join("manifest.json"), &bytes).unwrap();
    bytes
}
fn admitted(report: process::RawProcessReport, expected: bool) {
    assert_eq!(
        report.process.status.unwrap().success(),
        expected,
        "{}",
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
    );
    if !expected {
        let output = String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes());
        assert!(
            !output.contains("verified native SDK artifact")
                && !output.contains("verified portable native runtime artifact")
        );
    }
}
fn mutate(document: &mut serde_json::Value, case: &str) {
    let outside = hex::encode(Sha256::digest(b"keep"));
    match case {
        "library escape" => {
            document["runtime"]["libraries"] = json!(["../outside-sentinel"]);
            document["runtime"]["files"] = json!({"../outside-sentinel":outside});
        }
        "file escape" => {
            document["runtime"]["files"]["../outside-sentinel"] = json!(outside);
        }
        "tool escape" => {
            document["runtime"]["tools"] = json!({"../outside-sentinel":outside});
        }
        "primary escape" => {
            document["build"]["primary_library"] = json!("../outside-sentinel");
            document["build"]["library_sha256"] = json!(outside);
        }
        "backslash" => {
            document["runtime"]["libraries"] = json!(["lib\\llama.bin"]);
        }
        "drive" => {
            document["build"]["primary_library"] = json!("C:/outside-sentinel");
        }
        "platform type" => {
            document["runtime"]["platform"]["os"] = json!(["macos"]);
        }
        "target relation" => {
            document["runtime"]["platform"]["target"] = json!("aarch64-unknown-linux-gnu");
        }
        "backend relation" => {
            document["runtime"]["backend"]["kind"] = json!("cuda");
        }
        "missing files" => {
            document["runtime"].as_object_mut().unwrap().remove("files");
        }
        "missing library digest" => {
            document["runtime"]["files"] = json!({});
        }
        "missing primary" => {
            document["build"]
                .as_object_mut()
                .unwrap()
                .remove("primary_library");
        }
        "missing primary digest" => {
            document["build"]
                .as_object_mut()
                .unwrap()
                .remove("library_sha256");
        }
        _ => panic!("unknown semantic mutation"),
    }
}

#[test]
fn native_runtime_verifier_manifest_consumption_preserves_control_and_refuses_mutations() {
    let fixture = Fixture::new(false);
    let (artifact, valid) = runtime(&fixture);
    let mut cases = vec!["valid"];
    cases.extend([
        "library escape",
        "file escape",
        "tool escape",
        "primary escape",
        "backslash",
        "drive",
        "platform type",
        "target relation",
        "backend relation",
        "missing files",
        "missing library digest",
        "missing primary",
        "missing primary digest",
    ]);
    for case in cases {
        let mut document = valid.clone();
        if case != "valid" {
            mutate(&mut document, case);
        }
        let before = write_document(&artifact, &document);
        admitted(fixture.verify(false, &artifact), case == "valid");
        assert_eq!(fs::read(artifact.join("manifest.json")).unwrap(), before);
        assert_eq!(
            fs::read(artifact.parent().unwrap().join("outside-sentinel")).unwrap(),
            b"keep"
        );
        assert_eq!(
            fs::read(artifact.join("lib/llama.bin")).unwrap(),
            b"inert runtime library"
        );
    }
}

#[test]
fn native_runtime_verifier_rejects_resolved_manifest_escape_without_touching_target() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new(false);
    let (artifact, mut document) = runtime(&fixture);
    symlink(
        fixture.root.join("outside-sentinel"),
        artifact.join("lib/escape.bin"),
    )
    .unwrap();
    document["runtime"]["libraries"] = json!(["lib/escape.bin"]);
    document["runtime"]["files"] = json!({"lib/escape.bin":hex::encode(Sha256::digest(b"keep"))});
    document["build"]["primary_library"] = json!("lib/escape.bin");
    document["build"]["library_sha256"] = json!(hex::encode(Sha256::digest(b"keep")));
    let before = write_document(&artifact, &document);
    admitted(fixture.verify(false, &artifact), false);
    assert_eq!(fs::read(artifact.join("manifest.json")).unwrap(), before);
    assert_eq!(
        fs::read(artifact.parent().unwrap().join("outside-sentinel")).unwrap(),
        b"keep"
    );
}

fn archive(fixture: &Fixture, artifact: &Path, shape: &str, label: &str) -> PathBuf {
    let path = fixture.root.join(format!("{label}-{shape}.tar.gz"));
    let encoder = GzEncoder::new(fs::File::create(&path).unwrap(), Compression::default());
    let mut builder = tar::Builder::new(encoder);
    if shape != "single-file" {
        builder
            .append_dir_all(artifact.file_name().unwrap(), artifact)
            .unwrap();
    }
    if shape != "valid" {
        let bytes = b"unwanted top-level payload";
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "unexpected.txt", &bytes[..])
            .unwrap();
    }
    builder.into_inner().unwrap().finish().unwrap();
    let digest = hex::encode(Sha256::digest(fs::read(&path).unwrap()));
    fs::write(
        path.with_file_name(format!(
            "{}.sha256",
            path.file_name().unwrap().to_string_lossy()
        )),
        format!(
            "{digest}  {}\n",
            path.file_name().unwrap().to_string_lossy()
        ),
    )
    .unwrap();
    path
}
#[test]
fn native_runtime_and_sdk_archive_verifiers_accept_one_artifact_directory() {
    let fixture = Fixture::new(false);
    let (runtime, document) = runtime(&fixture);
    write_document(&runtime, &document);
    for (sdk, artifact, label) in [
        (false, runtime, "runtime"),
        (
            true,
            fixture.root.join("source/meshllm-native-linux-x86_64-cpu"),
            "sdk",
        ),
    ] {
        let package = archive(&fixture, &artifact, "valid", label);
        let bytes = fs::read(&package).unwrap();
        admitted(fixture.verify(sdk, &package), true);
        assert_eq!(fs::read(&package).unwrap(), bytes);
    }
}
#[test]
fn native_runtime_and_sdk_archive_verifiers_refuse_sibling_and_single_file_payloads() {
    let fixture = Fixture::new(false);
    let (runtime, document) = runtime(&fixture);
    write_document(&runtime, &document);
    for (sdk, artifact, label) in [
        (false, runtime, "runtime"),
        (
            true,
            fixture.root.join("source/meshllm-native-linux-x86_64-cpu"),
            "sdk",
        ),
    ] {
        for shape in ["sibling", "single-file"] {
            let package = archive(&fixture, &artifact, shape, label);
            let bytes = fs::read(&package).unwrap();
            admitted(fixture.verify(sdk, &package), false);
            assert_eq!(fs::read(&package).unwrap(), bytes);
        }
    }
}
