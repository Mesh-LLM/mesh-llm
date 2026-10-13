//! Sourced release adapter contracts; finite inert inputs and real typed owners.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory
            .path()
            .canonicalize()
            .unwrap()
            .join("release source with spaces");
        for relative in ["scripts/lib", "bin", "immutable host input"] {
            fs::create_dir_all(root.join(relative)).unwrap();
        }
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for relative in [
            "scripts/package-release.sh",
            "mesh/scripts/package-release.sh",
            "scripts/lib/automation.sh",
        ] {
            fs::create_dir_all(root.join(relative).parent().unwrap()).unwrap();
            fs::copy(source.join(relative), root.join(relative)).unwrap();
        }
        fs::write(root.join("Justfile"), "# finite fixture facade\n").unwrap();
        executable(
            &root.join("bin/just"),
            r#"#!/bin/bash
set -euo pipefail
[[ "$1" == --justfile && "$2" == "$PWD/Justfile" && "$3" == automation-run ]] || exit 92
printf '%s\0' "$@" >> "$PWD/just.calls"
shift 3
exec "$PACKAGE_FIXTURE_OWNER" "$@"
"#,
        );
        executable(
            &root.join("bin/cargo"),
            "#!/bin/sh\nprintf forbidden > \"$PWD/cargo.called\"\nexit 93\n",
        );
        Self {
            _directory: directory,
            root,
        }
    }
    fn runtime(&self, relative: &str, id: &str, kind: &str, build: &str) -> PathBuf {
        let path = self.root.join(relative).join(id);
        fs::create_dir_all(path.join("lib")).unwrap();
        fs::write(path.join("lib/libllama.so"), b"inert runtime bytes").unwrap();
        let document = json!({
            "schema_version":2,
            "runtime": {"id":id,"release_version":"9.0.0","skippy_abi":"0.1.0",
                "platform":{"os":"linux","arch":"x86_64","target":"x86_64-unknown-linux-gnu"},
                "backend":{"kind":kind},"rank":0,"libraries":["lib/libllama.so"],
                "url":null,"sha256":null,"signature":null},
            "build":{"backend":build}
        });
        fs::write(
            path.join("manifest.json"),
            serde_json::to_vec(&document).unwrap(),
        )
        .unwrap();
        path
    }
    fn invoke(
        &self,
        statement: &str,
        flavor: &str,
        fallback: bool,
        prestamped: bool,
        key: Option<&Path>,
    ) -> process::RawProcessReport {
        let mut environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            ("HOME", self.root.join("home").display().to_string()),
            (
                "PACKAGE_FIXTURE_OWNER",
                env!("CARGO_BIN_EXE_xtask").to_owned(),
            ),
            (
                "MESH_LLM_NATIVE_RUNTIME_ROOT",
                self.root.join("runtimes").display().to_string(),
            ),
            (
                "MESH_LLM_RELEASE_BIN_DIR",
                self.root.join("immutable host input").display().to_string(),
            ),
            ("MESH_RELEASE_OS", "Linux".to_owned()),
            ("MESH_RELEASE_ARCH", "x86_64".to_owned()),
            ("MESH_RELEASE_FLAVOR", flavor.to_owned()),
            (
                "MESH_RELEASE_HOST_PRESTAMPED",
                if prestamped { "1" } else { "0" }.to_owned(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        if !fallback {
            environment.insert(
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            );
        }
        if let Some(key) = key {
            environment.insert(
                "MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE".into(),
                Value::Public(key.as_os_str().to_owned()),
            );
        }
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                environment,
                arguments: [
                    "-c",
                    &format!("source scripts/package-release.sh\n{statement}"),
                ]
                .into_iter()
                .map(|a| Value::Public(a.into()))
                .collect(),
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
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(result.process.failure.is_none(), "{:?}", result.process);
        assert!(result.process.cleanup.complete, "{:?}", result.process);
        assert!(
            !self.root.join("cargo.called").exists(),
            "fixture must never build"
        );
        result
    }
}
fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn stdout(report: &process::RawProcessReport) -> &str {
    std::str::from_utf8(report.stdout.as_ref().unwrap().as_bytes()).unwrap()
}
fn stderr(report: &process::RawProcessReport) -> &str {
    std::str::from_utf8(report.stderr.as_ref().unwrap().as_bytes()).unwrap()
}

#[test]
fn package_release_adapter_selects_exact_runtime_and_rejects_ambiguity_through_normal_just() {
    let fixture = Fixture::new();
    let selected = fixture.runtime("runtimes", "linux-cpu", "cpu", "cpu");
    fixture.runtime("runtimes", "linux-vulkan", "vulkan", "vulkan");
    let result = fixture.invoke("select_native_runtime_dir", "cpu", true, false, None);
    assert!(result.process.success(), "{}", stderr(&result));
    assert_eq!(Path::new(stdout(&result).trim()), selected);
    assert!(
        fixture.root.join("just.calls").is_file(),
        "normal Just authority must be exercised"
    );
    fixture.runtime("runtimes", "linux-cpu-other", "cpu", "cpu");
    let rejected = fixture.invoke("select_native_runtime_dir", "cpu", true, false, None);
    assert!(!rejected.process.success());
    assert!(stdout(&rejected).is_empty());
    assert!(stderr(&rejected).contains("expected exactly one native runtime"));
}

#[test]
fn package_release_adapter_composes_product_contract_aliases_and_rejects_backend_before_write() {
    for (id, kind, build, requested, accepted) in [
        ("linux-cpu", "cpu", "cpu", "cpu", true),
        (
            "linux-cuda13-sm120",
            "cuda",
            "cuda-blackwell",
            "cuda-blackwell",
            true,
        ),
        ("linux-rocm", "rocm", "hip", "hip", true),
        ("linux-vulkan", "vulkan", "vulkan", "cpu", false),
    ] {
        let fixture = Fixture::new();
        let bundle = fixture.root.join("mesh-bundle");
        fs::create_dir(&bundle).unwrap();
        let host = "#!/bin/sh\n[ \"$*\" = \"--log-format json --print-build-contract\" ] || exit 98\nprintf '%s\\n' '{\"schema_version\":1,\"product_version\":\"0.73.1\",\"runtime_release\":\"1.0.0\",\"skippy_abi\":\"0.1.0\"}'\n";
        executable(&bundle.join("mesh-llm"), host);
        let runtime = fixture.runtime("mesh-bundle/native-runtimes", id, kind, build);
        let command = format!(
            "write_product_manifest \"$PWD/mesh-bundle\" \"$PWD/mesh-bundle/mesh-llm\" \"$PWD/mesh-bundle/native-runtimes/{id}\" v0.73.1 {requested}"
        );
        let result = fixture.invoke(&command, requested, false, false, None);
        assert_eq!(result.process.success(), accepted, "{}", stderr(&result));
        assert!(
            !fixture.root.join("just.calls").exists(),
            "configured owner must not bootstrap"
        );
        let output = bundle.join("product-manifest.json");
        if accepted {
            let manifest: Json = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
            assert_eq!(manifest["schema_version"], 2);
            assert_eq!(manifest["contract"], "mesh-llm-product-v2");
            assert_eq!(manifest["mesh_version"], "0.73.1");
            assert_eq!(manifest["backend"], requested);
            assert_eq!(manifest["host"]["path"], "mesh-llm");
            assert_eq!(
                manifest["host"]["sha256"],
                hex::encode(Sha256::digest(host))
            );
            assert_eq!(manifest["runtime"]["id"], id);
            assert_eq!(manifest["runtime"]["release_version"], "9.0.0");
            assert_eq!(manifest["runtime"]["path"], format!("native-runtimes/{id}"));
            assert_eq!(
                manifest["runtime"]["manifest_sha256"],
                hex::encode(Sha256::digest(
                    fs::read(runtime.join("manifest.json")).unwrap()
                ))
            );
        } else {
            assert!(!output.exists());
            assert!(stderr(&result).contains("backend mismatch"));
        }
    }
}

#[test]
fn package_release_adapter_preserves_explicit_host_directory_and_prestamp_public_key_admission() {
    let fixture = Fixture::new();
    let command = "printf '%s\\n' \"$RELEASE_BIN_DIR\"; validate_attestation_env";
    let result = fixture.invoke(command, "cpu", false, false, None);
    assert!(result.process.success(), "{}", stderr(&result));
    assert_eq!(
        Path::new(stdout(&result).trim()),
        fixture.root.join("immutable host input")
    );
    let missing = fixture.invoke("validate_attestation_env", "cpu", false, true, None);
    assert!(!missing.process.success());
    assert!(stderr(&missing).contains("MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE"));
    let key = fixture.root.join("public key");
    fs::write(&key, b"fixture public key admission only").unwrap();
    let admitted = fixture.invoke("validate_attestation_env", "cpu", false, true, Some(&key));
    assert!(admitted.process.success(), "{}", stderr(&admitted));
    assert!(
        !fixture.root.join("just.calls").exists(),
        "source admission must precede owner work"
    );
}
