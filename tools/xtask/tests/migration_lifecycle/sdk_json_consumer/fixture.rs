use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn executable(path: &Path, body: &str) {
    fs::write(path, format!("#!/bin/bash\nset -euo pipefail\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
pub(super) struct Fixture {
    _directory: tempfile::TempDir,
    pub root: PathBuf,
}
impl Fixture {
    pub fn new(report: &serde_json::Value) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory
            .path()
            .canonicalize()
            .unwrap()
            .join("workspace with spaces");
        for dir in [
            "scripts/lib",
            "skippy/scripts",
            "bin",
            "tmp",
            "host-input",
            "runtime-input/runtime/lib",
            "skippy/crates/skippy-ffi/src",
        ] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        for file in [
            "scripts/ci-compose-product-input.sh",
            "scripts/ci-prepare-native-runtime.sh",
            "scripts/verify-native-runtime-package.sh",
            "skippy/scripts/verify-native-runtime-package.sh",
            "scripts/lib/automation.sh",
            "skippy/crates/skippy-ffi/src/lib.rs",
        ] {
            fs::copy(repository().join(file), root.join(file)).unwrap();
        }
        executable(
            &root.join("scripts/ci-client-readiness-smoke.sh"),
            "test -f \"$GITHUB_WORKSPACE/sdk-reader-ran\"\nprintf 'readiness\\n' >> \"$GITHUB_WORKSPACE/events\"",
        );
        executable(
            &root.join("scripts/package-native-runtime.sh"),
            "touch \"$GITHUB_WORKSPACE/fallback-ran\"; exit 99",
        );
        executable(&root.join("bin/uname"), "printf 'Linux\\n'");
        // The verifier requires readelf availability even for non-ELF inert bytes.
        // Any inspection attempt must fail this finite caller fixture.
        for name in ["cargo", "just", "cmake", "readelf"] {
            executable(
                &root.join("bin").join(name),
                "touch \"$GITHUB_WORKSPACE/forbidden\"; exit 98",
            );
        }
        let ffi =
            fs::read_to_string(repository().join("skippy/crates/skippy-ffi/src/lib.rs")).unwrap();
        let abi = ["MAJOR", "MINOR", "PATCH"]
            .map(|part| {
                let line = ffi
                    .lines()
                    .find(|line| line.starts_with(&format!("pub const ABI_VERSION_{part}:")))
                    .unwrap();
                line.split('=')
                    .nth(1)
                    .unwrap()
                    .trim()
                    .trim_end_matches(';')
                    .to_owned()
            })
            .join(".");
        let host = "#!/bin/bash\nset -euo pipefail\nif [[ \"$*\" == '--log-format json --print-build-contract' ]]; then\n  printf '%s\\n' '{\"schema_version\":1,\"product_version\":\"1.0.0\",\"runtime_release\":\"9.0.0\",\"skippy_abi\":\"$FIXTURE_ABI\"}'\n  exit 0\nfi\nif [[ \" $* \" == *' --available '* ]]; then\n  [[ \" $* \" == *' --json '* && \" $* \" == *' --log-format json '* ]] || exit 97\n  [[ \"$MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK\" == 0 ]] || exit 97\n  [[ -z \"${MESH_LLM_CONFIG+x}\" && -z \"${MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR+x}\" && -z \"${MESH_LLM_NATIVE_RUNTIME_CACHE_DIR+x}\" ]] || exit 97\n  [[ \"$HOME\" != \"$GITHUB_WORKSPACE/ambient-home\" ]] || exit 97\n  touch \"$GITHUB_WORKSPACE/sdk-reader-ran\"\n  printf 'sdk\\n' >> \"$GITHUB_WORKSPACE/events\"\n  cat \"$GITHUB_WORKSPACE/report.json\"\nelse\n  printf 'mesh-llm 1.0.0\\n'\nfi\n".replace("$FIXTURE_ABI", &abi);
        fs::write(root.join("host-input/mesh-llm"), &host).unwrap();
        fs::write(
            root.join("host-input/mesh-llm.sha256"),
            format!(
                "{}  mesh-llm\n",
                hex::encode(Sha256::digest(host.as_bytes()))
            ),
        )
        .unwrap();
        fs::write(root.join("host-input/host-imports.json"), "{}").unwrap();
        let library = b"inert runtime fixture";
        fs::write(root.join("runtime-input/runtime/lib/runtime.bin"), library).unwrap();
        let digest = hex::encode(Sha256::digest(library));
        fs::write(root.join("runtime-input/runtime/manifest.json"), serde_json::to_vec(&json!({
            "schema_version":2,"runtime":{"id":"runtime","release_version":"9.0.0","skippy_abi":abi,
            "platform":{"os":"linux","arch":"x86_64","target":"x86_64-unknown-linux-gnu"},
            "backend":{"kind":"cpu"},"libraries":["lib/runtime.bin"],"files":{"lib/runtime.bin":digest},"tools":{}},
            "build":{"backend":"cpu","primary_library":"lib/runtime.bin","library_sha256":digest}
        })).unwrap()).unwrap();
        fs::write(
            root.join("report.json"),
            serde_json::to_vec(report).unwrap(),
        )
        .unwrap();
        Self {
            _directory: directory,
            root,
        }
    }
    pub fn run(&self) -> process::RawProcessReport {
        let environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            ("HOME", self.root.join("ambient-home").display().to_string()),
            ("RUNNER_TEMP", self.root.join("tmp").display().to_string()),
            ("GITHUB_WORKSPACE", self.root.display().to_string()),
            (
                "GITHUB_OUTPUT",
                self.root.join("outputs").display().to_string(),
            ),
            (
                "MESH_LLM_AUTOMATION_BIN",
                env!("CARGO_BIN_EXE_xtask").into(),
            ),
            ("MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK", "1".into()),
            ("MESH_LLM_CONFIG", "ambient-config".into()),
            (
                "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
                "ambient-bundle".into(),
            ),
            ("MESH_LLM_NATIVE_RUNTIME_CACHE_DIR", "ambient-cache".into()),
            ("INPUT_HOST_INPUT_DIR", "host-input".into()),
            ("INPUT_RUNTIME_INPUT_DIR", "runtime-input".into()),
            ("INPUT_OUTPUT_DIR", "product".into()),
            ("INPUT_BACKEND", "cpu".into()),
            ("INPUT_BINARY_NAME", "mesh-llm".into()),
            ("INPUT_READINESS_SMOKE", "true".into()),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value.into())))
        .collect();
        capture(ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: self.root.clone(),
            environment,
            arguments: vec![Value::Public(
                self.root
                    .join("scripts/ci-compose-product-input.sh")
                    .into_os_string(),
            )],
        })
    }
}
pub(super) fn capture(spec: ProcessSpec) -> process::RawProcessReport {
    let output = process::supervise_raw(
        &spec,
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
    assert!(output.process.failure.is_none(), "{:?}", output.process);
    assert!(output.process.cleanup.complete, "{:?}", output.process);
    output
}
