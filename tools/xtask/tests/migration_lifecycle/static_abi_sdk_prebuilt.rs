//! Actual SDK adapter + native directory query; all build boundaries are inert.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use serde_json::Value as Json;
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

fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("SDK private fixture");
        for relative in ["scripts/lib", "bin", "home", "tmp", ".deps/llama.cpp"] {
            fs::create_dir_all(root.join(relative)).unwrap();
        }
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for relative in [
            "scripts/package-native-sdk.sh",
            "mesh/scripts/package-native-sdk.sh",
            "skippy/scripts/build-llama.sh",
            "scripts/lib/cuda-toolkit.sh",
            "scripts/lib/macos-deployment-target.sh",
            "scripts/lib/macos-deployment-target.txt",
        ] {
            fs::create_dir_all(root.join(relative).parent().unwrap()).unwrap();
            fs::copy(source.join(relative), root.join(relative)).unwrap();
        }
        fs::copy(
            source.join("scripts/build-llama.sh"),
            root.join("scripts/query-build-llama.sh"),
        )
        .unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[workspace.package]\nversion = \"0.80.0\"\n",
        )
        .unwrap();
        executable(
            &root.join("scripts/prepare-llama.sh"),
            r#"#!/bin/bash
set -euo pipefail
printf 'prepare|%s\n' "$*" >> "$PWD/events"
printf '%s\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa > "$PWD/.deps/llama.cpp/.mesh-llm-patched-sha"
"#,
        );
        executable(
            &root.join("scripts/build-llama.sh"),
            r#"#!/bin/bash
set -euo pipefail
if [[ "$#" == 1 && "$1" == --print-build-dir ]]; then
  [[ -f "$PWD/.deps/llama.cpp/.mesh-llm-patched-sha" ]] || exit 91
  selected=$(/bin/bash "$PWD/scripts/query-build-llama.sh" --print-build-dir)
  printf 'query|%s|%s\n' "$LLAMA_STAGE_BACKEND" "$selected" >> "$PWD/events"
  printf '%s\n' "$selected"
  exit 0
fi
[[ "$LLAMA_BUILD_DIR" == "$LLAMA_STAGE_BUILD_DIR" ]] || exit 92
printf 'native|%s|%s|%s\n' "$LLAMA_STAGE_BACKEND" "$*" "$LLAMA_STAGE_BUILD_DIR" >> "$PWD/events"
[[ "$#" == 0 || ( "$#" == 1 && "$1" == --require-existing ) ]] || exit 93
if [[ "${SDK_FIXTURE_REFUSE:-0}" == 1 && "${1:-}" == --require-existing ]]; then
  printf 'inert existing ABI rejected\n' >&2
  exit 42
fi
"#,
        );
        executable(
            &root.join("bin/uname"),
            r#"#!/bin/sh
case "$1" in
-s) printf '%s\n' "$SDK_FIXTURE_OS" ;;
-m) printf 'x86_64\n' ;;
*) exit 94 ;;
esac
"#,
        );
        executable(
            &root.join("bin/cargo"),
            r#"#!/bin/bash
set -euo pipefail
if [[ "$1" == xtool ]]; then
  shift
  exec "$SDK_FIXTURE_OWNER" "$@"
fi
[[ "$1" == build ]] || exit 95
printf 'cargo|%s|%s|%s|%s\n' "$LLAMA_STAGE_BACKEND" "$SKIPPY_LLAMA_AUTO_BUILD" "$MESH_LLM_AUTO_BUILD_LLAMA" "$LLAMA_STAGE_BUILD_DIR" >> "$PWD/events"
mkdir -p "$PWD/target/release"
case "$SDK_FIXTURE_OS" in
Darwin) extension=dylib ;;
Linux) extension=so ;;
*) exit 96 ;;
esac
printf 'inert SDK bytes\n' > "$PWD/target/release/libmeshllm_ffi.$extension"
"#,
        );
        for tool in ["cmake", "ninja", "make", "nvcc", "sccache"] {
            executable(
                &root.join("bin").join(tool),
                "#!/bin/sh\nprintf forbidden > \"$PWD/native-tool-called\"\nexit 97\n",
            );
        }
        Self {
            _temporary: temporary,
            root,
        }
    }

    fn invoke(
        &self,
        arguments: &[String],
        backend: &str,
        refuse: bool,
    ) -> process::RawProcessReport {
        let os = if backend == "metal" {
            "Darwin"
        } else {
            "Linux"
        };
        let environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            ("HOME", self.root.join("home").display().to_string()),
            ("TMPDIR", self.root.join("tmp").display().to_string()),
            ("SDK_FIXTURE_OWNER", env!("CARGO_BIN_EXE_xtask").to_owned()),
            ("SDK_FIXTURE_OS", os.to_owned()),
            (
                "SDK_FIXTURE_REFUSE",
                if refuse { "1" } else { "0" }.to_owned(),
            ),
            ("LLAMA_STAGE_BACKEND", backend.to_owned()),
            ("LLAMA_STAGE_LINK_MODE", "static".to_owned()),
            ("SKIPPY_LLAMA_AUTO_BUILD", "1".to_owned()),
            ("MESH_LLM_AUTO_BUILD_LLAMA", "1".to_owned()),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value.into())))
        .collect();
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                arguments: arguments
                    .iter()
                    .map(|word| Value::Public(word.clone().into()))
                    .collect(),
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
        assert_eq!(report.process.outcome, Outcome::Exited);
        assert!(report.process.failure.is_none());
        assert!(report.process.cleanup.complete);
        assert!(!self.root.join("native-tool-called").exists());
        report
    }

    fn package(&self, backend: &str, prebuilt: bool, refuse: bool) -> process::RawProcessReport {
        let mut args = vec![
            self.root
                .join("scripts/package-native-sdk.sh")
                .display()
                .to_string(),
            "--build".into(),
            "--backend".into(),
            backend.into(),
        ];
        if prebuilt {
            args.push("--require-prebuilt-llama".into());
        }
        self.invoke(&args, backend, refuse)
    }

    fn expected_parent(&self) -> PathBuf {
        self.root.join(".deps/llama-build")
    }

    fn assert_directory(&self, path: &Path, backend: &str) {
        assert!(path.is_absolute());
        assert_eq!(path.parent().unwrap(), self.expected_parent());
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(
            name.starts_with(&format!("build-stage-abi-static-{backend}")),
            "{name}"
        );
        assert!(
            name.ends_with("-aaaaaaaaaaaa"),
            "pin identity absent: {name}"
        );
    }
}

#[test]
fn static_abi_sdk_prebuilt_and_ordinary_branches_preserve_child_intent() {
    for backend in ["cpu", "metal", "cuda", "rocm"] {
        for prebuilt in [true, false] {
            let fixture = Fixture::new();
            let report = fixture.package(backend, prebuilt, false);
            assert!(
                report.process.status.unwrap().success(),
                "{}",
                String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
            );
            let events = fs::read_to_string(fixture.root.join("events")).unwrap();
            let rows: Vec<_> = events.lines().collect();
            assert_eq!(rows.len(), 4, "{events}");
            assert!(rows[0].starts_with("prepare|"));
            let query: Vec<_> = rows[1].split('|').collect();
            assert_eq!(&query[..2], &["query", backend]);
            fixture.assert_directory(Path::new(query[2]), backend);
            let native: Vec<_> = rows[2].split('|').collect();
            assert_eq!(
                &native[..3],
                &[
                    "native",
                    backend,
                    if prebuilt { "--require-existing" } else { "" }
                ]
            );
            assert_eq!(native[3], query[2]);
            let cargo: Vec<_> = rows[3].split('|').collect();
            let auto = if prebuilt { "0" } else { "1" };
            assert_eq!(&cargo[..4], &["cargo", backend, auto, auto]);
            assert_eq!(cargo[4], query[2]);
            let platform = if backend == "metal" {
                "darwin"
            } else {
                "linux"
            };
            let id = format!("meshllm-native-{platform}-x86_64-{backend}");
            let artifact = fixture.root.join("dist/native-sdk").join(&id);
            let manifest: Json =
                serde_json::from_slice(&fs::read(artifact.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(manifest["artifact_id"], id);
            assert_eq!(manifest["backend"], backend);
            assert_eq!(manifest["cargo_profile"], "release");
            assert_eq!(
                fs::read(artifact.join(manifest["library"].as_str().unwrap())).unwrap(),
                b"inert SDK bytes\n"
            );
            let archive = artifact.with_extension("tar.gz");
            assert!(archive.is_file());
            assert!(PathBuf::from(format!("{}.sha256", archive.display())).is_file());
        }
    }
}

#[test]
fn static_abi_sdk_prebuilt_refusal_stops_before_cargo_and_packaging() {
    let fixture = Fixture::new();
    let report = fixture.package("cpu", true, true);
    assert_eq!(report.process.status.unwrap().code(), Some(42));
    assert!(
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
            .contains("inert existing ABI rejected")
    );
    let events = fs::read_to_string(fixture.root.join("events")).unwrap();
    assert!(!events.lines().any(|line| line.starts_with("cargo|")));
    assert!(!fixture.root.join("target").exists());
    assert!(!fixture.root.join("dist").exists());
}

#[test]
fn static_abi_canonical_backend_directory_query_does_not_build() {
    for backend in ["cpu", "metal", "cuda", "rocm"] {
        let fixture = Fixture::new();
        fs::write(
            fixture.root.join(".deps/llama.cpp/.mesh-llm-patched-sha"),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        )
        .unwrap();
        let report = fixture.invoke(
            &[
                fixture
                    .root
                    .join("scripts/query-build-llama.sh")
                    .display()
                    .to_string(),
                "--print-build-dir".into(),
            ],
            backend,
            false,
        );
        assert!(report.process.status.unwrap().success());
        let output = String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes());
        fixture.assert_directory(Path::new(output.trim()), backend);
        assert!(!fixture.root.join("events").exists());
        assert!(!fixture.root.join("target").exists());
        assert!(!fixture.root.join("dist").exists());
    }
}
