//! Maintained CUDA adapters with private tools and a real Rust path controller.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap, fs, num::NonZeroUsize, os::unix::fs::PermissionsExt, path::PathBuf,
    time::Duration,
};

const LIBRARY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/lib/cuda-toolkit.sh"
));
const DETECTOR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/detect-cuda-toolkit-version.sh"
));

fn tool(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|p| p.join(name))
        .find(|p| p.is_file() && fs::metadata(p).unwrap().permissions().mode() & 0o111 != 0)
        .unwrap_or_else(|| panic!("CUDA fixture requires {name}"))
        .canonicalize()
        .unwrap()
}

struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("CUDA source with spaces");
        fs::create_dir_all(root.join("bin")).unwrap();
        for name in [
            "bash", "dirname", "readlink", "realpath", "sed", "head", "uname",
        ] {
            std::os::unix::fs::symlink(tool(name), root.join("bin").join(name)).unwrap();
        }
        let fixture = Self {
            _temporary: temporary,
            root,
        };
        fixture.write("scripts/lib/cuda-toolkit.sh", LIBRARY);
        fixture.write("scripts/detect-cuda-toolkit-version.sh", DETECTOR);
        fixture
    }

    fn write(&self, relative: &str, contents: &str) {
        let path = self.root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    fn compiler(&self, relative: &str, version: &str) -> PathBuf {
        self.write(relative, &format!("#!/bin/sh\n[ \"${{1:-}}\" = --version ] || exit 91\nprintf 'Cuda compilation tools, release {version}, V{version}.0\\n'\n"));
        let path = self.root.join(relative);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    fn run(&self, body: &str, values: &[(&str, &str)]) -> (bool, String, String) {
        let mut environment = BTreeMap::from([
            ("PATH".into(), Value::Public(self.root.join("bin").into())),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
        ]);
        for (key, value) in values {
            environment.insert((*key).into(), Value::Public((*value).into()));
        }
        let script = format!("source scripts/lib/cuda-toolkit.sh\n{body}\n");
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: tool("bash"),
                cwd: self.root.clone(),
                environment,
                arguments: vec![
                    Value::Public("-euo".into()),
                    Value::Public("pipefail".into()),
                    Value::Public("-c".into()),
                    Value::Public(script.into()),
                ],
            },
            &Limits {
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 16384,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(16384),
                stderr: NonZeroUsize::new(16384),
            },
        )
        .unwrap();
        assert_eq!(report.process.outcome, Outcome::Exited);
        assert!(
            report.process.failure.is_none()
                && report.process.cleanup.complete
                && !report.process.cleanup.forced
                && !report.process.cleanup.graceful_signal_failed
                && report.process.cleanup.failure.is_none(),
            "{report:?}"
        );
        assert_eq!(
            report.stdout.as_ref().unwrap().as_bytes().len() as u64,
            report.process.stdout.bytes_seen
        );
        assert_eq!(
            report.stderr.as_ref().unwrap().as_bytes().len() as u64,
            report.process.stderr.bytes_seen
        );
        (
            report.process.status.unwrap().success(),
            String::from_utf8(report.stdout.unwrap().as_bytes().to_vec()).unwrap(),
            String::from_utf8(report.stderr.unwrap().as_bytes().to_vec()).unwrap(),
        )
    }

    fn detect(&self, values: &[(&str, &str)]) -> (bool, String, String) {
        self.run("bash scripts/detect-cuda-toolkit-version.sh", values)
    }
}

#[test]
fn symlinked_compiler_resolves_toolkit_headers_and_all_library_variables() {
    let fixture = Fixture::new();
    let compiler = fixture.compiler("cuda-13.2/bin/nvcc", "13.2");
    fixture.write("cuda-13.2/targets/sbsa-linux/include/cuda_runtime.h", "");
    for library in ["libcudart.so", "libcublas.so", "libcublasLt.so"] {
        fixture.write(&format!("cuda-13.2/targets/sbsa-linux/lib/{library}"), "");
    }
    std::os::unix::fs::symlink(&compiler, fixture.root.join("bin/nvcc")).unwrap();
    let (ok, output, error) = fixture.run("configure_cuda_toolkit_env\nprintf '%s\\n' \"$CUDACXX\" \"$CUDAToolkit_ROOT\" \"$NVCC\" \"$CUDA_LIBRARY_PATH\" \"$LIBRARY_PATH\" \"$LD_LIBRARY_PATH\"", &[]);
    assert!(ok, "{error}");
    let toolkit = fixture.root.join("cuda-13.2");
    let library = toolkit.join("targets/sbsa-linux/lib");
    let expected =
        [&compiler, &toolkit, &compiler, &library, &library, &library].map(|p| p.to_str().unwrap());
    assert_eq!(output.lines().collect::<Vec<_>>(), expected);
}

#[test]
fn compiler_selection_obeys_cudacxx_cmake_nvcc_and_path_priority() {
    let fixture = Fixture::new();
    let cudacxx = fixture.compiler("bin/cudacxx-nvcc", "13.0");
    let cmake = fixture.compiler("bin/cmake-nvcc", "12.8");
    let nvcc = fixture.compiler("bin/nvcc-12", "12.9");
    fixture.compiler("bin/nvcc", "11.8");
    let selectors = [
        ("CUDACXX", cudacxx.to_str().unwrap()),
        ("CMAKE_CUDA_COMPILER", cmake.to_str().unwrap()),
        ("NVCC", nvcc.to_str().unwrap()),
    ];
    for (offset, expected) in [(0, "13.0"), (1, "12.8"), (2, "12.9"), (3, "11.8")] {
        let (ok, output, error) = fixture.detect(&selectors[offset..]);
        assert!(ok, "{error}");
        assert_eq!(output.trim(), expected);
    }
    let (ok, _, error) = fixture.detect(&[("CUDACXX", "missing-compiler")]);
    assert!(!ok && error.contains("could not be detected"), "{error}");
}

#[test]
fn major_only_declaration_is_accepted_and_minor_mismatch_rejected() {
    let fixture = Fixture::new();
    fixture.compiler("bin/nvcc", "13.0");
    let (ok, output, error) = fixture.detect(&[("MESH_CUDA_VERSION", "13")]);
    assert!(ok, "{error}");
    assert_eq!(output.trim(), "13");
    let (ok, _, error) = fixture.detect(&[("MESH_CUDA_VERSION", "13.1.2")]);
    assert!(
        !ok && error.contains("selected CUDA compiler/toolkit version 13.0"),
        "{error}"
    );
}

#[test]
fn toolkit_owned_metadata_is_accepted_without_a_compiler() {
    let fixture = Fixture::new();
    fixture.write("cuda-12.9/version.json", "{\"version\":\"12.9.2\"}\n");
    let root = fixture.root.join("cuda-12.9");
    let (ok, output, error) = fixture.detect(&[("CUDAToolkit_ROOT", root.to_str().unwrap())]);
    assert!(ok, "{error}");
    assert_eq!(output.trim(), "12.9");
}

#[test]
fn missing_compiler_and_unavailable_metadata_fail_without_driver_evidence() {
    let fixture = Fixture::new();
    // The production selector can inspect fixed host roots. Supply an unavailable
    // root-reader boundary so this negative fixture never depends on host CUDA.
    let (ok, _, error) = fixture.run(
        "cuda_toolkit_version_from_root() { return 1; }\ncuda_toolkit_manifest_version",
        &[],
    );
    assert!(
        !ok && error.contains("CUDA toolkit version could not be detected"),
        "{error}"
    );
    assert!(!error.contains("nvidia-smi"));
}

#[test]
fn canonical_path_fallback_uses_the_real_rust_controller_without_python_or_cargo() {
    let fixture = Fixture::new();
    fixture.write("target", "unchanged\n");
    std::os::unix::fs::symlink(fixture.root.join("target"), fixture.root.join("link")).unwrap();
    let (ok, output, error) = fixture.run(
        "readlink() { return 1; }\nrealpath() { return 1; }\ncuda_canonical_path link",
        &[],
    );
    assert!(ok, "{error}");
    assert_eq!(output.trim(), fixture.root.join("target").to_str().unwrap());
    assert_eq!(
        fs::read_to_string(fixture.root.join("target")).unwrap(),
        "unchanged\n"
    );
    for prohibited in ["python", "python3", "cargo"] {
        assert!(!fixture.root.join("bin").join(prohibited).exists());
    }
}

#[test]
fn windows_compiler_path_restores_exe_suffix_and_explicit_environment_remains_authoritative() {
    let fixture = Fixture::new();
    let compiler = fixture.compiler("nvcc.exe", "13.0");
    let selected = fixture.root.join("nvcc");
    let (ok, output, error) = fixture.run("uname() { printf 'MINGW64_NT-10.0\\n'; }\nconfigure_cuda_toolkit_env\nprintf '%s\\n' \"$CUDACXX\"", &[("CUDACXX", selected.to_str().unwrap())]);
    assert!(ok, "{error}");
    assert_eq!(output.trim(), compiler.to_str().unwrap());
    let explicit = fixture.root.join("custom-toolkit");
    let (ok, output, error) = fixture.run(
        "configure_cuda_toolkit_env\nprintf '%s\\n' \"$CUDACXX\" \"$CUDAToolkit_ROOT\"",
        &[
            ("CUDACXX", compiler.to_str().unwrap()),
            ("CUDAToolkit_ROOT", explicit.to_str().unwrap()),
        ],
    );
    assert!(ok, "{error}");
    assert_eq!(
        output.lines().collect::<Vec<_>>(),
        [compiler.to_str().unwrap(), explicit.to_str().unwrap()]
    );
}

#[test]
fn configuration_propagates_missing_compiler_failure() {
    let fixture = Fixture::new();
    assert!(!fixture.run("configure_cuda_toolkit_env", &[("PATH", "")]).0);
}

#[path = "cuda_toolkit/package_intent.rs"]
mod package_intent;
