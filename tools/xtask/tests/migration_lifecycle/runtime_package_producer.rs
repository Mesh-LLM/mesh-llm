//! Actual package-owned shell functions with inert compiler/Cargo boundaries.
use crate::process;
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt as _,
    path::{Path, PathBuf},
    time::Duration,
};

fn owner(first: &str, next: &str) -> String {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../skippy/scripts/package-native-runtime.sh"),
    )
    .unwrap();
    assert_eq!(source.matches(first).count(), 1);
    source[source.find(first).unwrap()..source.find(next).unwrap()].into()
}
struct Fixture {
    directory: tempfile::TempDir,
}
impl Fixture {
    fn new() -> Self {
        let fixture = Self {
            directory: tempfile::tempdir().unwrap(),
        };
        for dir in ["bin", "stage/lib", "stage/tools", "build", "target"] {
            fs::create_dir_all(fixture.path().join(dir)).unwrap();
        }
        fixture.tool("patchelf", "exit 0");
        fixture
    }
    fn path(&self) -> &Path {
        self.directory.path()
    }
    fn tool(&self, name: &str, body: &str) -> PathBuf {
        let p = self.path().join("bin").join(name);
        fs::write(&p, format!("#!/bin/bash\nset -euo pipefail\n{body}\n")).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o700)).unwrap();
        p
    }
    fn run(&self, body: &str, extra: &[(&str, String)]) -> (bool, String, String) {
        let mut environment = BTreeMap::from([
            (
                "PATH".into(),
                process::Value::Public(
                    format!("{}:/usr/bin:/bin", self.path().join("bin").display()).into(),
                ),
            ),
            (
                "TEST_ROOT".into(),
                process::Value::Public(self.path().into()),
            ),
            (
                "AUTOMATION".into(),
                process::Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
        ]);
        for (k, v) in extra {
            environment.insert((*k).into(), process::Value::Public(v.as_str().into()));
        }
        let raw = process::supervise_raw(
            &process::ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.path().into(),
                environment,
                arguments: ["-euo", "pipefail", "-c", body]
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
        let report = raw.process;
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert!(
            report.failure.is_none()
                && report.cleanup.complete
                && !report.cleanup.forced
                && !report.cleanup.graceful_signal_failed
                && report.cleanup.failure.is_none(),
            "{report:?}"
        );
        let out = raw.stdout.unwrap();
        let err = raw.stderr.unwrap();
        assert_eq!(out.as_bytes().len() as u64, report.stdout.bytes_seen);
        assert_eq!(err.as_bytes().len() as u64, report.stderr.bytes_seen);
        (
            report.status.unwrap().success(),
            String::from_utf8(out.as_bytes().to_vec()).unwrap(),
            String::from_utf8(err.as_bytes().to_vec()).unwrap(),
        )
    }
}
#[test]
fn package_gpu_tools_use_shared_cuda_and_each_configured_rocm_arch_without_compiling() {
    let functions = owner(
        "gpu_benchmark_tool_path() {",
        "build_model_package_tool() {",
    );
    for backend in ["cuda", "rocm"] {
        let fixture = Fixture::new();
        fixture.tool("compiler", "printf '%s\\n' \"$@\" > \"$TEST_ROOT/arguments\"\nprevious=''\nfor argument in \"$@\"; do\n if [[ \"$previous\" == -o ]]; then output=\"$argument\"; fi\n previous=\"$argument\"\ndone\n: > \"$output\"");
        let body = format!(
            "{functions}\nBACKEND={backend}\nruntime_os=linux\nstage_dir=\"$TEST_ROOT/stage\"\nREPO_ROOT=\"$TEST_ROOT/inert-source\"\ntool_paths=()\nHIPCC=\"$TEST_ROOT/bin/compiler\"\nLLAMA_STAGE_AMDGPU_TARGETS='gfx90a;gfx942, gfx1151'\ncuda_selected_compiler() {{ printf '%s\\n' \"$TEST_ROOT/bin/compiler\"; }}\nbuild_gpu_benchmark_tool\nprintf '%s\\n' \"${{tool_paths[@]}}\"\n"
        );
        let (ok, out, error) = fixture.run(&body, &[]);
        assert!(ok, "{error}");
        assert_eq!(out, "tools/mesh-llm-gpu-benchmark\n");
        let arguments = fs::read_to_string(fixture.path().join("arguments")).unwrap();
        let args = arguments.lines().collect::<Vec<_>>();
        if backend == "cuda" {
            let index = args.iter().position(|v| *v == "-cudart").unwrap();
            assert_eq!(args[index + 1], "shared");
            // Without configured architectures, nvcc keeps its own default.
            assert!(
                !args
                    .iter()
                    .any(|v| v.starts_with("--generate-code") || v.starts_with("-arch"))
            );
        } else {
            assert_eq!(
                args.iter()
                    .copied()
                    .filter(|v| v.starts_with("--offload-arch="))
                    .collect::<Vec<_>>(),
                [
                    "--offload-arch=gfx90a",
                    "--offload-arch=gfx942",
                    "--offload-arch=gfx1151"
                ]
            );
        }
        assert!(
            fixture
                .path()
                .join("stage/tools/mesh-llm-gpu-benchmark")
                .is_file()
        );
        fixture.directory.close().unwrap();
    }
}
#[test]
fn package_cuda_benchmark_tool_uses_configured_or_blackwell_architectures() {
    let functions = owner(
        "gpu_benchmark_tool_path() {",
        "build_model_package_tool() {",
    );
    let cases = [
        (
            "cuda",
            "LLAMA_STAGE_CUDA_ARCHITECTURES='87;120a-real, 75-virtual'\n",
            vec![
                "--generate-code=arch=compute_87,code=[compute_87,sm_87]",
                "--generate-code=arch=compute_120a,code=sm_120a",
                "--generate-code=arch=compute_75,code=compute_75",
            ],
        ),
        (
            "cuda-blackwell",
            "",
            vec!["--generate-code=arch=compute_120,code=[compute_120,sm_120]"],
        ),
    ];
    for (backend, architectures, expected) in cases {
        let fixture = Fixture::new();
        fixture.tool("compiler", "printf '%s\\n' \"$@\" > \"$TEST_ROOT/arguments\"\nprevious=''\nfor argument in \"$@\"; do\n if [[ \"$previous\" == -o ]]; then output=\"$argument\"; fi\n previous=\"$argument\"\ndone\n: > \"$output\"");
        let body = format!(
            "{functions}\nBACKEND={backend}\nruntime_os=linux\nstage_dir=\"$TEST_ROOT/stage\"\nREPO_ROOT=\"$TEST_ROOT/inert-source\"\ntool_paths=()\n{architectures}cuda_selected_compiler() {{ printf '%s\\n' \"$TEST_ROOT/bin/compiler\"; }}\nbuild_gpu_benchmark_tool\n"
        );
        let (ok, _, error) = fixture.run(&body, &[]);
        assert!(ok, "{error}");
        let arguments = fs::read_to_string(fixture.path().join("arguments")).unwrap();
        assert_eq!(
            arguments
                .lines()
                .filter(|v| v.starts_with("--generate-code"))
                .collect::<Vec<_>>(),
            expected
        );
        fixture.directory.close().unwrap();
    }
}
#[test]
fn package_cuda_closure_invokes_native_collect_order_and_bundles_only_redistributed_license() {
    let functions = owner(
        "linux_cuda_redistributable_present() {",
        "rewrite_macos_runtime_paths() {",
    );
    for redistributable in [false, true] {
        let fixture = Fixture::new();
        let ordered = if redistributable {
            "printf '%s\\n' libcudart.so.12 libllama.so"
        } else {
            "printf '%s\\n' libllama.so"
        };
        fixture.tool("cargo", &format!("printf '%s\\n' \"$*\" >> \"$TEST_ROOT/calls\"\ncase \"$*\" in\n 'xtool native linux-runtime-deps collect '*) ;;\n 'xtool native linux-runtime-deps order '*) {ordered} ;;\n *) exit 97 ;;\nesac"));
        let body = format!(
            "{functions}\nTARGET_TRIPLE=x86_64-unknown-linux-gnu\nBACKEND=cuda\nruntime_arch=x86_64\nstage_dir=\"$TEST_ROOT/stage\"\nprimary_name=libllama.so\nlibrary_paths=(lib/libllama.so)\nlinux_cuda_dependency_search_dirs() {{ printf '%s\\n' /inert/cuda /inert/cuda; }}\ncuda_toolkit_major() {{ printf '12\\n'; }}\nbundle_cuda_distribution_license() {{ printf 'license\\n' > \"$TEST_ROOT/license-call\"; }}\ncollect_linux_cuda_dependencies\nprintf '%s\\n' \"${{library_paths[@]}}\"\n"
        );
        let (ok, out, error) = fixture.run(&body, &[]);
        assert!(ok, "{error}");
        let calls = fs::read_to_string(fixture.path().join("calls")).unwrap();
        let calls = calls.lines().collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert!(
            calls[0].contains("linux-runtime-deps collect")
                && calls[0].contains("--arch x86_64 --cuda-major 12")
        );
        assert_eq!(calls[0].matches("--search-dir /inert/cuda").count(), 1);
        assert!(
            calls[1].contains("linux-runtime-deps order")
                && calls[1].contains("--primary libllama.so")
        );
        assert_eq!(
            fixture.path().join("license-call").exists(),
            redistributable
        );
        assert_eq!(
            out,
            if redistributable {
                "lib/libcudart.so.12\nlib/libllama.so\n"
            } else {
                "lib/libllama.so\n"
            }
        );
        fixture.directory.close().unwrap();
    }
}
#[test]
fn package_model_tool_windows_omission_and_cargo_defaults_preserve_inherited_flags() {
    let functions = owner(
        "build_model_package_tool() {",
        "collect_runtime_libraries() {",
    );
    for runtime in ["windows", "linux"] {
        let fixture = Fixture::new();
        fixture.tool("cargo", "case \"${1:-}\" in\n build) printf '%s\\n' \"$*\" \"RUSTFLAGS=$RUSTFLAGS\" \"CARGO_ENCODED_RUSTFLAGS=$CARGO_ENCODED_RUSTFLAGS\" > \"$TEST_ROOT/build-call\"; mkdir -p \"$TEST_ROOT/target/x86_64-unknown-linux-gnu/release\"; printf '#!/bin/sh\\nexit 0\\n' > \"$TEST_ROOT/target/x86_64-unknown-linux-gnu/release/skippy-package-builder\"; chmod +x \"$TEST_ROOT/target/x86_64-unknown-linux-gnu/release/skippy-package-builder\" ;;\n metadata) printf '{\"target_directory\":\"%s/target\"}\\n' \"$TEST_ROOT\" ;;\n xtool) shift; exec \"$AUTOMATION\" \"$@\" ;;\n *) exit 98 ;;\nesac");
        let body = format!(
            "{functions}\nruntime_os={runtime}\nstage_dir=\"$TEST_ROOT/stage\"\nLLAMA_STAGE_BUILD_DIR=\"$TEST_ROOT/build\"\nTARGET_TRIPLE=x86_64-unknown-linux-gnu\ntool_paths=()\nmodel_package_tool_path() {{ printf 'tools/skippy-package-builder\\n'; }}\nbuild_backend() {{ printf 'cpu\\n'; }}\nbuild_model_package_tool\nprintf '%s' \"${{tool_paths[*]-}}\"\n"
        );
        let (ok, out, error) = fixture.run(
            &body,
            &[
                ("RUSTFLAGS", "-C debuginfo=1".into()),
                ("CARGO_ENCODED_RUSTFLAGS", "-C\u{1f}debuginfo=1".into()),
            ],
        );
        assert!(ok, "{error}");
        if runtime == "windows" {
            assert!(out.is_empty());
            assert!(!fixture.path().join("build-call").exists());
            assert!(
                !fixture
                    .path()
                    .join("stage/tools/skippy-package-builder")
                    .exists()
            );
        } else {
            assert_eq!(out, "tools/skippy-package-builder");
            let log = fs::read_to_string(fixture.path().join("build-call")).unwrap();
            assert!(log.contains(
                "build --release --locked --target x86_64-unknown-linux-gnu -p skippy-package-builder"
            ));
            assert!(log.contains("RUSTFLAGS=-C debuginfo=1"));
            assert!(log.contains("CARGO_ENCODED_RUSTFLAGS=-C\u{1f}debuginfo=1"));
            assert!(
                fixture
                    .path()
                    .join("stage/tools/skippy-package-builder")
                    .is_file()
            );
        }
        fixture.directory.close().unwrap();
    }
}

#[test]
fn package_cpu_rocm_complete_script_emits_consumable_manifest_archive_and_locale_floor() {
    for backend in ["cpu", "rocm"] {
        let fixture = Fixture::new();
        fs::write(
            fixture.path().join("build/libllama.so"),
            b"\x7fELFinert-runtime",
        )
        .unwrap();
        fixture.tool(
            "cargo",
            "[[ ${1:-} == xtool ]] || exit 96\nshift\nexec \"$AUTOMATION\" \"$@\"",
        );
        fixture.tool("readelf", "[[ $LC_ALL == C && ${1:-} == -V ]] || exit 95\nprintf 'Version definition section GLIBC_9.99\\nVersion needs section GLIBC_2.17 GLIBC_ABI_DT_RELR\\n'\nprintf '%s\\n' \"$LC_ALL\" >> \"$TEST_ROOT/readelf-locale\"");
        let model_tool = fixture.tool("model-tool", "exit 0");
        let hipcc = fixture.tool("hipcc", "printf '%s\\n' \"$@\" > \"$TEST_ROOT/hipcc-arguments\"\nprevious=''\nfor argument in \"$@\"; do\n if [[ \"$previous\" == -o ]]; then output=\"$argument\"; fi\n previous=\"$argument\"\ndone\n: > \"$output\"");
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../skippy/scripts/package-native-runtime.sh");
        let body = "\"$PACKAGE_SCRIPT\" --backend \"$PACKAGE_BACKEND\" --target x86_64-unknown-linux-gnu --out \"$TEST_ROOT/output\"";
        let (ok, _, error) = fixture.run(
            body,
            &[
                ("PACKAGE_SCRIPT", script.to_string_lossy().into_owned()),
                ("PACKAGE_BACKEND", backend.into()),
                (
                    "LLAMA_STAGE_BUILD_DIR",
                    fixture.path().join("build").to_string_lossy().into_owned(),
                ),
                (
                    "LLAMA_WORKDIR",
                    fixture
                        .path()
                        .join("absent-owned-source")
                        .to_string_lossy()
                        .into_owned(),
                ),
                (
                    "MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL",
                    model_tool.to_string_lossy().into_owned(),
                ),
                ("HIPCC", hipcc.to_string_lossy().into_owned()),
                (
                    "LLAMA_STAGE_AMDGPU_TARGETS",
                    "gfx90a;gfx942, gfx1151".into(),
                ),
                ("LC_ALL", "POSIX".into()),
            ],
        );
        assert!(ok, "{error}");
        let artifact = format!("meshllm-native-runtime-linux-x86_64-{backend}");
        let output = fixture.path().join("output");
        let stage = output.join(&artifact);
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(stage.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["schema_version"], 2);
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let runtime_version =
            fs::read_to_string(root.join("skippy/crates/skippy-native-runtime/RUNTIME_VERSION"))
                .unwrap();
        assert_eq!(
            manifest["runtime"]["release_version"],
            runtime_version.trim()
        );
        assert!(manifest["runtime"].get("mesh_version").is_none());
        assert_eq!(manifest["runtime"]["platform"]["min_glibc"], "2.36");
        assert_eq!(
            fs::read_to_string(fixture.path().join("readelf-locale")).unwrap(),
            "C\n"
        );
        let tools = manifest["runtime"]["tools"].as_object().unwrap();
        assert!(tools.contains_key("tools/skippy-package-builder"));
        assert_eq!(tools.len(), if backend == "cpu" { 1 } else { 2 });
        assert_eq!(
            fs::read(stage.join("tools/skippy-package-builder")).unwrap(),
            fs::read(&model_tool).unwrap()
        );
        if backend == "rocm" {
            let arguments = fs::read_to_string(fixture.path().join("hipcc-arguments")).unwrap();
            assert_eq!(
                arguments
                    .lines()
                    .filter(|v| v.starts_with("--offload-arch="))
                    .collect::<Vec<_>>(),
                [
                    "--offload-arch=gfx90a",
                    "--offload-arch=gfx942",
                    "--offload-arch=gfx1151"
                ]
            );
            assert_eq!(
                manifest["runtime"]["backend"]["rocm"]["gpu_arches"],
                serde_json::json!(["gfx90a", "gfx942", "gfx1151"])
            );
        }
        let archive = output.join(format!("{artifact}.tar.gz"));
        assert!(archive.is_file());
        let archive_name = archive.to_string_lossy().into_owned();
        let (ok, listing, error) =
            fixture.run("tar -tzf \"$ARCHIVE\"", &[("ARCHIVE", archive_name)]);
        assert!(ok, "{error}");
        assert!(
            listing
                .lines()
                .any(|entry| entry == format!("{artifact}/manifest.json"))
        );
        assert!(listing.lines().all(|entry| {
            !Path::new(entry)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("._")
        }));
        fixture.directory.close().unwrap();
    }
}

#[test]
fn package_runtime_version_reads_product_file_instead_of_differing_workspace_version() {
    let fixture = Fixture::new();
    let product = fixture.path().join("skippy/crates/skippy-native-runtime");
    fs::create_dir_all(&product).unwrap();
    fs::write(product.join("RUNTIME_VERSION"), b"1.2.3\n").unwrap();
    fs::write(
        fixture.path().join("Cargo.toml"),
        b"[workspace.package]\nversion = \"99.88.77\"\n",
    )
    .unwrap();
    fixture.tool(
        "cargo",
        "[[ ${1:-} == xtool ]] || exit 96\nshift\nexec \"$AUTOMATION\" \"$@\"",
    );
    let functions = owner("skippy_runtime_version() {", "skippy_abi_version() {");
    let body = format!("{functions}\nREPO_ROOT=\"$TEST_ROOT\"\nskippy_runtime_version\n");
    let (ok, output, error) = fixture.run(&body, &[]);
    assert!(ok, "{error}");
    assert_eq!(output, "1.2.3\n");
    fs::remove_file(product.join("RUNTIME_VERSION")).unwrap();
    let (ok, output, error) = fixture.run(&body, &[]);
    assert!(!ok);
    assert!(output.is_empty());
    assert!(!error.is_empty());
    fixture.directory.close().unwrap();
}

#[test]
fn prepopulated_cuda_closure_with_no_search_paths_still_redistributes_license() {
    let fixture = Fixture::new();
    fs::write(
        fixture.path().join("stage/lib/libllama.so"),
        b"inert primary",
    )
    .unwrap();
    fs::write(
        fixture.path().join("stage/lib/libcudart.so.12"),
        b"prepopulated redistributable",
    )
    .unwrap();
    fs::write(
        fixture.path().join("cuda-license"),
        b"owned redistribution terms\n",
    )
    .unwrap();
    fixture.tool("cargo", "printf '%s\\n' \"$*\" >> \"$TEST_ROOT/calls\"\ncase \"$*\" in\n 'xtool native linux-runtime-deps collect '*) ;;\n 'xtool native linux-runtime-deps order '*) printf '%s\\n' libcudart.so.12 libllama.so ;;\n *) exit 97 ;;\nesac");
    let functions = owner(
        "cuda_distribution_license_file() {",
        "rewrite_macos_runtime_paths() {",
    );
    let body = format!(
        "{functions}\nTARGET_TRIPLE=x86_64-unknown-linux-gnu\nBACKEND=cuda\nruntime_arch=x86_64\nstage_dir=\"$TEST_ROOT/stage\"\nprimary_name=libllama.so\nlibrary_paths=(lib/libcudart.so.12 lib/libllama.so)\nlicense_paths=()\nMESH_LLM_CUDA_LICENSE_FILE=\"$TEST_ROOT/cuda-license\"\nlinux_cuda_dependency_search_dirs() {{ return 0; }}\ncuda_toolkit_major() {{ printf '12\\n'; }}\ncollect_linux_cuda_dependencies\nprintf '%s\\n' \"${{license_paths[@]}}\"\n"
    );
    let (ok, output, error) = fixture.run(&body, &[]);
    assert!(ok, "{error}");
    assert_eq!(output, "licenses/NVIDIA-CUDA-LICENSE.txt\n");
    assert_eq!(
        fs::read(
            fixture
                .path()
                .join("stage/licenses/NVIDIA-CUDA-LICENSE.txt")
        )
        .unwrap(),
        fs::read(fixture.path().join("cuda-license")).unwrap()
    );
    let calls = fs::read_to_string(fixture.path().join("calls")).unwrap();
    assert_eq!(calls.lines().count(), 2);
    assert!(
        calls
            .lines()
            .next()
            .unwrap()
            .contains("linux-runtime-deps collect")
    );
    assert!(!calls.contains("--search-dir"));
    assert_eq!(
        fs::read(fixture.path().join("stage/lib/libcudart.so.12")).unwrap(),
        b"prepopulated redistributable"
    );
    fixture.directory.close().unwrap();
}

#[path = "runtime_package_producer/cuda_package_producer.rs"]
mod cuda_package_producer;
