//! Execute the unchanged build-script policy in a fresh test-binary child.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

// Only the included pre-existing build-script main exceeds xtask's line limit.
// New fixture code keeps the normal lint policy; production source is unchanged.
#[allow(clippy::too_many_lines)]
mod actual_build_script {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../skippy/crates/skippy-ffi/build.rs"
    ));
    pub(super) fn run() {
        main();
    }
}

const BASE: [(&str, &str); 6] = [
    ("src/libllama.a", "src/llama.lib"),
    ("common/libllama-common.a", "common/llama-common.lib"),
    (
        "common/libllama-common-base.a",
        "common/llama-common-base.lib",
    ),
    ("ggml/src/libggml.a", "ggml/src/ggml.lib"),
    ("ggml/src/libggml-base.a", "ggml/src/ggml-base.lib"),
    (
        "ggml/src/ggml-cpu/libggml-cpu.a",
        "ggml/src/ggml-cpu/ggml-cpu.lib",
    ),
];

struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    build: PathBuf,
    archives: Vec<PathBuf>,
    cache: Vec<u8>,
    msvc: bool,
}
fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn archive(build: &Path, relative: &str) -> PathBuf {
    let path = build.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, b"inert static archive input").unwrap();
    path
}

impl Fixture {
    fn new(backend: &str, msvc: bool, hash: bool) -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("FFI boundary with spaces");
        for name in ["skippy/crates/skippy-ffi", "scripts", "bin"] {
            fs::create_dir_all(root.join(name)).unwrap();
        }
        let build = root.join("prepared").join(format!("static-{backend}"));
        let mut archives: Vec<_> = BASE
            .into_iter()
            .map(|(unix, windows)| archive(&build, if msvc { windows } else { unix }))
            .collect();
        archives.push(archive(
            &build,
            if msvc {
                "tools/mtmd/mtmd.lib"
            } else {
                "tools/mtmd/libmtmd.a"
            },
        ));
        if hash {
            archives.push(archive(
                &build,
                if msvc {
                    "vendor/hash/vendor-hash.lib"
                } else {
                    "vendor/hash/libvendor-hash.a"
                },
            ));
        }
        let backend_archive = match backend {
            "cuda" => Some("ggml/src/ggml-cuda/libggml-cuda.a"),
            "rocm" => Some("ggml/src/ggml-hip/libggml-hip.a"),
            "metal" => Some("ggml/src/ggml-metal/libggml-metal.a"),
            _ => None,
        };
        if let Some(relative) = backend_archive {
            archives.push(archive(&build, relative));
        }
        let cache = format!("GGML_CUDA:BOOL={}\nGGML_HIP:BOOL={}\nGGML_METAL:BOOL={}\nGGML_VULKAN:BOOL=OFF\nGGML_OPENMP_ENABLED:BOOL=OFF\n",
            if backend == "cuda" { "ON" } else { "OFF" },
            if backend == "rocm" { "ON" } else { "OFF" },
            if backend == "metal" { "ON" } else { "OFF" }).into_bytes();
        fs::write(build.join("CMakeCache.txt"), &cache).unwrap();
        executable(
            &root.join("scripts/build-llama.sh"),
            r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 1 && "$1" == --print-build-dir && "$LLAMA_STAGE_LINK_MODE" == static ]] || { printf forbidden > "$FFI_FIXTURE_ROOT/native-called"; exit 93; }
printf '%s|%s|%s\n' "$LLAMA_STAGE_BACKEND" "$LLAMA_STAGE_LINK_MODE" "$*" >> "$FFI_FIXTURE_ROOT/query.calls"
if [[ "$FFI_QUERY_FAIL" == 1 ]]; then printf 'finite canonical query refusal\n' >&2; exit 46; fi
printf '%s\n' "$FFI_FIXTURE_ROOT/prepared/static-$LLAMA_STAGE_BACKEND"
"#,
        );
        let deny = "#!/bin/sh\nprintf forbidden > \"$FFI_FIXTURE_ROOT/native-called\"\nexit 94\n";
        executable(&root.join("scripts/prepare-llama.sh"), deny);
        for name in [
            "cargo", "rustc", "cmake", "make", "ninja", "nvcc", "sccache",
        ] {
            executable(&root.join("bin").join(name), deny);
        }
        fs::write(root.join("outside-sentinel"), b"preserve").unwrap();
        Self {
            _temporary: temporary,
            root,
            build,
            archives,
            cache,
            msvc,
        }
    }

    fn invoke(&self, backend: &str, explicit: bool, fail_query: bool) -> (bool, String, String) {
        let target = if self.msvc {
            "x86_64-pc-windows-msvc"
        } else if backend == "metal" {
            "x86_64-apple-darwin"
        } else {
            "x86_64-unknown-linux-gnu"
        };
        let mut environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            ("XTASK_STATIC_ABI_FFI_CHILD", "yes".to_owned()),
            ("FFI_FIXTURE_ROOT", self.root.display().to_string()),
            (
                "FFI_QUERY_FAIL",
                if fail_query { "1" } else { "0" }.to_owned(),
            ),
            (
                "CARGO_MANIFEST_DIR",
                self.root
                    .join("skippy/crates/skippy-ffi")
                    .display()
                    .to_string(),
            ),
            ("TARGET", target.to_owned()),
            ("LLAMA_STAGE_BACKEND", backend.to_owned()),
            ("SKIPPY_LLAMA_AUTO_BUILD", "0".to_owned()),
            ("MESH_LLM_AUTO_BUILD_LLAMA", "0".to_owned()),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        if explicit {
            environment.insert(
                "LLAMA_STAGE_BUILD_DIR".into(),
                Value::Public(self.build.clone().into_os_string()),
            );
        }
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: std::env::current_exe().unwrap(),
                cwd: self.root.clone(),
                arguments: [
                    "--exact",
                    "static_abi_ffi_boundary::static_abi_ffi_actual_boundary_contracts",
                    "--nocapture",
                    "--test-threads=1",
                ]
                .into_iter()
                .map(|word| Value::Public(word.into()))
                .collect(),
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
        assert!(
            !self.root.join("native-called").exists(),
            "unexpected native/build boundary called"
        );
        assert_eq!(
            fs::read(self.root.join("outside-sentinel")).unwrap(),
            b"preserve"
        );
        assert_eq!(
            fs::read(self.build.join("CMakeCache.txt")).unwrap(),
            self.cache
        );
        for path in &self.archives {
            assert_eq!(fs::read(path).unwrap(), b"inert static archive input");
        }
        (
            report.process.status.unwrap().success(),
            String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes()).into_owned(),
            String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes()).into_owned(),
        )
    }
}

fn assert_order(stdout: &str) {
    let emitted: Vec<_> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("cargo:rustc-link-lib=static="))
        .collect();
    let mtmd = emitted.iter().position(|line| *line == "mtmd").unwrap();
    let hash = emitted
        .iter()
        .position(|line| *line == "vendor-hash")
        .unwrap();
    assert!(mtmd < hash, "{stdout}");
}

#[test]
fn static_abi_ffi_actual_boundary_contracts() {
    if std::env::var_os("XTASK_STATIC_ABI_FFI_CHILD").is_some() {
        assert_eq!(std::env::var("XTASK_STATIC_ABI_FFI_CHILD").unwrap(), "yes");
        actual_build_script::run();
        return;
    }
    for msvc in [false, true] {
        let fixture = Fixture::new("cpu", msvc, true);
        let (success, stdout, stderr) = fixture.invoke("cpu", true, false);
        assert!(success, "{stderr}");
        assert_order(&stdout);
        assert!(stdout.contains(&format!(
            "cargo:rustc-link-search=native={}",
            fixture.build.join("vendor/hash").display()
        )));
        assert!(!fixture.root.join("query.calls").exists());
        let fixture = Fixture::new("cpu", msvc, false);
        let (success, stdout, stderr) = fixture.invoke("cpu", true, false);
        assert!(!success);
        assert!(stderr.contains("mtmd requires vendor::hash"), "{stderr}");
        assert!(
            !stdout
                .lines()
                .any(|line| line == "cargo:rustc-link-lib=static=vendor-hash")
        );
    }
    for backend in ["cpu", "metal", "cuda", "rocm"] {
        let fixture = Fixture::new(backend, false, true);
        let (success, stdout, stderr) = fixture.invoke(backend, false, false);
        assert!(success, "{stderr}");
        assert_order(&stdout);
        assert_eq!(
            fs::read_to_string(fixture.root.join("query.calls")).unwrap(),
            format!("{backend}|static|--print-build-dir\n")
        );
        assert!(stdout.contains(&format!(
            "cargo:rustc-link-search=native={}",
            fixture.build.join("src").display()
        )));
    }
    let fixture = Fixture::new("cpu", false, true);
    let (success, stdout, stderr) = fixture.invoke("cpu", false, true);
    assert!(!success);
    assert!(
        stderr.contains("finite canonical query refusal"),
        "{stderr}"
    );
    assert!(
        !stdout
            .lines()
            .any(|line| line.starts_with("cargo:rustc-link-lib="))
    );
}

#[path = "static_abi_ffi_boundary/dynamic_link.rs"]
mod dynamic_link;

#[path = "static_abi_ffi_boundary/static_backend_selection.rs"]
mod static_backend_selection;
