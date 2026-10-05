//! Actual native build-script dynamic directives, with no compiler or runtime build.
use super::*;

struct DynamicFixture {
    temporary: tempfile::TempDir,
    build: PathBuf,
    stubs: PathBuf,
}
impl DynamicFixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let build = temporary.path().join("native build");
        let stubs = temporary.path().join("selected toolkit/stubs");
        fs::create_dir_all(&build).unwrap();
        fs::create_dir_all(&stubs).unwrap();
        let driver = stubs.join("libcuda.so");
        fs::write(&driver, b"inert driver fixture").unwrap();
        fs::write(
            build.join("CMakeCache.txt"),
            format!("CUDA_cuda_driver_LIBRARY:FILEPATH={}\n", driver.display()),
        )
        .unwrap();
        Self {
            temporary,
            build,
            stubs,
        }
    }
    fn run(&self, target: &str, backend: &str, legacy: bool, loader: bool) -> String {
        let root = self.temporary.path().canonicalize().unwrap();
        let prefix = if legacy {
            "SKIPPY_LLAMA"
        } else {
            "LLAMA_STAGE"
        };
        let mut environment: BTreeMap<_, _> = [
            ("XTASK_STATIC_ABI_FFI_CHILD".to_owned(), "yes".into()),
            (
                "CARGO_MANIFEST_DIR".to_owned(),
                root.join("skippy/crates/skippy-ffi").into_os_string(),
            ),
            ("TARGET".to_owned(), target.into()),
            (format!("{prefix}_LINK_MODE"), "dynamic".into()),
            (
                format!("{prefix}_BUILD_DIR"),
                self.build.clone().into_os_string(),
            ),
            (
                format!("{prefix}_LIB_DIR"),
                root.join("staged libraries").into_os_string(),
            ),
            (format!("{prefix}_BACKEND"), backend.into()),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value)))
        .collect();
        if loader {
            environment.insert(
                "CARGO_FEATURE_DYNAMIC_RUNTIME".into(),
                Value::Public("1".into()),
            );
        }
        let cache = fs::read(self.build.join("CMakeCache.txt")).unwrap();
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: std::env::current_exe().unwrap(),
                cwd: root,
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
        assert!(report.process.failure.is_none() && report.process.cleanup.complete);
        let stderr = String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes());
        assert!(report.process.status.unwrap().success(), "{stderr}");
        assert_eq!(fs::read(self.build.join("CMakeCache.txt")).unwrap(), cache);
        assert_eq!(
            fs::read(self.stubs.join("libcuda.so")).unwrap(),
            b"inert driver fixture"
        );
        String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes()).into_owned()
    }
}

#[test]
fn linux_cuda_dynamic_link_selects_cached_driver_without_runtime_search_paths() {
    let fixture = DynamicFixture::new();
    for target in ["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"] {
        let stdout = fixture.run(target, "cuda", false, false);
        for line in [
            format!("cargo:rustc-link-search=native={}", fixture.stubs.display()),
            "cargo:rustc-link-lib=dylib=cuda".to_owned(),
            "cargo:rustc-link-lib=dylib=llama".to_owned(),
            format!(
                "cargo:rerun-if-changed={}",
                fixture.build.join("CMakeCache.txt").display()
            ),
        ] {
            assert!(stdout.lines().any(|actual| actual == line), "{stdout}");
        }
        assert!(
            !stdout
                .lines()
                .any(|line| line.contains("rpath") || line.contains("rustc-link-arg"))
        );
    }
}

#[test]
fn legacy_dynamic_selectors_preserve_selected_cuda_driver() {
    let fixture = DynamicFixture::new();
    let stdout = fixture.run("aarch64-unknown-linux-gnu", "cuda", true, false);
    assert!(
        stdout
            .lines()
            .any(|line| line == "cargo:rustc-link-lib=dylib=cuda")
    );
    assert!(stdout.contains(&format!(
        "cargo:rustc-link-search=native={}",
        fixture.stubs.display()
    )));
}

#[test]
fn other_dynamic_targets_do_not_acquire_cuda_driver_dependency() {
    let fixture = DynamicFixture::new();
    for (target, backend) in [
        ("aarch64-unknown-linux-gnu", "cpu"),
        ("x86_64-unknown-linux-gnu", "rocm"),
        ("x86_64-unknown-linux-gnu", "vulkan"),
        ("aarch64-apple-darwin", "metal"),
        ("x86_64-pc-windows-msvc", "cuda"),
    ] {
        let stdout = fixture.run(target, backend, false, false);
        assert!(!stdout.contains("cargo:rustc-link-lib=dylib=cuda"));
        assert!(!stdout.contains(&format!(
            "cargo:rustc-link-search=native={}",
            fixture.stubs.display()
        )));
        assert!(
            stdout
                .lines()
                .any(|line| line == "cargo:rustc-link-lib=dylib=llama")
        );
    }
}

#[test]
fn dynamic_runtime_loader_remains_free_of_native_link_directives() {
    let fixture = DynamicFixture::new();
    let stdout = fixture.run("aarch64-unknown-linux-gnu", "cuda", false, true);
    assert!(
        !stdout
            .lines()
            .any(|line| line.starts_with("cargo:rustc-link-"))
    );
}

#[cfg(target_os = "linux")]
#[path = "dynamic_link/linux_elf.rs"]
mod linux_elf;
