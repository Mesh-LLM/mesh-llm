//! Actual FFI build-script cache admission and stale accelerator selection.
use super::*;

const ACCELERATORS: [(&str, &str); 5] = [
    ("GGML_BLAS", "blas"),
    ("GGML_CUDA", "cuda"),
    ("GGML_HIP", "hip"),
    ("GGML_VULKAN", "vulkan"),
    ("GGML_METAL", "metal"),
];

struct SelectionFixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    build: PathBuf,
    archives: Vec<PathBuf>,
}
impl SelectionFixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let build = root.join("selected ABI with spaces");
        let mut archives: Vec<_> = BASE
            .into_iter()
            .map(|(unix, _)| archive(&build, unix))
            .collect();
        for relative in ["tools/mtmd/libmtmd.a", "vendor/hash/libvendor-hash.a"] {
            archives.push(archive(&build, relative));
        }
        for (_, backend) in ACCELERATORS {
            archives.push(archive(
                &build,
                &format!("ggml/src/ggml-{backend}/libggml-{backend}.a"),
            ));
        }
        fs::write(root.join("outside-sentinel"), b"preserve").unwrap();
        Self {
            _temporary: temporary,
            root,
            build,
            archives,
        }
    }
    fn cache(&self, flags: &[(&str, &str)], newline: &str) {
        let text = flags
            .iter()
            .map(|(key, value)| format!("{key}:BOOL={value}{newline}"))
            .collect::<String>();
        fs::write(self.build.join("CMakeCache.txt"), text).unwrap();
    }
    fn run(&self, backend: &str) -> (bool, String, String) {
        let cache = self.build.join("CMakeCache.txt");
        let before = fs::read(&cache).ok();
        let was_directory = cache.is_dir();
        let environment: BTreeMap<_, _> = [
            ("XTASK_STATIC_ABI_FFI_CHILD", "yes".into()),
            (
                "CARGO_MANIFEST_DIR",
                self.root.join("skippy/crates/skippy-ffi").into_os_string(),
            ),
            ("TARGET", "aarch64-apple-darwin".into()),
            ("LLAMA_STAGE_BACKEND", backend.into()),
            ("LLAMA_STAGE_BUILD_DIR", self.build.clone().into_os_string()),
            ("LLAMA_STAGE_LINK_MODE", "static".into()),
            ("SKIPPY_LLAMA_AUTO_BUILD", "0".into()),
            ("MESH_LLM_AUTO_BUILD_LLAMA", "0".into()),
        ]
        .into_iter()
        .map(|(name, value)| (name.into(), Value::Public(value)))
        .collect();
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
                retained_bytes_per_stream: 32768,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(32768),
                stderr: NonZeroUsize::new(32768),
            },
        )
        .unwrap();
        assert_eq!(report.process.outcome, Outcome::Exited);
        assert!(report.process.failure.is_none() && report.process.cleanup.complete);
        assert_eq!(fs::read(&cache).ok(), before);
        assert_eq!(cache.is_dir(), was_directory);
        assert_eq!(
            fs::read(self.root.join("outside-sentinel")).unwrap(),
            b"preserve"
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

#[test]
fn cpu_never_links_stale_disabled_accelerator_archives_or_metal_frameworks() {
    let fixture = SelectionFixture::new();
    let flags: Vec<_> = ACCELERATORS.iter().map(|(key, _)| (*key, "OFF")).collect();
    fixture.cache(&flags, "\n");
    let (success, stdout, stderr) = fixture.run("cpu");
    assert!(success, "{stderr}");
    for (_, backend) in ACCELERATORS {
        assert!(
            !stdout
                .lines()
                .any(|line| line == format!("cargo:rustc-link-lib=static=ggml-{backend}")),
            "{stdout}"
        );
    }
    for framework in ["Foundation", "Metal", "MetalKit"] {
        assert!(
            !stdout
                .lines()
                .any(|line| line == format!("cargo:rustc-link-lib=framework={framework}")),
            "{stdout}"
        );
    }
}

#[test]
fn metal_and_blas_only_link_when_enabled_with_portable_cache_booleans() {
    let fixture = SelectionFixture::new();
    for (value, newline) in [
        ("ON", "\n"),
        ("ON", "\r\n"),
        ("ON\r", "\n"),
        (" TRUE \r", "\n"),
        (" 1 ", "\n"),
    ] {
        fixture.cache(&[("GGML_METAL", value), ("GGML_BLAS", "ON")], newline);
        let (success, stdout, stderr) = fixture.run("metal");
        assert!(success, "{stderr}");
        for library in ["ggml-metal", "ggml-blas"] {
            assert!(
                stdout
                    .lines()
                    .any(|line| line == format!("cargo:rustc-link-lib=static={library}")),
                "{stdout}"
            );
        }
        assert!(
            stdout
                .lines()
                .any(|line| line == "cargo:rustc-link-lib=framework=Metal")
        );
        assert!(
            !stdout
                .lines()
                .any(|line| line == "cargo:rustc-link-lib=static=ggml-cuda")
        );
    }
}

#[test]
fn disabled_selected_backend_is_rejected_despite_present_archive() {
    let fixture = SelectionFixture::new();
    fixture.cache(&[("GGML_METAL", "OFF")], "\n");
    let (success, _, stderr) = fixture.run("metal");
    assert!(!success);
    assert!(
        stderr.contains("selected backend requires GGML_METAL=ON"),
        "{stderr}"
    );
}

#[test]
fn every_unselected_enabled_accelerator_is_rejected_before_link_admission() {
    let fixture = SelectionFixture::new();
    for (key, _) in ACCELERATORS.iter().filter(|(key, _)| *key != "GGML_BLAS") {
        fixture.cache(&[(*key, "ON")], "\n");
        let (success, _, stderr) = fixture.run("cpu");
        assert!(!success);
        assert!(
            stderr.contains(&format!("staged backend mismatch: {key}=ON")),
            "{stderr}"
        );
    }
}

#[test]
fn missing_directory_and_invalid_utf8_cache_never_imply_disabled_backends() {
    for backend in ["cpu", "metal"] {
        for mode in ["missing", "directory", "invalid-utf8"] {
            let fixture = SelectionFixture::new();
            let cache = fixture.build.join("CMakeCache.txt");
            match mode {
                "directory" => fs::create_dir(&cache).unwrap(),
                "invalid-utf8" => fs::write(&cache, b"GGML_CUDA:BOOL=ON\n\xff").unwrap(),
                "missing" => {}
                _ => unreachable!(),
            }
            let (success, _, stderr) = fixture.run(backend);
            assert!(!success, "{backend}/{mode}");
            assert!(
                stderr.contains("cannot verify native backend configuration")
                    && stderr.contains("CMakeCache.txt"),
                "{stderr}"
            );
            assert!(!stderr.contains("selected backend requires"), "{stderr}");
        }
    }
}
