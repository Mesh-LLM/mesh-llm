//! Real Linux GNU linker closure, with tiny local C libraries and no driver runtime.
use super::*;
use std::ffi::OsString;

fn installed(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").expect("Linux link test requires PATH"))
        .map(|root| root.join(name))
        .find(|path| path.is_file())
        .unwrap_or_else(|| panic!("Linux link test requires {name}"))
}

fn execute(root: &Path, name: &str, arguments: Vec<OsString>) -> (bool, String, String) {
    let environment = ["PATH", "RUSTUP_HOME", "CARGO_HOME", "RUSTUP_TOOLCHAIN"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (name.into(), Value::Public(value))))
        .collect();
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: installed(name),
            cwd: root.to_owned(),
            arguments: arguments.into_iter().map(Value::Public).collect(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(20),
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
    (
        report.process.status.unwrap().success(),
        String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes()).into_owned(),
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes()).into_owned(),
    )
}

fn shared_library(root: &Path, source: &str, code: &str, output: &Path, extra: &[OsString]) {
    let path = root.join(source);
    fs::write(&path, code).unwrap();
    let mut arguments = vec!["-shared".into(), "-fPIC".into(), path.into_os_string()];
    arguments.extend_from_slice(extra);
    arguments.extend(["-o".into(), output.as_os_str().to_owned()]);
    let (success, _, stderr) = execute(root, "cc", arguments);
    assert!(success, "{stderr}");
}

fn link_arguments(stdout: &str) -> Vec<OsString> {
    let mut result = Vec::new();
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix("cargo:rustc-link-search=") {
            result.extend(["-L".into(), value.into()]);
        } else if let Some(value) = line.strip_prefix("cargo:rustc-link-lib=") {
            result.extend(["-l".into(), value.into()]);
        }
    }
    result
}

fn prepare_libraries(root: &Path, fixture: &DynamicFixture, staged: &Path) {
    shared_library(
        root,
        "driver.c",
        "int cuGetErrorString(void) { return 0; }\n",
        &fixture.stubs.join("libcuda.so"),
        &["-Wl,-soname,libcuda.so.1".into()],
    );
    shared_library(
        root,
        "runtime.c",
        "extern int cuGetErrorString(void);\nint llama_entry(void) { return cuGetErrorString(); }\n",
        &staged.join("libllama.so"),
        &[
            "-L".into(),
            fixture.stubs.as_os_str().to_owned(),
            "-lcuda".into(),
            "-Wl,-soname,libllama.so".into(),
        ],
    );
    for name in ["mtmd", "llama-common"] {
        shared_library(
            root,
            "empty.c",
            "void unused(void) {}\n",
            &staged.join(format!("lib{name}.so")),
            &[],
        );
    }
}

fn compile_probe(root: &Path, arguments: &[OsString]) -> PathBuf {
    let source = root.join("probe.rs");
    fs::write(&source, "unsafe extern \"C\" { fn llama_entry() -> i32; }\nfn main() { std::process::exit(unsafe { llama_entry() }); }\n").unwrap();
    let binary = root.join("probe");
    let command = vec![
        "--edition=2024".into(),
        "-C".into(),
        "link-arg=-fuse-ld=bfd".into(),
        source.into_os_string(),
        "-o".into(),
        binary.clone().into_os_string(),
    ];
    let mut broken = command.clone();
    broken.extend_from_slice(&arguments[..arguments.len() - 2]);
    let (success, _, stderr) = execute(root, "rustc", broken);
    assert!(!success);
    assert!(stderr.contains("cuGetErrorString"), "{stderr}");
    let mut complete = command;
    complete.extend_from_slice(arguments);
    let (success, _, stderr) = execute(root, "rustc", complete);
    assert!(success, "{stderr}");
    binary
}

#[test]
fn driverless_gnu_elf_link_requires_explicit_driver_without_bundling_or_rpath() {
    installed("ld.bfd");
    let fixture = DynamicFixture::new();
    let root = fixture.temporary.path().canonicalize().unwrap();
    let target = format!("{}-unknown-linux-gnu", std::env::consts::ARCH);
    let stdout = fixture.run(&target, "cuda", false, false);
    let arguments = link_arguments(&stdout);
    assert_eq!(
        &arguments[arguments.len() - 2..],
        [OsString::from("-l"), OsString::from("dylib=cuda")]
    );
    let staged = root.join("staged libraries");
    fs::create_dir(&staged).unwrap();
    prepare_libraries(&root, &fixture, &staged);
    let binary = compile_probe(&root, &arguments);
    let (success, dynamic, stderr) =
        execute(&root, "readelf", vec!["-d".into(), binary.into_os_string()]);
    assert!(success, "{stderr}");
    let (success, runtime, stderr) = execute(
        &root,
        "readelf",
        vec!["-d".into(), staged.join("libllama.so").into_os_string()],
    );
    assert!(success, "{stderr}");
    assert!(runtime.contains("[libcuda.so.1]"), "{runtime}");
    for forbidden in [fixture.stubs.to_str().unwrap(), "RPATH", "RUNPATH"] {
        assert!(!dynamic.contains(forbidden), "{dynamic}");
    }
    assert!(!staged.join("libcuda.so").exists());
    assert!(!staged.join("libcuda.so.1").exists());
}
