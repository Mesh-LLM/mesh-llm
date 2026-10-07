use super::fixture;
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    TamperedHost,
    RuntimeReplacement,
}
#[derive(Default)]
struct Case {
    interactive: bool,
    no_setup: bool,
    flavor: &'static str,
    fault: Fault,
}
pub(super) struct Installed {
    pub(super) root: tempfile::TempDir,
    pub(super) report: process::ProcessReport,
}
impl Installed {
    pub(super) fn finish(self) {
        self.root
            .close()
            .expect("owned installer fixture directory deletion failed");
    }
    pub(super) fn path(&self, relative: &str) -> PathBuf {
        self.root.path().join(relative)
    }
    pub(super) fn calls(&self) -> Vec<String> {
        match fs::File::open(self.path("calls.txt")) {
            Ok(file) => {
                let mut bytes = Vec::new();
                file.take(4097).read_to_end(&mut bytes).unwrap();
                assert!(bytes.len() <= 4096, "fixture call trace exceeds bound");
                String::from_utf8(bytes)
                    .unwrap()
                    .lines()
                    .map(str::to_owned)
                    .collect()
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => panic!("fixture call trace unreadable: {error}"),
        }
    }
    pub(super) fn output(&self) -> String {
        format!(
            "{}\n{}",
            String::from_utf8_lossy(&self.report.stdout.bytes_retained),
            String::from_utf8_lossy(&self.report.stderr.bytes_retained)
        )
    }
    pub(super) fn accepted(&self) {
        assert!(
            self.report.success(),
            "{:?}\n{}",
            self.report,
            self.output()
        );
    }
    pub(super) fn rejected(&self) {
        assert!(!self.report.success(), "{:?}", self.report);
        assert!(
            self.report
                .status
                .is_some_and(|status| status.code().is_some_and(|code| code != 0)),
            "failure must be actual native process exit"
        );
    }
}
pub(super) fn system_powershell() -> PathBuf {
    let root = std::env::var_os("SystemRoot").expect("native Windows SystemRoot required");
    let path = PathBuf::from(root).join("System32/WindowsPowerShell/v1.0/powershell.exe");
    assert!(path.is_file(), "Windows PowerShell prerequisite missing");
    path
}
pub(super) fn fixture_host() -> Vec<u8> {
    let unit_binary = std::env::current_exe().unwrap();
    let target_debug = unit_binary.parent().unwrap().parent().unwrap();
    let path = target_debug.join("examples/migration_generator_fixture.exe");
    assert!(fs::symlink_metadata(&path).unwrap().file_type().is_file());
    let mut bytes = Vec::new();
    fs::File::open(path)
        .expect("build existing migration_generator_fixture example before native installer tests")
        .take(32 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(
        bytes.starts_with(b"MZ") && bytes.len() < 32 * 1024 * 1024,
        "need bounded actual native PE fixture executable"
    );
    bytes
}
pub(super) fn environment(root: &Path) -> BTreeMap<std::ffi::OsString, Value> {
    let mut values = [
        "SystemRoot",
        "WINDIR",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
    ]
    .into_iter()
    .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
    .collect::<BTreeMap<_, _>>();
    let executable_extensions =
        std::env::var("PATHEXT").expect("native Windows PowerShell fixture requires PATHEXT");
    assert!(
        executable_extensions
            .split(';')
            .any(|extension| extension.eq_ignore_ascii_case(".EXE")),
        "native Windows PowerShell fixture requires .EXE in PATHEXT"
    );
    for name in ["TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA", "APPDATA"] {
        let destination = root.join(name);
        fs::create_dir_all(&destination).unwrap();
        values.insert(name.into(), Value::Public(destination.into_os_string()));
    }
    values.insert(
        "MESH_WINDOWS_INSTALL_FIXTURE_CALLS".into(),
        Value::Public(root.join("calls.txt").into_os_string()),
    );
    let core_modules = system_powershell().parent().unwrap().join("Modules");
    assert!(
        core_modules.is_dir(),
        "Windows PowerShell core modules required"
    );
    values.insert(
        "PSModulePath".into(),
        Value::Public(core_modules.into_os_string()),
    );
    values
}
fn run(case: Case, prepare_existing: impl FnOnce(&Path)) -> Installed {
    let root = tempfile::Builder::new()
        .prefix("windows-install-native-")
        .tempdir()
        .unwrap();
    let prepared = root.path().join("prepared");
    fs::create_dir(&prepared).unwrap();
    fixture::write_bundle(
        &prepared,
        &fixture_host(),
        matches!(case.fault, Fault::TamperedHost),
    );
    fs::create_dir(root.path().join("bin")).unwrap();
    prepare_existing(&root.path().join("bin"));
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../install.ps1")
        .canonicalize()
        .unwrap();
    let source_digest = hex::encode(Sha256::digest(fs::read(&source).unwrap()));
    let compiled_digest = hex::encode(Sha256::digest(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../install.ps1"
    ))));
    assert_eq!(
        source_digest, compiled_digest,
        "installer differs from compiled fixture source"
    );
    let harness = root.path().join("harness.ps1");
    fs::write(&harness, include_str!("harness.ps1")).unwrap();
    let options = root.path().join("options.json");
    fs::write(&options, serde_json::to_vec(&json!({"interactive":case.interactive,"skip_setup":case.no_setup,"flavor":case.flavor,"fail_runtime":matches!(case.fault, Fault::RuntimeReplacement)})).unwrap()).unwrap();
    let arguments = vec![
        "-NoProfile".into(),
        "-NonInteractive".into(),
        "-ExecutionPolicy".into(),
        "Bypass".into(),
        "-File".into(),
        harness.into_os_string(),
        "-Root".into(),
        root.path().as_os_str().to_owned(),
        "-Source".into(),
        source.into_os_string(),
        "-SourceSha256".into(),
        source_digest.into(),
        "-Options".into(),
        options.into_os_string(),
    ];
    let spec = ProcessSpec {
        executable: system_powershell(),
        arguments: arguments.into_iter().map(Value::Public).collect(),
        cwd: root.path().to_owned(),
        environment: environment(root.path()),
    };
    let report = process::supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(45),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 128 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, process::Outcome::Exited, "{report:?}");
    assert!(
        report.failure.is_none() && report.cleanup.failure.is_none(),
        "{report:?}"
    );
    assert!(
        report.cleanup.complete && !report.cleanup.forced && !report.cleanup.graceful_signal_failed,
        "owned PowerShell/fixture tree must finish cleanly: {report:?}"
    );
    assert!(
        report.stdout.line_capture_complete
            && report.stderr.line_capture_complete
            && !report.stdout.truncated
            && !report.stderr.truncated
            && report.stdout.suppressed_lines == 0
            && report.stderr.suppressed_lines == 0,
        "fixture output must be complete"
    );
    Installed { root, report }
}
pub(super) fn old_install(path: &Path) {
    fs::write(path.join("mesh-llm.exe"), b"existing\n").unwrap();
    fs::create_dir_all(path.join("native-runtimes/old-runtime")).unwrap();
    fs::write(
        path.join("native-runtimes/old-runtime/manifest.json"),
        b"old-runtime\n",
    )
    .unwrap();
    fs::write(path.join("product-manifest.json"), b"old-manifest\n").unwrap();
}
#[test]
fn native_windows_installer_interactive_runs_setup_and_warns_for_legacy_flavor() {
    let result = run(
        Case {
            interactive: true,
            flavor: "cuda",
            ..Case::default()
        },
        |_| {},
    );
    result.accepted();
    assert_eq!(result.calls(), ["--version", "setup"]);
    assert!(
        result
            .output()
            .contains("Installing Windows x64 MeshLLM product bundle")
    );
    assert!(result.output().contains("Ignoring legacy -Flavor 'cuda'"));
    result.finish();
}
#[test]
fn native_windows_installer_noninteractive_prints_setup_and_installs_runtime() {
    let result = run(Case::default(), |_| {});
    result.accepted();
    assert_eq!(result.calls(), ["--version"]);
    assert!(result.output().contains("Run this next:"));
    assert!(result.output().contains("mesh-llm.exe\" setup"));
    assert!(
        result
            .path("bin/native-runtimes/test-runtime/manifest.json")
            .is_file()
    );
    result.finish();
}
#[test]
fn native_windows_installer_no_setup_prints_command_without_executing_setup() {
    let result = run(
        Case {
            interactive: true,
            no_setup: true,
            ..Case::default()
        },
        |_| {},
    );
    result.accepted();
    assert_eq!(result.calls(), ["--version"]);
    assert!(result.output().contains("Run this next:"));
    assert!(result.output().contains("mesh-llm.exe\" setup"));
    result.finish();
}
#[test]
fn native_windows_installer_tampered_host_digest_preserves_existing_install() {
    let result = run(
        Case {
            fault: Fault::TamperedHost,
            ..Case::default()
        },
        |path| {
            fs::write(path.join("mesh-llm.exe"), b"existing\n").unwrap();
            fs::write(path.join("product-manifest.json"), b"{\"existing\":true}\n").unwrap();
        },
    );
    result.rejected();
    assert!(result.output().contains("host.sha256 mismatch"));
    assert_eq!(
        fs::read(result.path("bin/mesh-llm.exe")).unwrap(),
        b"existing\n"
    );
    assert_eq!(
        fs::read(result.path("bin/product-manifest.json")).unwrap(),
        b"{\"existing\":true}\n"
    );
    assert!(result.calls().is_empty());
    result.finish();
}
#[test]
fn native_windows_installer_removes_stale_host_imports_when_bundle_omits_it() {
    let result = run(Case::default(), |path| {
        fs::write(path.join("host-imports.json"), b"stale\n").unwrap();
    });
    result.accepted();
    assert!(!result.path("bin/host-imports.json").exists());
    result.finish();
}
#[test]
fn native_windows_installer_failed_runtime_replacement_restores_previous_product() {
    let result = run(
        Case {
            fault: Fault::RuntimeReplacement,
            ..Case::default()
        },
        old_install,
    );
    result.rejected();
    assert!(
        result
            .output()
            .contains("test requested failure after runtime replacement")
    );
    assert_eq!(
        fs::read(result.path("bin/mesh-llm.exe")).unwrap(),
        b"existing\n"
    );
    assert_eq!(
        fs::read(result.path("bin/native-runtimes/old-runtime/manifest.json")).unwrap(),
        b"old-runtime\n"
    );
    assert_eq!(
        fs::read(result.path("bin/product-manifest.json")).unwrap(),
        b"old-manifest\n"
    );
    assert!(!result.path("bin/native-runtimes/test-runtime").exists());
    assert!(result.calls().is_empty());
    result.finish();
}
#[test]
fn native_windows_installer_stale_cleanup_failure_keeps_committed_new_product() {
    let result = run(Case::default(), |path| {
        fs::create_dir(path.join("rpc-server.exe")).unwrap();
        fs::write(
            path.join("rpc-server.exe/child"),
            b"prevent nonrecursive removal\n",
        )
        .unwrap();
    });
    result.rejected();
    assert!(result.path("bin/mesh-llm.exe").is_file());
    assert!(
        result
            .path("bin/native-runtimes/test-runtime/manifest.json")
            .is_file()
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(result.path("bin/product-manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["runtime"]["id"], "test-runtime");
    assert!(result.path("bin/rpc-server.exe/child").is_file());
    assert!(result.calls().is_empty());
    result.finish();
}
