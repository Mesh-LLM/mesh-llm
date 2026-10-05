//! Entire source-bound install.ps1 entrypoint, actual CLI and loopback download.
use super::{
    consumer_http::{ASSET, Server},
    fixture,
    native::{Installed, environment, fixture_host, old_install, system_powershell},
};
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path, time::Duration};
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
    help: bool,
    invalid_value: bool,
}

fn execute(
    root: &Path,
    arguments: Vec<std::ffi::OsString>,
    environment: std::collections::BTreeMap<std::ffi::OsString, Value>,
    seconds: u64,
) -> process::ProcessReport {
    let report = process::supervise(
        &ProcessSpec {
            executable: system_powershell(),
            arguments: arguments.into_iter().map(Value::Public).collect(),
            cwd: root.into(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(seconds),
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
        report.cleanup.complete && !report.cleanup.forced && !report.cleanup.graceful_signal_failed,
        "{report:?}"
    );
    assert!(
        report.failure.is_none() && report.cleanup.failure.is_none(),
        "{report:?}"
    );
    assert!(
        !report.stdout.truncated
            && !report.stderr.truncated
            && report.stdout.suppressed_lines == 0
            && report.stderr.suppressed_lines == 0,
        "{report:?}"
    );
    report
}

fn arguments(script: &Path) -> Vec<std::ffi::OsString> {
    [
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
    ]
    .into_iter()
    .map(Into::into)
    .chain([script.as_os_str().to_owned()])
    .collect()
}

fn run(case: Case, prepare_existing: impl FnOnce(&Path)) -> Installed {
    let root = tempfile::Builder::new()
        .prefix("windows-consumer-native-")
        .tempdir()
        .unwrap();
    let prepared = root.path().join("prepared");
    fs::create_dir(&prepared).unwrap();
    fixture::write_bundle(
        &prepared,
        &fixture_host(),
        matches!(case.fault, Fault::TamperedHost),
    );
    let bin = root.path().join("bin");
    fs::create_dir(&bin).unwrap();
    prepare_existing(&bin);
    let source = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../install.ps1"));
    let maintained =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../install.ps1")).unwrap();
    assert_eq!(
        maintained, source,
        "installer differs from compiled consumer source"
    );
    let snapshot = root.path().join("install.ps1");
    fs::write(&snapshot, source).unwrap();
    let preparation = root.path().join("prepare_zip.ps1");
    fs::write(&preparation, include_str!("prepare_zip.ps1")).unwrap();
    let mut prep = arguments(&preparation);
    prep.extend([
        "-Root".into(),
        root.path().as_os_str().to_owned(),
        "-Source".into(),
        snapshot.as_os_str().to_owned(),
        "-SourceSha256".into(),
        hex::encode(Sha256::digest(source)).into(),
    ]);
    let report = execute(root.path(), prep, environment(root.path()), 20);
    assert!(report.success(), "{report:?}");
    let mut archive = Vec::new();
    fs::File::open(root.path().join("fixture.zip"))
        .unwrap()
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut archive)
        .unwrap();
    let server = Server::start(archive);
    let mut env = environment(root.path());
    env.insert(
        "MESH_LLM_INSTALL_URL_BASE".into(),
        Value::Public(server.base.clone().into()),
    );
    env.insert(
        "MESH_LLM_INSTALL_INTERACTIVE".into(),
        Value::Public(if case.interactive { "1" } else { "0" }.into()),
    );
    env.insert(
        "MESH_LLM_REQUIRE_CHECKSUM".into(),
        Value::Public("1".into()),
    );
    if matches!(case.fault, Fault::RuntimeReplacement) {
        // The preceding native preparation already executed actual x64 admission.
        // Only this failure case uses the two existing production fault-hook flags.
        env.insert(
            "MESH_LLM_INSTALL_TEST_ALLOW_NONWINDOWS".into(),
            Value::Public("1".into()),
        );
        env.insert(
            "MESH_LLM_INSTALL_TEST_FAIL_AFTER_RUNTIME_REPLACE".into(),
            Value::Public("1".into()),
        );
    }
    let mut argv = arguments(&snapshot);
    argv.extend([
        "-InstallDir".into(),
        bin.into_os_string(),
        "-NoPathUpdate".into(),
    ]);
    if case.no_setup {
        argv.push("-NoSetup".into());
    }
    if !case.flavor.is_empty() {
        argv.extend(["-Flavor".into(), case.flavor.into()]);
    }
    if case.help {
        argv.push("-Help".into());
    }
    if case.invalid_value {
        argv.push("-Flavor".into());
    }
    let report = execute(root.path(), argv, env, 60);
    let requests = server.finish();
    if case.help || case.invalid_value {
        assert!(requests.is_empty(), "{requests:?}");
    } else {
        assert_eq!(requests, [format!("/{ASSET}"), format!("/{ASSET}.sha256")]);
        let stdout = String::from_utf8_lossy(&report.stdout.bytes_retained);
        assert!(stdout.contains("Release channel: stable"), "{stdout}");
        assert!(stdout.contains("Verified checksum:"), "{stdout}");
    }
    // The actual top-level finally must remove its randomized installer root,
    // including after integrity, replacement and post-commit cleanup errors.
    for name in ["TEMP", "TMP"] {
        assert_eq!(
            fs::read_dir(root.path().join(name)).unwrap().count(),
            0,
            "installer leaked {name} contents"
        );
    }
    Installed { root, report }
}

#[test]
fn native_windows_consumer_interactive_runs_setup_and_warns_for_legacy_flavor() {
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
fn native_windows_consumer_noninteractive_prints_setup_and_installs_runtime() {
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
fn native_windows_consumer_no_setup_prints_command_without_executing_setup() {
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
fn native_windows_consumer_tampered_host_digest_preserves_existing_install() {
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
fn native_windows_consumer_removes_stale_host_imports_when_bundle_omits_it() {
    let result = run(Case::default(), |path| {
        fs::write(path.join("host-imports.json"), b"stale\n").unwrap();
    });
    result.accepted();
    assert!(!result.path("bin/host-imports.json").exists());
    result.finish();
}
#[test]
fn native_windows_consumer_failed_runtime_replacement_restores_previous_product() {
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
fn native_windows_consumer_stale_cleanup_failure_keeps_committed_new_product() {
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

#[test]
fn native_windows_consumer_help_and_missing_cli_value_avoid_download_and_mutation() {
    let help = run(
        Case {
            help: true,
            ..Case::default()
        },
        |_| {},
    );
    help.accepted();
    assert!(help.output().contains("Usage: install.ps1"));
    assert!(help.calls().is_empty());
    assert_eq!(fs::read_dir(help.path("bin")).unwrap().count(), 0);
    help.finish();
    let invalid = run(
        Case {
            invalid_value: true,
            ..Case::default()
        },
        |_| {},
    );
    invalid.rejected();
    assert!(invalid.output().contains("Flavor"));
    assert!(invalid.calls().is_empty());
    assert_eq!(fs::read_dir(invalid.path("bin")).unwrap().count(), 0);
    invalid.finish();
}
