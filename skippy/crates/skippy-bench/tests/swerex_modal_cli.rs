//! Native CLI/template boundaries plus explicitly selected prepared SDK qualification.
//! The SDK qualification uses no deployment constructors or services.
#![cfg(unix)]
use std::{
    fs,
    os::unix::process::CommandExt,
    path::PathBuf,
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};
const BINARY: &str = env!("CARGO_BIN_EXE_skippy-bench");
const TEMPLATE: &str = include_str!("../src/evals/adapters/templates/swe_bench_pro_run.sh");
#[path = "swerex_modal_cli/actual_sdk.rs"]
mod actual_sdk;
struct Fixture {
    root: PathBuf,
    _directory: tempfile::TempDir,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::Builder::new()
            .prefix("swerex-modal-cli-")
            .tempdir()
            .unwrap();
        Self {
            root: directory.path().canonicalize().unwrap(),
            _directory: directory,
        }
    }
    fn file(&self, relative: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
        let path = self.root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }
}
fn run(command: &mut Command) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            let pid = i32::try_from(child.id()).unwrap();
            // SAFETY: fixture child owns this process group.
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
            child.wait().unwrap();
            panic!("Modal fixture timed out");
        }
        thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}

const OFFICIAL: [(&str, &[u8]); 3] = [
    (
        "deployment/modal.py",
        include_bytes!("fixtures/swerex_modal/official/deployment/modal.py.fixture"),
    ),
    (
        "deployment/config.py",
        include_bytes!("fixtures/swerex_modal/official/deployment/config.py.fixture"),
    ),
    (
        "runtime/remote.py",
        include_bytes!("fixtures/swerex_modal/official/runtime/remote.py.fixture"),
    ),
];
const LEGACY: [(&str, &[u8]); 3] = [
    (
        "swerex/deployment/modal.py",
        include_bytes!("fixtures/swerex_modal/legacy/swerex/deployment/modal.py.fixture"),
    ),
    (
        "swerex/deployment/config.py",
        include_bytes!("fixtures/swerex_modal/legacy/swerex/deployment/config.py.fixture"),
    ),
    (
        "swerex/deployment/runtime/remote.py",
        include_bytes!("fixtures/swerex_modal/legacy/swerex/deployment/runtime/remote.py.fixture"),
    ),
];
fn populate(fixture: &Fixture) -> PathBuf {
    for (path, bytes) in OFFICIAL {
        fixture.file(&format!(".venv/lib/site-packages/swerex/{path}"), bytes);
    }
    for (path, bytes) in LEGACY {
        fixture.file(&format!("swerex_patches/{path}"), bytes);
    }
    fixture.file(
        ".venv/lib/site-packages/swe_rex-1.4.0.dist-info/METADATA",
        include_bytes!("fixtures/swerex_modal/METADATA"),
    );
    fixture.file(
        ".venv/lib/site-packages/swerex/__init__.py",
        b"SDK locator fixture",
    )
}
fn patch_command(fixture: &Fixture, locator: &std::path::Path) -> Command {
    let mut command = Command::new(BINARY);
    command
        .args(["eval", "patch-swerex-modal", "--module-source"])
        .arg(locator)
        .arg("--environment-root")
        .arg(fixture.root.join(".venv"))
        .arg("--patch-root")
        .arg(fixture.root.join("swerex_patches"))
        .env(
            "MESH_LLM_DATA_DIR",
            fixture.root.join("must-not-create-model-cache"),
        );
    command
}
#[test]
fn actual_cli_preserves_official_interfaces_and_backups_and_is_idempotent_without_model_preparation()
 {
    let fixture = Fixture::new();
    let locator = populate(&fixture);
    for _ in 0..2 {
        let output = run(&mut patch_command(&fixture, &locator));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        assert!(output.stderr.is_empty());
        assert!(!fixture.root.join("must-not-create-model-cache").exists());
    }
    for (path, original) in OFFICIAL {
        let destination = fixture
            .root
            .join(format!(".venv/lib/site-packages/swerex/{path}"));
        if path == "deployment/config.py" {
            assert_eq!(fs::read(&destination).unwrap(), original);
            assert!(!destination.with_extension("py.bak").exists());
        } else {
            assert_eq!(
                fs::read(destination.with_extension("py.bak")).unwrap(),
                original
            );
            assert_ne!(fs::read(&destination).unwrap(), original);
        }
    }
    let modal = fs::read_to_string(
        fixture
            .root
            .join(".venv/lib/site-packages/swerex/deployment/modal.py"),
    )
    .unwrap();
    assert!(modal.contains("await modal.Sandbox.create.aio("));
    let remote = fs::read_to_string(
        fixture
            .root
            .join(".venv/lib/site-packages/swerex/runtime/remote.py"),
    )
    .unwrap();
    assert!(remote.contains("headers[\"X-Request-ID\"] = request_id"));
    assert!(
        !fixture
            .root
            .join(".venv/lib/site-packages/swerex/deployment/runtime/remote.py")
            .exists()
    );
}
#[test]
fn actual_cli_refuses_final_source_corruption_before_any_destination_changes() {
    let fixture = Fixture::new();
    let locator = populate(&fixture);
    fixture.file(
        "swerex_patches/swerex/deployment/runtime/remote.py",
        b"corrupt pinned source",
    );
    let output = run(&mut patch_command(&fixture, &locator));
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    for (path, original) in OFFICIAL {
        let destination = fixture
            .root
            .join(format!(".venv/lib/site-packages/swerex/{path}"));
        assert_eq!(fs::read(&destination).unwrap(), original);
        assert!(!destination.with_extension("py.bak").exists());
    }
}
#[test]
fn actual_modal_rendered_run_uses_prepared_sdk_without_runtime_mutation() {
    let fixture = Fixture::new();
    let cache = fixture.root.join("cache");
    let output_dir = fixture.root.join("output");
    let result = run(Command::new(BINARY)
        .args([
            "eval",
            "run",
            "swe-bench-pro",
            "--dry-run",
            "--model",
            "fixture",
        ])
        .arg("--cache-root")
        .arg(&cache)
        .arg("--output-dir")
        .arg(&output_dir)
        .env("SWE_BENCH_PRO_DEPLOYMENT_TYPE", "modal"));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let rendered = fs::read_to_string(output_dir.join("raw/swe-bench-pro-run.sh")).unwrap();
    assert!(rendered.contains("-I -B -m sweagent.run.run run-batch"));
    assert!(rendered.contains("swe-sdk-v1/environment/bin/python"));
    assert!(rendered.contains("--instances.deployment.startup_timeout 1800"));
    assert!(rendered.contains("--instances.deployment.runtime_timeout 3600"));
    for removed in [
        "uv venv",
        "uv pip",
        "uv run",
        "eval patch-swerex-modal",
        "swerex_patches/patch.py",
    ] {
        assert!(!rendered.contains(removed), "{removed}");
    }
    assert!(!cache.join("swe-sdk-v1").exists());
    assert!(!TEMPLATE.contains("eval patch-swerex-modal"));
}
