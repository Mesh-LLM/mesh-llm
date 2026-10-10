#![cfg(unix)]
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

#[allow(dead_code, unused_imports)]
#[path = "../src/process/mod.rs"]
mod process;

const PATCH_DIGEST: &str = "3800406abecfae8bd783a773345793e24506e649d42731e3c1da6c6d2d113d31";
const UPSTREAM: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn git(checkout: &Path, args: &[&str]) -> String {
    let output = Command::new("/usr/bin/git")
        .env_clear()
        .env("GIT_MASTER", "1")
        .env("PATH", "/usr/bin:/bin")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ALLOW_PROTOCOL", "file")
        .args([
            "--no-optional-locks",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Prepared Source Fixture",
            "-c",
            "user.email=fixture@example.invalid",
        ])
        .args(args)
        .current_dir(checkout)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().into()
}

struct Fixture {
    state: tempfile::TempDir,
    head: String,
}
impl Fixture {
    fn new() -> Self {
        let state = tempfile::tempdir().unwrap();
        let root = state.path();
        let checkout = root.join(".deps/llama.cpp");
        fs::create_dir_all(&checkout).unwrap();
        git(&checkout, &["init", "-q"]);
        fs::write(checkout.join("tracked.txt"), "selected prepared source\n").unwrap();
        git(&checkout, &["add", "tracked.txt"]);
        git(&checkout, &["commit", "-qm", "fixture"]);
        let head = git(&checkout, &["rev-parse", "HEAD"]);
        for (name, bytes) in [
            ("0001-base.patch", "base\n"),
            ("model_support/0001-test-support.patch", "support\n"),
            ("model_support/series", "0001-test-support.patch\r\n"),
            ("generated/0001-family-test.patch", "generated\n"),
            ("generated/series", "0001-family-test.patch\n"),
        ] {
            let path = root.join("third_party/llama.cpp/patches").join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        fs::write(
            root.join("third_party/llama.cpp/upstream.txt"),
            format!("{UPSTREAM}\n"),
        )
        .unwrap();
        for (name, value) in [
            ("upstream-sha", UPSTREAM),
            ("patch-digest", PATCH_DIGEST),
            ("patched-sha", &head),
            ("prepare-schema", "5"),
        ] {
            fs::write(
                checkout.join(format!(".mesh-llm-{name}")),
                format!("{value}\n"),
            )
            .unwrap();
        }
        Self { state, head }
    }
    fn run(&self) -> Output {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "canary-receipts", "prepared-source", "--root"])
            .arg(self.state.path())
            .output()
            .unwrap()
    }
    fn write(&self, path: &str, value: &str) {
        fs::write(self.state.path().join(path), value).unwrap();
    }
    fn reject(&self) {
        let output = self.run();
        assert!(!output.status.success());
        assert!(
            output.stdout.is_empty(),
            "failed admission must not emit a SHA"
        );
        assert!(!output.stderr.is_empty());
    }
}

#[test]
fn prepared_source_cli_reuses_three_lane_recipe_and_prints_only_clean_head() {
    let fixture = Fixture::new();
    let output = fixture.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, format!("{}\n", fixture.head).as_bytes());
    assert!(output.stderr.is_empty());
    // The fixed independent digest above includes core/support/generated paths
    // and bytes in preparation order; CRLF series must still admit.
    for lane in [
        "0001-base.patch",
        "model_support/0001-test-support.patch",
        "generated/0001-family-test.patch",
    ] {
        let fixture = Fixture::new();
        fixture.write(
            &format!("third_party/llama.cpp/patches/{lane}"),
            "changed\n",
        );
        fixture.reject();
    }
}

#[test]
fn prepared_source_cli_rejects_markers_upstream_head_and_dirty_checkout() {
    for (path, value) in [
        (".deps/llama.cpp/.mesh-llm-prepare-schema", "4\n"),
        (
            ".deps/llama.cpp/.mesh-llm-upstream-sha",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n",
        ),
        (
            ".deps/llama.cpp/.mesh-llm-patched-sha",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n",
        ),
        (".deps/llama.cpp/.mesh-llm-patch-digest", "invalid\n"),
        (
            "third_party/llama.cpp/upstream.txt",
            "not-a-source-revision\n",
        ),
        (".deps/llama.cpp/tracked.txt", "dirty selected source\n"),
    ] {
        let fixture = Fixture::new();
        fixture.write(path, value);
        fixture.reject();
    }
    let fixture = Fixture::new();
    fs::remove_file(
        fixture
            .state
            .path()
            .join(".deps/llama.cpp/.mesh-llm-prepare-schema"),
    )
    .unwrap();
    fixture.reject();
}

#[test]
fn prepared_source_cli_rejects_unsafe_or_incomplete_lane_and_symlinked_patch() {
    for series in [
        "",
        "../escape.patch\n",
        "0002-test-support.patch\n",
        "0001-test-support.patch\n0001-test-support.patch\n",
    ] {
        let fixture = Fixture::new();
        fixture.write("third_party/llama.cpp/patches/model_support/series", series);
        fixture.reject();
    }
    let fixture = Fixture::new();
    let patch = fixture
        .state
        .path()
        .join("third_party/llama.cpp/patches/0001-base.patch");
    fs::remove_file(&patch).unwrap();
    std::os::unix::fs::symlink("/dev/null", patch).unwrap();
    fixture.reject();
}

#[test]
fn prepared_source_cli_uses_explicit_root_and_owner_selected_git_not_path_override() {
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new();
    let decoy = tempfile::tempdir().unwrap();
    let marker = decoy.path().join("must-not-run");
    let fake_git = decoy.path().join("git");
    fs::write(
        &fake_git,
        format!(
            "#!/bin/sh\nprintf unexpected > '{}'\nexit 99\n",
            marker.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake_git, fs::Permissions::from_mode(0o700)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "canary-receipts", "prepared-source", "--root"])
        .arg(fixture.state.path())
        .env("PATH", decoy.path())
        .current_dir(decoy.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, format!("{}\n", fixture.head).as_bytes());
    assert!(!marker.exists());
    for args in [
        vec![],
        vec!["--root", "."],
        vec!["--root", "/unused", "--executable", "git"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "canary-receipts", "prepared-source"])
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}

// Finite outer bound protects this fixture if metadata admission regresses.
fn bounded_metadata_cli(fixture: &Fixture) -> Output {
    use std::{
        io::Read as _,
        process::Stdio,
        time::{Duration, Instant},
    };
    let stdout = tempfile::NamedTempFile::new().unwrap();
    let stderr = tempfile::NamedTempFile::new().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "canary-receipts", "prepared-source", "--root"])
        .arg(fixture.state.path())
        .stdin(Stdio::null())
        .stdout(stdout.reopen().unwrap())
        .stderr(stderr.reopen().unwrap())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("prepared-source metadata admission blocked past outer deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let read = |file: &tempfile::NamedTempFile| {
        let mut bytes = Vec::new();
        file.reopen()
            .unwrap()
            .take(65537)
            .read_to_end(&mut bytes)
            .unwrap();
        assert!(bytes.len() <= 65536);
        bytes
    };
    Output {
        status,
        stdout: read(&stdout),
        stderr: read(&stderr),
    }
}
#[test]
fn prepared_source_cli_rejects_fifo_directory_and_symlink_revision_metadata_without_hanging() {
    use std::os::unix::ffi::OsStrExt as _;
    for relative in [
        ".deps/llama.cpp/.mesh-llm-upstream-sha",
        ".deps/llama.cpp/.mesh-llm-patch-digest",
        ".deps/llama.cpp/.mesh-llm-patched-sha",
        ".deps/llama.cpp/.mesh-llm-prepare-schema",
        "third_party/llama.cpp/upstream.txt",
    ] {
        for kind in ["fifo", "directory", "symlink"] {
            let fixture = Fixture::new();
            let path = fixture.state.path().join(relative);
            fs::remove_file(&path).unwrap();
            match kind {
                "fifo" => {
                    let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
                    // SAFETY: valid fixture-owned NUL-terminated path.
                    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
                }
                "directory" => fs::create_dir(&path).unwrap(),
                _ => std::os::unix::fs::symlink("/dev/null", &path).unwrap(),
            }
            let before = std::time::Instant::now();
            let output = bounded_metadata_cli(&fixture);
            assert!(!output.status.success(), "{relative} {kind}");
            assert!(
                output.stdout.is_empty(),
                "failed admission emitted a digest"
            );
            assert!(
                String::from_utf8_lossy(&output.stderr).contains("regular file"),
                "{output:?}"
            );
            assert!(before.elapsed() < std::time::Duration::from_secs(4));
        }
    }
}
#[test]
fn prepared_source_cli_bounds_revision_metadata_and_rejects_invalid_utf8() {
    for bytes in [vec![b'5'; 4097], vec![0xff]] {
        let fixture = Fixture::new();
        fs::write(
            fixture
                .state
                .path()
                .join(".deps/llama.cpp/.mesh-llm-prepare-schema"),
            bytes,
        )
        .unwrap();
        let output = bounded_metadata_cli(&fixture);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
}

fn preparation_schema_operations() -> (u64, String) {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../skippy/scripts/prepare-llama.sh"),
    )
    .unwrap();
    let declarations: Vec<_> = source
        .lines()
        .filter(|line| line.starts_with("PREPARE_SCHEMA="))
        .collect();
    assert_eq!(
        declarations.len(),
        1,
        "one preparation schema owner required"
    );
    let declared_schema = declarations[0]
        .strip_prefix("PREPARE_SCHEMA=")
        .unwrap()
        .parse::<u64>()
        .expect("producer declares a numeric preparation schema");
    let writers: Vec<_> = source
        .lines()
        .filter(|line| line.contains(".mesh-llm-prepare-schema") && line.contains('>'))
        .collect();
    assert_eq!(
        writers.len(),
        1,
        "one final schema marker producer required"
    );
    (
        declared_schema,
        format!("set -euo pipefail\n{}\n{}\n", declarations[0], writers[0]),
    )
}

fn execute_preparation_schema_marker(fixture: &Fixture, body: String) {
    let spec = process::ProcessSpec {
        executable: "/bin/bash".into(),
        arguments: vec![
            process::Value::Public("-c".into()),
            process::Value::Public(body.into()),
        ],
        cwd: fixture.state.path().into(),
        environment: std::collections::BTreeMap::from([(
            "LLAMA_WORKDIR".into(),
            process::Value::Public(fixture.state.path().join(".deps/llama.cpp").into()),
        )]),
    };
    let limits = process::Limits {
        execution: std::time::Duration::from_secs(3),
        graceful_shutdown: std::time::Duration::from_secs(1),
        forced_shutdown: std::time::Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: process::Readiness::None,
        completion: process::Completion::Exit,
    };
    let report = process::supervise(
        &spec,
        &limits,
        &process::Cancellation::default(),
        process::OutputFiles::default(),
    )
    .unwrap();
    assert!(report.success(), "{report:?}");
    assert!(
        report.cleanup.complete && !report.stdout.truncated && !report.stderr.truncated,
        "{report:?}"
    );
}

#[test]
fn prepared_source_verifier_accepts_the_actual_producer_schema_marker_and_rejects_other_versions() {
    let (declared_schema, body) = preparation_schema_operations();
    let fixture = Fixture::new();
    let marker = fixture
        .state
        .path()
        .join(".deps/llama.cpp/.mesh-llm-prepare-schema");
    fs::remove_file(&marker).unwrap();
    execute_preparation_schema_marker(&fixture, body);
    let produced = fs::read(&marker).unwrap();
    let schema = std::str::from_utf8(&produced)
        .unwrap()
        .trim()
        .parse::<u64>()
        .expect("numeric preparation schema");
    assert_eq!(
        schema, declared_schema,
        "producer marker must represent its declared format"
    );
    let output = bounded_metadata_cli(&fixture);
    assert!(
        output.status.success(),
        "producer schema {schema} rejected: {output:?}"
    );
    assert_eq!(output.stdout, format!("{}\n", fixture.head).as_bytes());
    assert!(output.stderr.is_empty());
    assert_eq!(
        fs::read(&marker).unwrap(),
        produced,
        "verifier must not repair producer metadata"
    );
    let mut incompatible = vec![schema.checked_add(1).expect("schema has a successor")];
    if let Some(previous) = schema.checked_sub(1) {
        incompatible.push(previous);
    }
    for value in incompatible {
        let bytes = format!("{value}\n");
        fs::write(&marker, &bytes).unwrap();
        let output = bounded_metadata_cli(&fixture);
        assert!(
            !output.status.success(),
            "unproduced schema {value} admitted"
        );
        assert!(
            output.stdout.is_empty(),
            "incompatible schema emitted a prepared SHA"
        );
        assert_eq!(fs::read(&marker).unwrap(), bytes.as_bytes());
    }
}

#[path = "prepared_source/prepare_producer.rs"]
mod prepare_producer;
