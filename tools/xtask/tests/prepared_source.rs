#![cfg(unix)]
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

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
