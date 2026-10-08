//! Actual optional CLI startup refusals; no upstream/model/task execution.
#![cfg(unix)]
use std::{
    fs,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "mcp-prep-cli-{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        Self(root.canonicalize().unwrap())
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn invoke(root: &Path, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_skippy-bench"));
    command
        .args(args)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", root.join("home"))
        .env("MESH_LLM_DATA_DIR", root.join("model-data"))
        .env("HF_HOME", root.join("hf"))
        .env("HF_HUB_CACHE", root.join("hub"))
        .env("HF_XET_CACHE", root.join("xet"));
    let mut child = command
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            // SAFETY: this child owns this fresh process group; no unrelated PID is targeted.
            unsafe {
                libc::kill(-i32::try_from(child.id()).unwrap(), libc::SIGKILL);
            }
            let _ = child.wait();
            panic!("bounded MCP startup fixture timed out");
        }
        thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}
fn unchanged(root: &Path) {
    for name in ["home", "model-data", "hf", "hub", "xet", "cache", "output"] {
        assert!(!root.join(name).exists(), "unexpected {name} side effect");
    }
}
#[test]
fn optional_mcp_prepare_dryrun_and_source_refusal_have_no_model_directory_side_effects() {
    let f = Fixture::new();
    let cache = f.0.join("cache");
    let cache = cache.to_str().unwrap();
    let dry = invoke(
        &f.0,
        &[
            "eval",
            "prepare-mcp",
            "--cache-root",
            cache,
            "--uv",
            "/not-executed/uv",
            "--python",
            "/not-executed/python",
            "--dry-run",
        ],
    );
    assert!(dry.status.success(), "{dry:?}");
    unchanged(&f.0);
    let refused = invoke(
        &f.0,
        &[
            "eval",
            "prepare-mcp",
            "--cache-root",
            cache,
            "--uv",
            "/not-executed/uv",
            "--python",
            "/not-executed/python",
        ],
    );
    assert!(!refused.status.success(), "{refused:?}");
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("Git"),
        "{refused:?}"
    );
    unchanged(&f.0);
}
#[test]
fn unavailable_mcp_run_refuses_before_global_model_cache_and_run_output() {
    let f = Fixture::new();
    let cache = f.0.join("cache");
    let output = f.0.join("output");
    let refused = invoke(
        &f.0,
        &[
            "eval",
            "run",
            "mcp-atlas",
            "--cache-root",
            cache.to_str().unwrap(),
            "--output-dir",
            output.to_str().unwrap(),
            "--model",
            "fixture",
            "--api-key",
            "fixture-key",
        ],
    );
    assert!(!refused.status.success(), "{refused:?}");
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("not installed"),
        "{refused:?}"
    );
    unchanged(&f.0);
}
