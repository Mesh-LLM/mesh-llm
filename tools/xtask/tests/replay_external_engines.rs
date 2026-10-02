#![cfg(unix)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Fixture {
    state: tempfile::TempDir,
    executable: PathBuf,
    config: PathBuf,
}
impl Fixture {
    fn new(engine: &str) -> Self {
        let state = tempfile::Builder::new()
            .prefix("external engine ")
            .tempdir()
            .unwrap();
        let executable = state.path().join("engine fixture");
        write_executable(
            &executable,
            &format!(
                "#!/bin/sh\ncase \"$1\" in --version|-c) printf 'fixture engine 1.2\\n'; exit 0;; esac\nprintf '%s\\n' \"$PWD\" > {}\nprintf '%s\\n' \"$@\" > {}\nexec {} \"$@\"\n",
                quote(&state.path().join("cwd.txt")),
                quote(&state.path().join("argv.txt")),
                quote(Path::new(env!("CARGO_BIN_EXE_laya-product-fixture")))
            ),
        );
        let config = state.path().join("engines.json");
        let document = json!({"schema_version":1,"comparison":{"model":"opaque/comparison"},"arms":[{
            "label":"external.fixture", "engine":engine,"executable":"./engine fixture", "model":"not-present.GGUF",
            "context_size":131072, "max_concurrency":4,"tokenizer":"not-present-tokenizer", "hf_config":"not-present-config",
            "prefix_cache":false,"extra_args":["--fixture-extra","two words"]}]});
        write_json(&config, &document);
        Self {
            state,
            executable,
            config,
        }
    }
    fn plan(&self, output: &Path) -> Output {
        self.command()
            .args(["external-config", "--config"])
            .arg(&self.config)
            .args(["--model", "opaque/comparison", "--output"])
            .arg(output)
            .output()
            .unwrap()
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command.args(["automation", "replay-matrix"]);
        command
    }
    fn modify(&self, mutate: impl FnOnce(&mut Value)) {
        let mut document: Value =
            serde_json::from_slice(&std::fs::read(&self.config).unwrap()).unwrap();
        mutate(&mut document);
        write_json(&self.config, &document);
    }
}

#[test]
fn config_probes_exact_identity_and_constructs_each_engine_command() {
    for (engine, required, forbidden) in [
        ("llama", "--no-cache-prompt", "--load-format"),
        ("vllm", "--no-enable-prefix-caching", "--no-cache-prompt"),
        ("sglang", "--disable-radix-cache", "--hf-config-path"),
    ] {
        let fixture = Fixture::new(engine);
        let output = fixture.state.path().join("plan.json");
        assert_success(&fixture.plan(&output));
        let plan: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        let build = &plan["builds"][0];
        assert_eq!(
            build["version_sha256"],
            hex::encode(Sha256::digest(b"fixture engine 1.2"))
        );
        assert_eq!(build["commit"], build["version_sha256"]);
        assert_eq!(build["served_model"], "opaque/comparison");
        assert_eq!(
            plan["engine_config"]["sha256"],
            hex::encode(Sha256::digest(std::fs::read(&fixture.config).unwrap()))
        );
        let args = plan["external_server_commands"][0].as_array().unwrap();
        assert!(args.iter().any(|value| value == required));
        assert!(!args.iter().any(|value| value == forbidden));
        assert_eq!(args.last().unwrap(), "two words");
        assert_eq!(
            args[0],
            serde_json::to_value(fixture.executable.canonicalize().unwrap()).unwrap()
        );
        if engine != "llama" {
            assert!(args.iter().any(|value| value == "--load-format"));
        }
    }
}

#[test]
fn malformed_config_and_overlap_admission_do_not_launch_an_engine() {
    for (field, value) in [
        ("context_size", json!(true)),
        ("context_size", json!(1.0)),
        ("max_concurrency", json!(0)),
        ("prefix_cache", json!(1)),
        ("label", json!("..")),
        ("extra_args", json!([7])),
    ] {
        let fixture = Fixture::new("vllm");
        fixture.modify(|document| document["arms"][0][field] = value);
        assert!(
            !fixture
                .plan(&fixture.state.path().join("bad.json"))
                .status
                .success()
        );
        assert!(!fixture.state.path().join("argv.txt").exists());
    }
    let fixture = Fixture::new("vllm");
    fixture.modify(|document| {
        let duplicate = document["arms"][0].clone();
        document["arms"].as_array_mut().unwrap().push(duplicate)
    });
    assert!(
        !fixture
            .plan(&fixture.state.path().join("duplicate.json"))
            .status
            .success()
    );
}

#[test]
fn version_resolution_preserves_virtualenv_style_symlinks_and_cwd() {
    let fixture = Fixture::new("sglang");
    let link = fixture.state.path().join("venv-python");
    std::os::unix::fs::symlink(&fixture.executable, &link).unwrap();
    fixture.modify(|document| document["arms"][0]["executable"] = "./venv-python".into());
    let output = fixture.state.path().join("plan.json");
    assert_success(&fixture.plan(&output));
    let plan: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(
        plan["builds"][0]["provenance"]["resolved_executable"],
        serde_json::to_value(
            fixture
                .state
                .path()
                .canonicalize()
                .unwrap()
                .join(link.file_name().unwrap())
        )
        .unwrap()
    );
    assert_eq!(
        plan["external_server_commands"][0][0],
        plan["builds"][0]["provenance"]["resolved_executable"]
    );
}

#[test]
fn external_cell_uses_owned_http_server_without_mesh_runtime_or_model_files() {
    let fixture = Fixture::new("llama");
    let plan_path = fixture.state.path().join("plan.json");
    assert_success(&fixture.plan(&plan_path));
    let plan: Value = serde_json::from_slice(&std::fs::read(plan_path).unwrap()).unwrap();
    let port = std::net::TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let workload = fixture.state.path().join("workload.json");
    write_json(
        &workload,
        &json!({"trajectories":[{"session_id":"session","agent_framework":"fixture","source_dataset":"fixture",
        "messages":[{"role":"user","content":"fixture"},{"role":"assistant","content":"recorded"}]}],
        "model":"pending","base_url":format!("http://127.0.0.1:{port}/v1"),"concurrency":1,
        "max_output_tokens":2048,"request_timeout_seconds":2}),
    );
    let input = fixture.state.path().join("cell.json");
    let requests = fixture.state.path().join("requests.jsonl");
    let summary = fixture.state.path().join("summary.json");
    write_json(
        &input,
        &json!({"build":plan["builds"][0],"workload":workload,
        "requests_output":requests,"summary_output":summary,"server_log":fixture.state.path().join("server.log"),
        "startup_timeout_seconds":3,"timeout_seconds":10}),
    );
    assert_success(
        &fixture
            .command()
            .args(["external-cell", "--input"])
            .arg(&input)
            .output()
            .unwrap(),
    );
    let summary: Value = serde_json::from_slice(&std::fs::read(summary).unwrap()).unwrap();
    assert_eq!(summary["model_id"], "laya-fixture");
    assert!(
        !std::fs::read_to_string(fixture.state.path().join("argv.txt"))
            .unwrap()
            .contains("--console")
    );
    assert_eq!(
        std::fs::read_to_string(fixture.state.path().join("cwd.txt"))
            .unwrap()
            .trim(),
        fixture
            .state
            .path()
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap()
    );
    assert!(requests.exists());
    let lifecycle: Value = serde_json::from_slice(
        &std::fs::read(fixture.state.path().join("lifecycle.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(lifecycle["infrastructure_clean"], true);
}

#[test]
fn long_context_mesh_qualification_rejects_external_before_version_probe() {
    let fixture = Fixture::new("vllm");
    write_executable(&fixture.executable, "#!/bin/sh\nexit 99\n");
    let output = fixture.state.path().join("plan.json");
    let result = fixture
        .command()
        .args(["external-config", "--config"])
        .arg(&fixture.config)
        .args([
            "--model",
            "opaque/comparison",
            "--minimum-context",
            "131072",
            "--output",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("requires mesh arms"));
    assert!(!output.exists());
}

#[test]
fn empty_and_failed_version_probes_never_create_a_plan() {
    for text in [
        "#!/bin/sh\nexit 0\n",
        "#!/bin/sh\nprintf 'failed version\\n' >&2\nexit 7\n",
    ] {
        let fixture = Fixture::new("vllm");
        write_executable(&fixture.executable, text);
        let output = fixture.state.path().join("plan.json");
        assert!(!fixture.plan(&output).status.success());
        assert!(!output.exists());
    }
    let fixture = Fixture::new("llama.cpp");
    write_executable(
        &fixture.executable,
        "#!/bin/sh\nprintf 'stderr version\\n' >&2\n",
    );
    let output = fixture.state.path().join("plan.json");
    assert_success(&fixture.plan(&output));
    let plan: Value = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(plan["builds"][0]["version"], "stderr version");
}

#[test]
fn interrupted_version_probe_cleans_its_descendant_and_preserves_sentinel() {
    let fixture = Fixture::new("vllm");
    let marker = fixture.state.path().join("version-child.pid");
    write_executable(
        &fixture.executable,
        &format!(
            "#!/bin/sh\n/bin/sleep 60 &\nprintf '%s\\n' \"$!\" > {}\nwait\n",
            quote(&marker)
        ),
    );
    let mut sentinel = Sentinel(Command::new("/bin/sleep").arg("60").spawn().unwrap());
    let output = fixture.state.path().join("plan.json");
    let mut child = fixture
        .command()
        .args(["external-config", "--config"])
        .arg(&fixture.config)
        .args(["--model", "opaque/comparison", "--output"])
        .arg(&output)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let started = std::time::Instant::now();
    while !marker.exists() && started.elapsed() < std::time::Duration::from_secs(5) {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        Command::new("/bin/kill")
            .args(["-TERM", &child.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    while child.try_wait().unwrap().is_none()
        && started.elapsed() < std::time::Duration::from_secs(12)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    if child.try_wait().unwrap().is_none() {
        child.kill().unwrap();
    }
    assert!(!child.wait().unwrap().success());
    assert!(marker.exists(), "version fixture never started");
    let pid = std::fs::read_to_string(marker).unwrap();
    let status = Command::new("/bin/ps")
        .args(["-o", "stat=", "-p", pid.trim()])
        .output()
        .unwrap();
    assert!(
        !status.status.success()
            || String::from_utf8_lossy(&status.stdout)
                .trim()
                .starts_with('Z')
    );
    assert!(sentinel.0.try_wait().unwrap().is_none());
    assert!(!output.exists());
}

struct Sentinel(std::process::Child);
impl Drop for Sentinel {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn quote(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
}
fn write_json(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
fn write_executable(path: &Path, text: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, text).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}
fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
