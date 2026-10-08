//! Real hidden CLI and rendered SWE template contracts, without an upstream SDK or model.
#![cfg(unix)]
use std::{
    fs,
    os::unix::{fs::PermissionsExt, process::CommandExt},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const BINARY: &str = env!("CARGO_BIN_EXE_skippy-bench");
const OLD: &str = r#"f"RUN /root/python3.11/bin/pip3 install --no-cache-dir {PACKAGE_NAME}\n\n""#;

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "swerex-cli-{}-{}-{}",
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
    fn file(&self, relative: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }
    fn executable(&self, relative: &str, script: &str) -> PathBuf {
        let path = self.file(relative, script);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run_bounded(command: &mut Command) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            let pid = i32::try_from(child.id()).unwrap();
            // SAFETY: this child owns a new process group; terminate that owned group only.
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
            let _ = child.wait();
            panic!("bounded SWE fixture command timed out");
        }
        thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}

#[test]
fn hidden_cli_preserves_quoted_url_identity_and_skips_model_directory_preparation() {
    let fixture = Fixture::new();
    let environment = fixture.0.join("environment with spaces");
    let module = fixture.file("environment with spaces/docker.py", OLD);
    let poison_data = fixture.0.join("must-not-create-model-data");
    let url = "http://[::1]:8080/simple?one=1&selected=[abc]";
    for _ in 0..2 {
        let output = run_bounded(
            Command::new(BINARY)
                .args(["eval", "patch-swerex-index", "--module-source"])
                .arg(&module)
                .arg("--environment-root")
                .arg(&environment)
                .args(["--index-url", url])
                .env("MESH_LLM_DATA_DIR", &poison_data),
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        assert!(output.stderr.is_empty());
        assert!(!poison_data.exists());
    }
    assert_eq!(
        fs::read_to_string(module).unwrap(),
        OLD.replace(
            "--no-cache-dir",
            &format!("--index-url '{url}' --no-cache-dir")
        )
    );
}

#[test]
fn hidden_cli_bad_url_and_unknown_source_refuse_without_disclosing_credentials() {
    let fixture = Fixture::new();
    let module = fixture.file("environment/docker.py", OLD);
    let before = fs::read(&module).unwrap();
    let output = run_bounded(
        Command::new(BINARY)
            .args(["eval", "patch-swerex-index", "--module-source"])
            .arg(&module)
            .arg("--environment-root")
            .arg(fixture.0.join("environment"))
            .args(["--index-url", "https://user:credential-marker@host/bad'url"]),
    );
    assert!(!output.status.success());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("credential-marker"));
    assert_eq!(fs::read(&module).unwrap(), before);
    fs::write(&module, "unknown source").unwrap();
    let output = run_bounded(
        Command::new(BINARY)
            .args(["eval", "patch-swerex-index", "--module-source"])
            .arg(&module)
            .arg("--environment-root")
            .arg(fixture.0.join("environment"))
            .args(["--index-url", "https://host/simple"]),
    );
    assert!(!output.status.success());
    assert_eq!(fs::read_to_string(module).unwrap(), "unknown source");
}

fn render_template(fixture: &Fixture, url: &str) -> PathBuf {
    let helper = fixture.0.join("native helper with spaces");
    fs::copy(BINARY, &helper).unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o755)).unwrap();
    let cache = fixture.0.join("cache with spaces");
    let output = fixture.0.join("run with spaces");
    let result = run_bounded(
        Command::new(helper)
            .args([
                "eval",
                "run",
                "swe-bench-pro",
                "--dry-run",
                "--model",
                "fixture",
                "--api-key",
                "fixture-secret",
            ])
            .arg("--cache-root")
            .arg(&cache)
            .arg("--output-dir")
            .arg(&output)
            .env("MESH_LLM_DATA_DIR", fixture.0.join("dry-run-data"))
            .env_remove("SWE_BENCH_PRO_PYTHON")
            .env("SWE_BENCH_PRO_DEPLOYMENT_TYPE", "docker")
            .env("SWE_BENCH_PRO_SWEREX_SPEC", "swe-rex[modal]==1.4.0")
            .env("SWE_BENCH_PRO_SWEREX_PIP_INDEX_URL", url),
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let script = output.join("raw/swe-bench-pro-run.sh");
    let text = fs::read_to_string(&script).unwrap();
    assert!(text.contains("cache with spaces/swe-sdk-v1/environment/bin/python"));
    assert!(text.contains("\"$PREPARED_PYTHON\" -I -B -m sweagent.run.run run-batch"));
    assert!(!text.contains("fixture-secret"));
    assert!(!text.contains("text.replace(old, new)"));
    script
}

fn template_environment(fixture: &Fixture) -> (PathBuf, PathBuf) {
    let module=fixture.file("cache with spaces/swe-sdk-v1/environment/lib/python3.11/site-packages/swerex/deployment/docker.py","already sealed source");
    fs::create_dir_all(
        fixture
            .0
            .join("cache with spaces/swe-sdk-v1/project/source/swe-bench-pro/SWE-agent"),
    )
    .unwrap();
    fs::create_dir_all(fixture.0.join("cache with spaces/harnesses/swe-bench-pro")).unwrap();
    fixture.executable("cache with spaces/swe-sdk-v1/environment/bin/python",r#"#!/bin/sh
printf '%s\n' "$@" >> "$SDK_ARGS"
if [ "$1" != "-I" ] || [ "$2" != "-B" ]; then exit 92; fi
if [ "$3" = "-m" ]; then
 printf 'run-batch\n' >> "$SDK_EVENTS"
 exit "${SDK_AGENT_STATUS:-0}"
else
 case "$3" in
  */swe-generate-instances.py) printf 'generator\n' >> "$SDK_EVENTS";exit "${SDK_GENERATOR_STATUS:-0}" ;;
  */swe-expert-instances.py) printf 'expert\n' >> "$SDK_EVENTS" ;;
  */swe-evaluate.py) printf 'evaluator\n' >> "$SDK_EVENTS" ;;
  */gather_patches.py) printf 'gather\n' >> "$SDK_EVENTS" ;;
  *) exit 94 ;;
 esac
fi
"#);
    (module, fixture.0.join("events"))
}
fn execute_template(
    fixture: &Fixture,
    script: &Path,
    generator_status: i32,
    agent_status: i32,
) -> Output {
    run_bounded(
        Command::new("zsh")
            .arg(script)
            .env("SKIPPY_BENCH_API_KEY", "fixture-secret")
            .env("SDK_EVENTS", fixture.0.join("events"))
            .env("SDK_ARGS", fixture.0.join("sdk-args"))
            .env("SDK_STDIN", fixture.0.join("sdk-body"))
            .env("SDK_GENERATOR_STATUS", generator_status.to_string())
            .env("SDK_AGENT_STATUS", agent_status.to_string()),
    )
}
#[test]
fn rendered_template_uses_sealed_sdk_and_preserves_agent_failure_without_runtime_patching() {
    let fixture = Fixture::new();
    let script = render_template(&fixture, "https://host/simple");
    let (module, events) = template_environment(&fixture);
    let before = fs::read(&module).unwrap();
    let output = execute_template(&fixture, &script, 0, 67);
    assert_eq!(
        output.status.code(),
        Some(67),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(events).unwrap(),
        "generator\nexpert\nrun-batch\n"
    );
    assert_eq!(fs::read(module).unwrap(), before);
    let argv = fs::read_to_string(fixture.0.join("sdk-args")).unwrap();
    assert!(argv.contains("-I\n-B\n-m\nsweagent.run.run\nrun-batch\n"));
    assert!(argv.contains("--agent.model.api_base\n"));
}
#[test]
fn rendered_template_success_finishes_gather_and_evaluator_without_sdk_mutation() {
    let fixture = Fixture::new();
    let script = render_template(&fixture, "https://host/simple");
    let (module, events) = template_environment(&fixture);
    let before = fs::read(&module).unwrap();
    let output = execute_template(&fixture, &script, 0, 0);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(events).unwrap(),
        "generator\nexpert\nrun-batch\ngather\nevaluator\n"
    );
    assert_eq!(fs::read(module).unwrap(), before);
    assert!(
        fs::read_to_string(fixture.0.join("sdk-args"))
            .unwrap()
            .contains("sweap_eval_full_v2.jsonl")
    );
}
#[test]
fn rendered_template_generator_refusal_prevents_agent_and_evaluator() {
    let fixture = Fixture::new();
    let script = render_template(&fixture, "https://host/simple");
    let (module, events) = template_environment(&fixture);
    let before = fs::read(&module).unwrap();
    let output = execute_template(&fixture, &script, 23, 0);
    assert_eq!(output.status.code(), Some(23));
    assert_eq!(fs::read_to_string(events).unwrap(), "generator\n");
    assert!(!fixture.0.join("sdk-body").exists());
    assert_eq!(fs::read(module).unwrap(), before);
}
#[test]
fn actual_dry_run_refuses_ambient_credential_index_before_output_or_secret_publication() {
    for url in [
        "https://user:credential-marker@host/simple",
        "https://host/simple?token=credential-marker",
    ] {
        let fixture = Fixture::new();
        let output_dir = fixture.0.join("must-not-exist");
        let cache = fixture.0.join("cache");
        let result = run_bounded(
            Command::new(BINARY)
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
                .env("SWE_BENCH_PRO_SWEREX_PIP_INDEX_URL", url),
        );
        assert!(!result.status.success());
        assert!(!output_dir.exists());
        assert!(!cache.join("swe-sdk-v1").exists());
        assert!(!String::from_utf8_lossy(&result.stdout).contains("credential-marker"));
        assert!(!String::from_utf8_lossy(&result.stderr).contains("credential-marker"));
    }
}
#[test]
fn actual_safe_unprepared_dry_run_renders_without_retired_environment_fields() {
    let fixture = Fixture::new();
    let script = render_template(&fixture, "https://host/simple");
    let rendered = fs::read_to_string(script).unwrap();
    for removed in [
        "SWEREX_PIP_INDEX_URL=",
        "SWEREX_SPEC=",
        "SWEAGENT_PYTHON=",
        "ADAPTER_HELPER=",
    ] {
        assert!(!rendered.contains(removed), "{removed}");
    }
    assert!(rendered.contains("DEPLOYMENT_TYPE=docker\n"));
    assert!(rendered.contains("--dockerhub_username"));
    assert!(!fixture.0.join("cache with spaces/swe-sdk-v1").exists());
}
