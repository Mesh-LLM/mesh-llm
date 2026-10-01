use super::options::Options;
use crate::{
    command::DynResult,
    process::{
        self, Value,
        retained::{Launch, MemberId},
    },
};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) struct Prepared {
    pub directory: PathBuf,
    pub work: PathBuf,
    pub initial: Launch,
    pub restarted: Launch,
    pub browser: Launch,
    pub browser_wait: Duration,
}

struct NodeInputs<'a> {
    root: &'a Path,
    options: &'a Options,
    directory: &'a Path,
    work: &'a Path,
    config: &'a Path,
}

pub(super) fn prepare(root: &Path, options: &Options) -> DynResult<Prepared> {
    let mut random = [0; 16];
    getrandom::fill(&mut random).map_err(|_| "logging evidence entropy unavailable")?;
    let directory = options
        .evidence_root
        .join(format!("logging-console-e2e-{}", hex::encode(random)));
    std::fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    let work = directory.join("state");
    for name in [
        "state/home",
        "state/runtime",
        "state/application-state",
        "logs",
        "requests",
        "playwright",
    ] {
        std::fs::create_dir_all(directory.join(name))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&work, std::fs::Permissions::from_mode(0o700))?;
    }
    let config = work.join("logging.toml");
    let state_path = toml::Value::String(
        work.join("application-state")
            .to_string_lossy()
            .into_owned(),
    )
    .to_string();
    std::fs::write(
        &config,
        format!(
            "[logging]\nenabled = true\napplication_state_root = {state_path}\nretention_max_rows = 64\nreplay_capacity = 1\ncleanup_cadence_secs = 300\n[logging.artifact]\ncapture_mode = \"redacted_artifacts\"\n"
        ),
    )?;
    super::evidence::write_json(
        &directory.join("manifest.json"),
        &serde_json::json!({
            "binary":options.binary,"evidence_dir":directory,"isolated_work_root":work,
            "api_port":options.base_port,"console_port":options.base_port+1,"logs_api_routes_mocked":false
        }),
    )?;
    let inputs = NodeInputs {
        root,
        options,
        directory: &directory,
        work: &work,
        config: &config,
    };
    let initial = daemon(&inputs, MemberId::Seed, "initial");
    let restarted = daemon(&inputs, MemberId::Seed.next_generation()?, "restarted");
    let browser_wait = options.wait() + Duration::from_secs(600);
    let mut environment = inherited();
    for (key, value) in [
        ("MESH_LOGS_E2E", "1".into()),
        (
            "MESH_LOGS_E2E_BASE_URL",
            format!("http://127.0.0.1:{}", options.base_port + 1),
        ),
        (
            "MESH_LOGS_E2E_OPENAI_URL",
            format!("http://127.0.0.1:{}/v1/chat/completions", options.base_port),
        ),
        (
            "PLAYWRIGHT_OUTPUT_DIR",
            directory.join("playwright").to_string_lossy().into_owned(),
        ),
        (
            "PLAYWRIGHT_JSON_REPORT",
            directory
                .join("playwright/report.json")
                .to_string_lossy()
                .into_owned(),
        ),
    ] {
        environment.insert(key.into(), Value::Public(value.into()));
    }
    let browser = Launch {
        member: MemberId::WorkerOne,
        spec: process::ProcessSpec {
            executable: executable("pnpm")?,
            cwd: root.join("mesh/crates/mesh-llm-ui"),
            environment,
            arguments: [
                "exec",
                "playwright",
                "test",
                "e2e/logs/real-console.spec.ts",
                "--workers=1",
            ]
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect(),
        },
        files: process::OutputFiles {
            stdout: Some(directory.join("logs/playwright.stdout.log")),
            stderr: Some(directory.join("logs/playwright.stderr.log")),
        },
        readiness_deadline: browser_wait,
    };
    let commands = [
        serde_json::json!({"name":"start-node","path":directory.join("logs/initial.stdout.log")}),
        serde_json::json!({"name":"restart-node","path":directory.join("logs/restarted.stdout.log")}),
        serde_json::json!({"name":"playwright-real-console","path":directory.join("logs/playwright.stdout.log")}),
    ];
    let mut bytes = Vec::new();
    for command in commands {
        serde_json::to_writer(&mut bytes, &command)?;
        bytes.push(b'\n');
    }
    std::fs::write(directory.join("commands.jsonl"), bytes)?;
    Ok(Prepared {
        directory,
        work,
        initial,
        restarted,
        browser,
        browser_wait,
    })
}

fn daemon(inputs: &NodeInputs<'_>, member: MemberId, label: &str) -> Launch {
    let mut environment = inherited();
    environment.insert(
        "HOME".into(),
        Value::Public(inputs.work.join("home").into()),
    );
    environment.insert(
        "MESH_LLM_RUNTIME_ROOT".into(),
        Value::Public(inputs.work.join("runtime").into()),
    );
    environment.insert("MESH_LLM_EPHEMERAL_KEY".into(), Value::Public("1".into()));
    Launch {
        member,
        spec: process::ProcessSpec {
            executable: inputs.options.binary.clone(),
            cwd: inputs.root.to_owned(),
            environment,
            arguments: [
                "serve".into(),
                "--log-format".into(),
                "json".into(),
                "--config".into(),
                inputs.config.to_string_lossy().into_owned(),
                "--port".into(),
                inputs.options.base_port.to_string(),
                "--console".into(),
                (inputs.options.base_port + 1).to_string(),
                "--bind-port".into(),
                (inputs.options.base_port + 2).to_string(),
            ]
            .into_iter()
            .map(|value: String| Value::Public(value.into()))
            .collect(),
        },
        files: process::OutputFiles {
            stdout: Some(inputs.directory.join(format!("logs/{label}.stdout.log"))),
            stderr: Some(inputs.directory.join(format!("logs/{label}.stderr.log"))),
        },
        readiness_deadline: inputs.options.wait(),
    }
}

fn inherited() -> std::collections::BTreeMap<std::ffi::OsString, Value> {
    std::env::vars_os()
        .map(|(key, value)| {
            let sensitive = key.to_string_lossy().to_ascii_uppercase();
            let secret = ["TOKEN", "SECRET", "PASSWORD", "API_KEY"]
                .iter()
                .any(|part| sensitive.contains(part));
            (
                key,
                if secret && !value.is_empty() {
                    Value::Secret(value)
                } else {
                    Value::Public(value)
                },
            )
        })
        .collect()
}

fn executable(name: &str) -> DynResult<PathBuf> {
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let path = directory.join(name);
        if path.is_file() {
            return Ok(path.canonicalize()?);
        }
    }
    Err("logging console executable missing".into())
}
