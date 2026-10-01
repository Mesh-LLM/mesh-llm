use super::options::Options;
use crate::{
    command::DynResult,
    process::{
        self, Value,
        retained::{Launch, MemberId},
    },
};
use std::path::{Path, PathBuf};
pub(super) struct Prepared {
    pub directory: PathBuf,
    pub initial: Launch,
    pub restarted: Launch,
    pub fail_open: Launch,
}
pub(super) fn prepare(root: &Path, options: &Options) -> DynResult<Prepared> {
    let mut random = [0; 16];
    getrandom::fill(&mut random).map_err(|_| "logging recovery entropy")?;
    let directory = options
        .evidence
        .join(format!("logging-recovery-{}", hex::encode(random)));
    std::fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    for name in ["logs", "requests", "state", "state/application-state"] {
        std::fs::create_dir_all(directory.join(name))?;
    }
    let config = directory.join("state/restart.toml");
    write_config(&config, &directory.join("state/application-state"), "")?;
    let unusable = directory.join("state/logging-state-file");
    std::fs::write(&unusable, b"")?;
    let fail_config = directory.join("state/fail-open.toml");
    write_config(&fail_config, &unusable, &options.endpoint)?;
    let initial = node(
        root,
        options,
        &directory,
        &config,
        MemberId::Seed,
        10,
        "restart-before",
    )?;
    let restarted = node(
        root,
        options,
        &directory,
        &config,
        MemberId::Seed.next_generation()?,
        20,
        "restart-after",
    )?;
    let fail_open = node(
        root,
        options,
        &directory,
        &fail_config,
        MemberId::WorkerOne,
        30,
        "fail-open",
    )?;
    Ok(Prepared {
        directory,
        initial,
        restarted,
        fail_open,
    })
}
fn write_config(path: &Path, state: &Path, endpoint: &str) -> DynResult<()> {
    let state = toml::Value::String(state.to_string_lossy().into_owned()).to_string();
    let mut text = format!(
        "[logging]\nenabled = true\napplication_state_root = {state}\nretention_max_rows = 64\nreplay_capacity = 1\ncleanup_cadence_secs = 300\n[logging.artifact]\ncapture_mode = \"metadata_only\"\n"
    );
    if !endpoint.is_empty() {
        text.push_str(&format!(
            "\n[[plugin]]\nname = \"openai-endpoint\"\nurl = {}\n",
            toml::Value::String(endpoint.into())
        ));
    }
    std::fs::write(path, text)?;
    Ok(())
}
fn node(
    root: &Path,
    options: &Options,
    directory: &Path,
    config: &Path,
    member: MemberId,
    offset: u16,
    label: &str,
) -> DynResult<Launch> {
    let home = directory.join(format!("state/{label}/home"));
    let runtime = directory.join(format!("state/{label}/runtime"));
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&runtime)?;
    let mut environment = std::env::vars_os()
        .map(|(key, value)| (key, Value::Public(value)))
        .collect::<std::collections::BTreeMap<_, _>>();
    environment.insert("HOME".into(), Value::Public(home.into()));
    environment.insert(
        "MESH_LLM_RUNTIME_ROOT".into(),
        Value::Public(runtime.into()),
    );
    environment.insert("MESH_LLM_EPHEMERAL_KEY".into(), Value::Public("1".into()));
    Ok(Launch {
        member,
        spec: process::ProcessSpec {
            executable: options.binary.clone(),
            cwd: root.to_owned(),
            environment,
            arguments: [
                "--log-format".into(),
                "json".into(),
                "serve".into(),
                "--headless".into(),
                "--config".into(),
                config.to_string_lossy().into_owned(),
                "--port".into(),
                (options.base + offset).to_string(),
                "--console".into(),
                (options.base + offset + 1).to_string(),
                "--bind-port".into(),
                (options.base + offset + 2).to_string(),
            ]
            .into_iter()
            .map(|value: String| Value::Public(value.into()))
            .collect(),
        },
        files: process::OutputFiles {
            stdout: Some(directory.join(format!("logs/{label}.stdout.log"))),
            stderr: Some(directory.join(format!("logs/{label}.stderr.log"))),
        },
        readiness_deadline: options.wait,
    })
}
