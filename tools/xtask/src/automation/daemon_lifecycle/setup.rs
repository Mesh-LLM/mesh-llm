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
    pub launches: Vec<Launch>,
    pub version: Launch,
    pub auth: Launch,
}
pub(super) fn prepare(root: &Path, options: &Options) -> DynResult<Prepared> {
    let mut random = [0; 16];
    getrandom::fill(&mut random).map_err(|_| "daemon evidence entropy")?;
    let directory = options
        .evidence
        .join(format!("runtime-daemon-lifecycle-{}", hex::encode(random)));
    std::fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    for name in ["logs", "status", "control", "versions", "state"] {
        std::fs::create_dir_all(directory.join(name))?;
    }
    let key = directory.join("state/owner-keystore.json");
    let environment = || {
        std::env::vars_os()
            .map(|(key, value)| (key, Value::Public(value)))
            .collect()
    };
    let command = |label: &str, arguments: Vec<String>| -> DynResult<Launch> {
        Ok(Launch {
            member: MemberId::new(label, 0)?,
            spec: process::ProcessSpec {
                executable: options.binary.clone(),
                cwd: root.to_owned(),
                environment: environment(),
                arguments: arguments
                    .into_iter()
                    .map(|value| Value::Public(value.into()))
                    .collect(),
            },
            files: process::OutputFiles {
                stdout: Some(directory.join(format!("logs/{label}.stdout.log"))),
                stderr: Some(directory.join(format!("logs/{label}.stderr.log"))),
            },
            readiness_deadline: options.wait,
        })
    };
    let version = command(
        "version",
        vec!["--log-format".into(), "json".into(), "--version".into()],
    )?;
    let auth = command(
        "owner-auth",
        vec![
            "--log-format".into(),
            "json".into(),
            "auth".into(),
            "init".into(),
            "--owner-key".into(),
            key.to_string_lossy().into_owned(),
            "--no-passphrase".into(),
            "--force".into(),
        ],
    )?;
    let mut launches = Vec::new();
    for (index, label) in [
        "zero-model",
        "mode-serve",
        "mode-on-demand",
        "best-effort",
        "fail-fast",
    ]
    .iter()
    .enumerate()
    {
        let home = directory.join(format!("state/{label}/home"));
        let runtime = directory.join(format!("state/{label}/runtime"));
        std::fs::create_dir_all(&home)?;
        std::fs::create_dir_all(&runtime)?;
        let base = options.base + u16::try_from(index)? * 4;
        let mut launch = command(
            label,
            vec![
                "--log-format".into(),
                "json".into(),
                "--owner-key".into(),
                key.to_string_lossy().into_owned(),
                "serve".into(),
                "--headless".into(),
                "--port".into(),
                base.to_string(),
                "--console".into(),
                (base + 1).to_string(),
                "--bind-port".into(),
                (base + 2).to_string(),
            ],
        )?;
        launch
            .spec
            .environment
            .insert("HOME".into(), Value::Public(home.into()));
        launch.spec.environment.insert(
            "MESH_LLM_RUNTIME_ROOT".into(),
            Value::Public(runtime.into()),
        );
        launch
            .spec
            .environment
            .insert("MESH_LLM_EPHEMERAL_KEY".into(), Value::Public("1".into()));
        if index >= 2 {
            let config = directory.join(format!("state/{label}.toml"));
            let text = match index {
                2 => "[runtime]\nmode = \"on_demand\"\n",
                3 => "[runtime]\nstartup_failure_policy = \"best_effort\"\n",
                _ => "[runtime]\nstartup_failure_policy = \"fail_fast\"\n",
            };
            std::fs::write(&config, text)?;
            launch.spec.arguments.extend([
                Value::Public("--config".into()),
                Value::Public(config.into()),
            ]);
        }
        if index >= 3 {
            launch.spec.arguments.extend([
                Value::Public("--model".into()),
                Value::Public("NonExistent-Model-That-Does-Not-Exist-Q4_K_M".into()),
            ]);
        }
        launches.push(launch);
    }
    Ok(Prepared {
        directory,
        launches,
        version,
        auth,
    })
}
