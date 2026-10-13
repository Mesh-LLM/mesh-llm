use super::options::Options;
use crate::{
    command::DynResult,
    process::{self, Value, retained::Launch},
};
use std::path::{Path, PathBuf};

pub(super) struct Prepared {
    pub work: PathBuf,
    pub process_root: PathBuf,
    pub process_owned: bool,
    pub launches: Vec<Launch>,
}
pub(super) fn prepare(root: &Path, options: &Options) -> DynResult<Prepared> {
    let mut random = [0; 16];
    getrandom::fill(&mut random).map_err(|_| "split state entropy unavailable")?;
    let id = hex::encode(random);
    let work = options
        .work
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(format!("mesh-split-recovery-{id}")));
    let process_root = options
        .process_root
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(format!("mesh-split-proc-{id}")));
    std::fs::create_dir_all(&work)?;
    std::fs::create_dir_all(&process_root)?;
    let work = work.canonicalize()?;
    let process_root = process_root.canonicalize()?;
    let mut launches = Vec::new();
    for index in 0..=options.workers {
        let label = if index == 0 {
            "seed".into()
        } else {
            format!("worker-{index}")
        };
        let home = process_root.join(&label).join("h");
        let runtime = process_root.join(&label).join("r");
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
        let vram = if index == 0 {
            &options.seed_vram
        } else {
            options
                .worker_vram
                .get(index - 1)
                .or_else(|| options.worker_vram.last())
                .ok_or("worker VRAM list empty")?
        };
        let offset = u16::try_from(index)?;
        let mut arguments = vec![
            "--log-format".into(),
            "json".into(),
            "serve".into(),
            "--model".into(),
            options.model.clone(),
            "--split".into(),
            "--no-draft".into(),
            "--ctx-size".into(),
            options.context.to_string(),
            "--max-vram".into(),
            vram.clone(),
            "--port".into(),
            (options.api + offset).to_string(),
            "--console".into(),
            (options.console + offset).to_string(),
            "--bind-port".into(),
            (options.bind + offset).to_string(),
            "--headless".into(),
        ];
        if !options.discovery.is_empty() {
            arguments.extend(["--mesh-discovery-mode".into(), options.discovery.clone()]);
        }
        if !options.device.is_empty() {
            arguments.extend(["--device".into(), options.device.clone()]);
        }
        launches.push(Launch {
            member: options.member(index)?,
            spec: process::ProcessSpec {
                executable: options.binary.clone(),
                cwd: root.to_owned(),
                environment,
                arguments: arguments
                    .into_iter()
                    .map(|value: String| Value::Public(value.into()))
                    .collect(),
            },
            files: process::OutputFiles {
                stdout: Some(work.join(format!("{label}.stdout.log"))),
                stderr: Some(work.join(format!("{label}.stderr.log"))),
            },
            readiness_deadline: options.startup,
        });
    }
    Ok(Prepared {
        work,
        process_root,
        process_owned: options.process_root.is_none(),
        launches,
    })
}
