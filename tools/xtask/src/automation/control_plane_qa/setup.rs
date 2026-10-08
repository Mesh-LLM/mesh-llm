mod builder;
mod control;
mod mesh;
mod protocol;
use super::{coordinator::Step, options::Options};
use crate::command::DynResult;
use builder::Builder;
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
};

pub(super) struct Prepared {
    pub directory: PathBuf,
    pub steps: VecDeque<Step>,
}

pub(super) fn prepare(root: &Path, options: &Options) -> DynResult<Prepared> {
    if std::fs::read(&options.current)? == std::fs::read(&options.released)? {
        return Err("released and current executables must have distinct content".into());
    }
    let mut random = [0; 16];
    getrandom::fill(&mut random).map_err(|_| "control evidence entropy")?;
    let directory = options.evidence.join(format!(
        "control-plane-mixed-version-{}",
        hex::encode(random)
    ));
    std::fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    for name in [
        "logs", "control", "state", "status", "models", "chat", "versions",
    ] {
        std::fs::create_dir_all(directory.join(name))?;
    }
    let mut builder = Builder {
        root,
        options,
        directory,
        steps: VecDeque::new(),
    };
    prerequisites(&mut builder)?;
    mesh::append(&mut builder)?;
    protocol::append(&mut builder)?;
    control::append(&mut builder)?;
    Ok(Prepared {
        directory: builder.directory,
        steps: builder.steps,
    })
}

fn prerequisites(builder: &mut Builder<'_>) -> DynResult<()> {
    let options = builder.options;
    for (name, binary) in [
        ("released-version", &options.released),
        ("current-version", &options.current),
    ] {
        builder.command(
            name,
            binary,
            vec!["--log-format".into(), "json".into(), "--version".into()],
            true,
        )?;
    }
    for (name, binary) in [
        ("released-help", &options.released),
        ("current-help", &options.current),
    ] {
        builder.command(name, binary, vec!["--help".into()], true)?;
    }
    builder.steps.push_back(Step::Capabilities {
        current: builder.directory.join("logs/current-help.stdout.log"),
        released: builder.directory.join("logs/released-help.stdout.log"),
    });
    for (name, file) in [
        ("shared-owner-auth", "owner.json"),
        ("wrong-owner-auth", "wrong-owner.json"),
    ] {
        builder.command(
            name,
            &options.current,
            vec![
                "--log-format".into(),
                "json".into(),
                "auth".into(),
                "init".into(),
                "--owner-key".into(),
                builder
                    .directory
                    .join("state")
                    .join(file)
                    .to_string_lossy()
                    .into_owned(),
                "--no-passphrase".into(),
                "--force".into(),
            ],
            true,
        )?;
    }
    Ok(())
}
