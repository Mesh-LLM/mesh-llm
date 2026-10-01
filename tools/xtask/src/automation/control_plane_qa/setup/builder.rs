use super::super::{coordinator::Step, http_checks::Check, options::Options};
use crate::{
    command::DynResult,
    process::{
        self, Value,
        retained::{Launch, MemberId},
    },
};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) struct Builder<'a> {
    pub root: &'a Path,
    pub options: &'a Options,
    pub directory: PathBuf,
    pub steps: VecDeque<Step>,
}

pub(super) enum Mode<'a> {
    Serve(&'a str),
    JoinedClient,
    PublicClient,
    WrongOwner,
}

pub(super) struct Node<'a> {
    pub name: &'a str,
    pub binary: &'a Path,
    pub offset: u16,
    pub mode: Mode<'a>,
}

impl Builder<'_> {
    pub fn command(
        &mut self,
        name: &str,
        binary: &Path,
        arguments: Vec<String>,
        prerequisite: bool,
    ) -> DynResult<()> {
        let launch = self.launch(
            name,
            binary,
            arguments,
            self.options.wait + Duration::from_secs(600),
        )?;
        self.steps.push_back(Step::Command {
            launch,
            prerequisite,
        });
        Ok(())
    }

    fn launch(
        &self,
        name: &str,
        binary: &Path,
        arguments: Vec<String>,
        deadline: Duration,
    ) -> DynResult<Launch> {
        let mut environment = std::env::vars_os()
            .map(|(key, value)| {
                let sensitive = key.to_string_lossy().to_ascii_uppercase();
                let secret = ["TOKEN", "PASSWORD", "SECRET", "API_KEY"]
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
            .collect::<std::collections::BTreeMap<_, _>>();
        let home = self.directory.join(format!("state/{name}/home"));
        let runtime = self.directory.join(format!("state/{name}/runtime"));
        std::fs::create_dir_all(&home)?;
        std::fs::create_dir_all(&runtime)?;
        environment.insert("HOME".into(), Value::Public(home.into()));
        environment.insert(
            "MESH_LLM_RUNTIME_ROOT".into(),
            Value::Public(runtime.into()),
        );
        environment.insert("MESH_LLM_EPHEMERAL_KEY".into(), Value::Public("1".into()));
        Ok(Launch {
            member: MemberId::new(name, 0)?,
            spec: process::ProcessSpec {
                executable: binary.to_owned(),
                cwd: self.root.to_owned(),
                environment,
                arguments: arguments
                    .into_iter()
                    .map(|value| Value::Public(value.into()))
                    .collect(),
            },
            files: process::OutputFiles {
                stdout: Some(self.directory.join(format!("logs/{name}.stdout.log"))),
                stderr: Some(self.directory.join(format!("logs/{name}.stderr.log"))),
            },
            readiness_deadline: deadline,
        })
    }

    pub fn node(&mut self, node: Node<'_>) -> DynResult<()> {
        let base = self.options.base + node.offset;
        let owner = self.directory.join(match node.mode {
            Mode::WrongOwner => "state/wrong-owner.json",
            Mode::Serve(_) | Mode::JoinedClient | Mode::PublicClient => "state/owner.json",
        });
        let mut arguments = vec![
            "--log-format".into(),
            "json".into(),
            "--owner-key".into(),
            owner.to_string_lossy().into_owned(),
            "--headless".into(),
            "--port".into(),
            base.to_string(),
            "--console".into(),
            (base + 1).to_string(),
        ];
        match node.mode {
            Mode::JoinedClient => {
                arguments.extend(["--client".into(), "--join".into(), "QA_JOIN_TOKEN".into()])
            }
            Mode::PublicClient => arguments.extend(["--client".into(), "--auto".into()]),
            Mode::Serve(model) => {
                arguments.extend(["serve".into(), "--bind-port".into(), (base + 2).to_string()]);
                if !model.is_empty() {
                    arguments.extend([
                        "--model".into(),
                        model.into(),
                        "--no-draft".into(),
                        "--device".into(),
                        "CPU".into(),
                        "--ctx-size".into(),
                        self.options.context.to_string(),
                    ]);
                }
            }
            Mode::WrongOwner => {
                arguments.extend(["serve".into(), "--bind-port".into(), (base + 2).to_string()])
            }
        }
        let launch = self.launch(node.name, node.binary, arguments, self.options.wait)?;
        let member = launch.member;
        self.steps.push_back(Step::Start(launch));
        self.steps.push_back(Step::Check {
            name: "node-ready-no-control-leak",
            check: Check::Ready { console: base + 1 },
        });
        self.steps.push_back(Step::Admit(member));
        Ok(())
    }
}
