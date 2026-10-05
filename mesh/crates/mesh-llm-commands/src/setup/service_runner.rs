use anyhow::{Context, Result, anyhow};
use std::process::Command;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ServiceCommand {
    pub(crate) program: String,
    pub(crate) args: Vec<String>,
}

impl ServiceCommand {
    pub(crate) fn new(
        program: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
        }
    }

    pub(crate) fn display(&self) -> String {
        let mut rendered = Vec::with_capacity(self.args.len() + 1);
        rendered.push(self.program.clone());
        rendered.extend(self.args.iter().cloned());
        rendered.join(" ")
    }
}

pub(crate) trait ServiceCommandRunner {
    fn run(&mut self, command: &ServiceCommand) -> Result<()>;
}

pub(crate) struct CliServiceCommandRunner;

impl ServiceCommandRunner for CliServiceCommandRunner {
    fn run(&mut self, command: &ServiceCommand) -> Result<()> {
        let status = Command::new(&command.program)
            .args(&command.args)
            .status()
            .with_context(|| format!("failed to run `{}`", command.display()))?;
        if status.success() {
            Ok(())
        } else {
            Err(anyhow!(
                "`{}` exited with status {status}",
                command.display()
            ))
        }
    }
}
