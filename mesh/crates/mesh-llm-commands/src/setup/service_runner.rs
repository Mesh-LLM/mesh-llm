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

    /// Runs `command` and returns its standard output.
    fn output(&mut self, command: &ServiceCommand) -> Result<String> {
        self.run(command).map(|()| String::new())
    }
}

pub(crate) struct CliServiceCommandRunner;

impl ServiceCommandRunner for CliServiceCommandRunner {
    fn run(&mut self, command: &ServiceCommand) -> Result<()> {
        self.output(command).map(|_| ())
    }

    // The tools' own output is captured rather than printed, so setup's summary
    // stays the only output, and a failure carries what the tool said.
    fn output(&mut self, command: &ServiceCommand) -> Result<String> {
        let output = Command::new(&command.program)
            .args(&command.args)
            .output()
            .with_context(|| format!("failed to run `{}`", command.display()))?;
        if output.status.success() {
            return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
        }
        let said = [&output.stderr, &output.stdout]
            .iter()
            .map(|stream| String::from_utf8_lossy(stream).trim().to_string())
            .find(|text| !text.is_empty());
        match said {
            Some(said) => Err(anyhow!(
                "`{}` exited with status {}: {said}",
                command.display(),
                output.status
            )),
            None => Err(anyhow!(
                "`{}` exited with status {}",
                command.display(),
                output.status
            )),
        }
    }
}
