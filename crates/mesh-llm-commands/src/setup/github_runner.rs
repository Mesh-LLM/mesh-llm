use std::fmt::{self, Display, Formatter};
use std::io;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const GITHUB_REPOSITORY: &str = "Mesh-LLM/mesh-llm";
const GH_COMMAND_TIMEOUT: Duration = Duration::from_secs(10);
const GH_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GhCommand {
    CheckAvailability,
    CheckAuthentication,
    CheckViewerHasStarred,
    StarRepository,
}

impl GhCommand {
    const fn args(self) -> &'static [&'static str] {
        match self {
            Self::CheckAvailability => &["--version"],
            Self::CheckAuthentication => {
                &["auth", "status", "--active", "--hostname", "github.com"]
            }
            Self::CheckViewerHasStarred => &[
                "repo",
                "view",
                GITHUB_REPOSITORY,
                "--json",
                "viewerHasStarred",
                "--jq",
                ".viewerHasStarred",
            ],
            Self::StarRepository => &[
                "api",
                "--method",
                "PUT",
                "/user/starred/Mesh-LLM/mesh-llm",
                "--silent",
            ],
        }
    }

    pub(crate) const fn display_name(self) -> &'static str {
        match self {
            Self::CheckAvailability => "gh --version",
            Self::CheckAuthentication => "gh auth status --active --hostname github.com",
            Self::CheckViewerHasStarred => {
                "gh repo view Mesh-LLM/mesh-llm --json viewerHasStarred --jq .viewerHasStarred"
            }
            Self::StarRepository => "gh api --method PUT /user/starred/Mesh-LLM/mesh-llm --silent",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GhCommandOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GhCommandError {
    NotOnPath,
    SpawnFailed(String),
    WaitFailed(String),
    TimedOut(&'static str),
    KillFailed(String),
}

impl Display for GhCommandError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotOnPath => f.write_str("GitHub CLI is not on PATH"),
            Self::SpawnFailed(error) => write!(f, "failed to start gh: {error}"),
            Self::WaitFailed(error) => write!(f, "failed while waiting for gh: {error}"),
            Self::TimedOut(command) => write!(f, "timed out running `{command}`"),
            Self::KillFailed(error) => write!(f, "failed to stop timed out gh command: {error}"),
        }
    }
}

pub(crate) trait GhCommandRunner {
    fn run(&mut self, command: GhCommand) -> Result<GhCommandOutput, GhCommandError>;
}

pub(crate) struct ProcessGhCommandRunner {
    timeout: Duration,
}

impl Default for ProcessGhCommandRunner {
    fn default() -> Self {
        Self {
            timeout: GH_COMMAND_TIMEOUT,
        }
    }
}

impl GhCommandRunner for ProcessGhCommandRunner {
    fn run(&mut self, command: GhCommand) -> Result<GhCommandOutput, GhCommandError> {
        let mut child = Command::new("gh")
            .args(command.args())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| match error.kind() {
                io::ErrorKind::NotFound => GhCommandError::NotOnPath,
                _ => GhCommandError::SpawnFailed(error.to_string()),
            })?;

        let started_at = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(_)) => {
                    let output = child
                        .wait_with_output()
                        .map_err(|error| GhCommandError::WaitFailed(error.to_string()))?;
                    return Ok(GhCommandOutput {
                        success: output.status.success(),
                        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                    });
                }
                Ok(None) => {
                    if started_at.elapsed() >= self.timeout {
                        match child.try_wait() {
                            Ok(Some(_)) => {
                                let output = child.wait_with_output().map_err(|error| {
                                    GhCommandError::WaitFailed(error.to_string())
                                })?;
                                return Ok(GhCommandOutput {
                                    success: output.status.success(),
                                    stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                                });
                            }
                            Ok(None) => {
                                child.kill().map_err(|error| {
                                    GhCommandError::KillFailed(error.to_string())
                                })?;
                                let _ = child.wait();
                                return Err(GhCommandError::TimedOut(command.display_name()));
                            }
                            Err(error) => {
                                return Err(GhCommandError::WaitFailed(error.to_string()));
                            }
                        }
                    }
                    thread::sleep(GH_POLL_INTERVAL);
                }
                Err(error) => return Err(GhCommandError::WaitFailed(error.to_string())),
            }
        }
    }
}
