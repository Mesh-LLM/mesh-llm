use super::github_runner::{GhCommand, GhCommandError, GhCommandOutput, GhCommandRunner};
use super::service_paths::ServiceInstallContext;
use super::service_runner::{ServiceCommand, ServiceCommandRunner};
use super::{
    SetupActions, SetupConfirmPrompt, SetupGitHubStarPlan, SetupPlatform, SetupPrompter, SetupStep,
};
use anyhow::anyhow;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::{Ready, ready};
use std::rc::Rc;
use tempfile::TempDir;

#[derive(Default)]
pub(super) struct FakePrompter {
    pub(super) prompts: Vec<SetupConfirmPrompt>,
    replies: VecDeque<Option<bool>>,
}

impl FakePrompter {
    pub(super) fn with_replies(replies: impl IntoIterator<Item = Option<bool>>) -> Self {
        Self {
            prompts: Vec::new(),
            replies: replies.into_iter().collect(),
        }
    }
}

impl SetupPrompter for FakePrompter {
    fn confirm(&mut self, prompt: SetupConfirmPrompt) -> Option<bool> {
        self.prompts.push(prompt);
        self.replies.pop_front().unwrap_or(None)
    }
}

#[derive(Default)]
pub(super) struct RecordingActions {
    pub(super) steps: Vec<SetupStep>,
    pub(super) github: Vec<SetupGitHubStarPlan>,
    failed_step: Option<SetupStep>,
}

impl RecordingActions {
    pub(super) fn failing_on(step: SetupStep) -> Self {
        Self {
            steps: Vec::new(),
            github: Vec::new(),
            failed_step: Some(step),
        }
    }
}

impl SetupActions for RecordingActions {
    type Error = anyhow::Error;
    type GitHubStarFuture<'a>
        = Ready<Result<(), Self::Error>>
    where
        Self: 'a;
    type StepFuture<'a>
        = Ready<Result<(), Self::Error>>
    where
        Self: 'a;

    fn run_step(&mut self, step: SetupStep) -> Self::StepFuture<'_> {
        self.steps.push(step);
        if self.failed_step == Some(step) {
            return ready(Err(anyhow!("simulated step failure for {step:?}")));
        }
        ready(Ok(()))
    }

    fn handle_github_star<'a>(
        &'a mut self,
        plan: SetupGitHubStarPlan,
        _prompter: &'a mut dyn SetupPrompter,
    ) -> Self::GitHubStarFuture<'a> {
        self.github.push(plan);
        ready(Ok(()))
    }
}

#[derive(Default)]
pub(super) struct SharedGhRunnerState {
    pub(super) commands: Vec<GhCommand>,
    responses: VecDeque<Result<GhCommandOutput, GhCommandError>>,
}

pub(super) struct SharedGhRunner {
    state: Rc<RefCell<SharedGhRunnerState>>,
}

impl SharedGhRunner {
    pub(super) fn new(
        responses: impl IntoIterator<Item = Result<GhCommandOutput, GhCommandError>>,
    ) -> (Self, Rc<RefCell<SharedGhRunnerState>>) {
        let state = Rc::new(RefCell::new(SharedGhRunnerState {
            commands: Vec::new(),
            responses: responses.into_iter().collect(),
        }));
        (
            Self {
                state: Rc::clone(&state),
            },
            state,
        )
    }
}

impl GhCommandRunner for SharedGhRunner {
    fn run(&mut self, command: GhCommand) -> Result<GhCommandOutput, GhCommandError> {
        let mut state = self.state.borrow_mut();
        state.commands.push(command);
        state.responses.pop_front().unwrap_or_else(|| {
            Err(GhCommandError::WaitFailed(
                "missing fake gh response".into(),
            ))
        })
    }
}

#[derive(Default)]
pub(super) struct FakeServiceRunner;

impl ServiceCommandRunner for FakeServiceRunner {
    fn run(&mut self, _command: &ServiceCommand) -> anyhow::Result<()> {
        Ok(())
    }
}

pub(super) fn service_context_fixture() -> (TempDir, ServiceInstallContext) {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let binary_path = temp.path().join("bin/mesh-llm");
    std::fs::create_dir_all(binary_path.parent().expect("binary parent should exist"))
        .expect("binary dir should exist");
    std::fs::write(&binary_path, "binary").expect("binary should write");
    let context = ServiceInstallContext {
        platform: SetupPlatform::Linux,
        home_dir: temp.path().join("home"),
        config_root: temp.path().join("config"),
        binary_path,
        user_id: String::new(),
        start_service: false,
    };
    (temp, context)
}

pub(super) fn success_output(stdout: &str) -> Result<GhCommandOutput, GhCommandError> {
    Ok(GhCommandOutput {
        success: true,
        stdout: stdout.to_string(),
        stderr: String::new(),
    })
}
