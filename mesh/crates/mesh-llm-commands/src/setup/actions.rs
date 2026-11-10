use super::{SetupGitHubStarPlan, SetupPluginPlan, SetupPrompter, SetupStep};
use std::future::Future;

pub trait SetupActions {
    type Error;
    type GitHubStarFuture<'a>: Future<Output = Result<(), Self::Error>> + 'a
    where
        Self: 'a;
    type PluginInstallFuture<'a>: Future<Output = Result<(), Self::Error>> + 'a
    where
        Self: 'a;
    type StepFuture<'a>: Future<Output = Result<(), Self::Error>> + 'a
    where
        Self: 'a;

    fn run_step(&mut self, step: SetupStep) -> Self::StepFuture<'_>;

    /// The optional plugin-set phase, after core setup succeeded. A failure
    /// here must not fail setup.
    fn handle_plugin_install<'a>(
        &'a mut self,
        plan: SetupPluginPlan,
        prompter: &'a mut dyn SetupPrompter,
    ) -> Self::PluginInstallFuture<'a>;

    fn handle_github_star<'a>(
        &'a mut self,
        plan: SetupGitHubStarPlan,
        prompter: &'a mut dyn SetupPrompter,
    ) -> Self::GitHubStarFuture<'a>;
}
