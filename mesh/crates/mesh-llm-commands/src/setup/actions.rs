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

    /// The plugin compatibility check and repairs every host upgrade owes
    /// installed plugins. Setup is where the bootstrap scripts land after
    /// replacing the host, so it runs here against the host now installed.
    /// An error here does not fail setup: the host is already replaced, so
    /// `run_setup` reports it and continues.
    fn check_installed_plugins(&mut self) -> Self::PluginInstallFuture<'_>;

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
