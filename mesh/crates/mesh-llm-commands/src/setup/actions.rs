use super::{SetupGitHubStarPlan, SetupPrompter, SetupStep};
use std::future::Future;

pub trait SetupActions {
    type Error;
    type GitHubStarFuture<'a>: Future<Output = Result<(), Self::Error>> + 'a
    where
        Self: 'a;
    type StepFuture<'a>: Future<Output = Result<(), Self::Error>> + 'a
    where
        Self: 'a;

    fn run_step(&mut self, step: SetupStep) -> Self::StepFuture<'_>;

    fn handle_github_star<'a>(
        &'a mut self,
        plan: SetupGitHubStarPlan,
        prompter: &'a mut dyn SetupPrompter,
    ) -> Self::GitHubStarFuture<'a>;
}
