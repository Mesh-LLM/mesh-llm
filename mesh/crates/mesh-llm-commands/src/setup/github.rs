use super::github_runner::{GhCommand, GhCommandError, GhCommandOutput, GhCommandRunner};
use super::{
    SetupConfirmPrompt, SetupGitHubStarPlan, SetupGitHubStarSkipReason, SetupPlan,
    SetupPromptDefault, SetupPromptKind, SetupPrompter,
};

const GITHUB_STAR_PROMPT: &str = "Star Mesh-LLM/mesh-llm on GitHub?";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SetupGitHubOutcome {
    NotEvaluated,
    AutomaticYes,
    HiddenPrompt,
    CliUnavailable,
    NotAuthenticated,
    AlreadyStarred,
    DeclinedAtPrompt,
    Starred,
    EligibilityCheckFailed(String),
    StarRequestFailed(String),
}

pub(crate) fn execute_github_star_plan<R: GhCommandRunner + ?Sized>(
    plan: SetupGitHubStarPlan,
    runner: &mut R,
    prompter: &mut dyn SetupPrompter,
) -> SetupGitHubOutcome {
    match plan {
        SetupGitHubStarPlan::Skip(skip_reason) => match skip_reason {
            SetupGitHubStarSkipReason::AutomaticYes => SetupGitHubOutcome::AutomaticYes,
            SetupGitHubStarSkipReason::HiddenPrompt => SetupGitHubOutcome::HiddenPrompt,
        },
        SetupGitHubStarPlan::PromptIfEligible => {
            run_github_star_prompt_if_eligible(runner, prompter)
        }
    }
}

pub(crate) fn github_summary(plan: &SetupPlan, outcome: &SetupGitHubOutcome) -> String {
    match plan.github_star {
        SetupGitHubStarPlan::PromptIfEligible => match outcome {
            SetupGitHubOutcome::NotEvaluated => "not recorded".to_string(),
            SetupGitHubOutcome::CliUnavailable => "skipped; GitHub CLI is not on PATH".to_string(),
            SetupGitHubOutcome::NotAuthenticated => {
                "skipped; GitHub CLI is not authenticated for github.com".to_string()
            }
            SetupGitHubOutcome::AlreadyStarred => {
                "already starred via the authenticated GitHub CLI account".to_string()
            }
            SetupGitHubOutcome::DeclinedAtPrompt => {
                "skipped at the visible GitHub star prompt".to_string()
            }
            SetupGitHubOutcome::Starred => {
                "starred Mesh-LLM/mesh-llm with the authenticated GitHub CLI account".to_string()
            }
            SetupGitHubOutcome::EligibilityCheckFailed(error) => {
                format!("skipped; GitHub eligibility check failed: {error}")
            }
            SetupGitHubOutcome::StarRequestFailed(error) => {
                format!("not starred; GitHub star request failed: {error}")
            }
            SetupGitHubOutcome::AutomaticYes | SetupGitHubOutcome::HiddenPrompt => {
                "not recorded".to_string()
            }
        },
        SetupGitHubStarPlan::Skip(_) => match outcome {
            SetupGitHubOutcome::AutomaticYes => {
                "skipped because --yes suppresses prompts".to_string()
            }
            SetupGitHubOutcome::HiddenPrompt => "skipped because prompts were hidden".to_string(),
            SetupGitHubOutcome::NotEvaluated
            | SetupGitHubOutcome::CliUnavailable
            | SetupGitHubOutcome::NotAuthenticated
            | SetupGitHubOutcome::AlreadyStarred
            | SetupGitHubOutcome::DeclinedAtPrompt
            | SetupGitHubOutcome::Starred
            | SetupGitHubOutcome::EligibilityCheckFailed(_)
            | SetupGitHubOutcome::StarRequestFailed(_) => "not requested".to_string(),
        },
    }
}

fn run_github_star_prompt_if_eligible<R: GhCommandRunner + ?Sized>(
    runner: &mut R,
    prompter: &mut dyn SetupPrompter,
) -> SetupGitHubOutcome {
    match runner.run(GhCommand::CheckAvailability) {
        Ok(output) if output.success => {}
        Ok(output) => {
            return SetupGitHubOutcome::EligibilityCheckFailed(command_failure(
                GhCommand::CheckAvailability,
                &output,
            ));
        }
        Err(GhCommandError::NotOnPath) => return SetupGitHubOutcome::CliUnavailable,
        Err(error) => return SetupGitHubOutcome::EligibilityCheckFailed(error.to_string()),
    }

    match runner.run(GhCommand::CheckAuthentication) {
        Ok(output) if output.success => {}
        Ok(_) => return SetupGitHubOutcome::NotAuthenticated,
        Err(error) => return SetupGitHubOutcome::EligibilityCheckFailed(error.to_string()),
    }

    let viewer_has_starred = match runner.run(GhCommand::CheckViewerHasStarred) {
        Ok(output) if output.success => match output.stdout.trim() {
            "true" => true,
            "false" => false,
            other => {
                return SetupGitHubOutcome::EligibilityCheckFailed(format!(
                    "unexpected output from `{}`: {other}",
                    GhCommand::CheckViewerHasStarred.display_name()
                ));
            }
        },
        Ok(output) => {
            return SetupGitHubOutcome::EligibilityCheckFailed(command_failure(
                GhCommand::CheckViewerHasStarred,
                &output,
            ));
        }
        Err(error) => return SetupGitHubOutcome::EligibilityCheckFailed(error.to_string()),
    };
    if viewer_has_starred {
        return SetupGitHubOutcome::AlreadyStarred;
    }

    let prompt = SetupConfirmPrompt {
        kind: SetupPromptKind::GitHubStar,
        message: GITHUB_STAR_PROMPT,
        default: SetupPromptDefault::Yes,
    };
    let accepted = prompt.default.resolve(prompter.confirm(prompt));
    if !accepted {
        return SetupGitHubOutcome::DeclinedAtPrompt;
    }

    match runner.run(GhCommand::StarRepository) {
        Ok(output) if output.success => SetupGitHubOutcome::Starred,
        Ok(output) => SetupGitHubOutcome::StarRequestFailed(command_failure(
            GhCommand::StarRepository,
            &output,
        )),
        Err(error) => SetupGitHubOutcome::StarRequestFailed(error.to_string()),
    }
}

fn command_failure(command: GhCommand, output: &GhCommandOutput) -> String {
    let detail = first_non_empty_line(&output.stderr)
        .or_else(|| first_non_empty_line(&output.stdout))
        .unwrap_or("no output");
    format!("`{}` reported: {detail}", command.display_name())
}

fn first_non_empty_line(value: &str) -> Option<&str> {
    value.lines().map(str::trim).find(|line| !line.is_empty())
}
