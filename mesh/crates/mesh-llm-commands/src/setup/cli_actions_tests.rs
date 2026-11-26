use super::SetupPluginStore;
use super::command::{CliSetupActions, SetupServiceOutcome};
use super::github::SetupGitHubOutcome;
use super::github_runner::{GhCommand, GhCommandError};
use super::test_support::{
    FakePrompter, FakeServiceRunner, SharedGhRunner, service_context_fixture, success_output,
};
use super::{SetupActions, SetupEnvironment, SetupOptions, SetupPlatform, SetupStep, run_setup};
use crate::runtime_native::{
    NativeRuntimeConfigSelection, SetupNativeRuntimeOutcome, SetupNativeRuntimePruneResult,
    SetupNativeRuntimeStatus,
};

#[tokio::test]
async fn cli_setup_actions_treat_runtime_prune_warning_as_non_fatal() {
    let (_temp, context) = service_context_fixture();
    let mut actions = CliSetupActions::with_service_support(
        SetupEnvironment {
            platform: SetupPlatform::Linux,
            interactive: false,
        },
        NativeRuntimeConfigSelection::default(),
        context,
        Box::new(FakeServiceRunner),
        Box::new(SharedGhRunner::new([]).0),
    );
    actions.runtime_outcome = Some(SetupNativeRuntimeOutcome {
        status: SetupNativeRuntimeStatus::Skipped,
        prune: SetupNativeRuntimePruneResult::Warning("prune failed".to_string()),
    });

    actions
        .run_step(SetupStep::PruneInactiveRuntimes)
        .await
        .expect("prune warnings should stay non-fatal");

    assert_eq!(actions.service_outcome, SetupServiceOutcome::NotRequested);
}

#[tokio::test]
async fn run_setup_with_cli_actions_records_nonfatal_github_failure() {
    let (_temp, context) = service_context_fixture();
    let (runner, state) = SharedGhRunner::new([
        success_output("gh version 2.85.0"),
        success_output("authenticated"),
        success_output("false"),
        Err(GhCommandError::TimedOut("gh api")),
    ]);
    let mut actions = CliSetupActions::with_service_support(
        SetupEnvironment {
            platform: SetupPlatform::Linux,
            interactive: true,
        },
        NativeRuntimeConfigSelection::default(),
        context,
        Box::new(FakeServiceRunner),
        Box::new(runner),
    );
    let mut prompter = FakePrompter::with_replies([None]);
    let options = SetupOptions {
        skip_runtime: true,
        skip_plugin_install: true,
        no_service: true,
        ..SetupOptions::default()
    };

    let plan = run_setup(
        options,
        SetupEnvironment {
            platform: SetupPlatform::Linux,
            interactive: true,
        },
        &mut prompter,
        &mut actions,
    )
    .await
    .expect("github star failures should remain non-fatal");

    assert!(plan.core_steps.is_empty());
    assert_eq!(prompter.prompts.len(), 1);
    assert_eq!(
        actions.github_outcome,
        SetupGitHubOutcome::StarRequestFailed("timed out running `gh api`".to_string())
    );
    assert_eq!(
        state.borrow().commands,
        vec![
            GhCommand::CheckAvailability,
            GhCommand::CheckAuthentication,
            GhCommand::CheckViewerHasStarred,
            GhCommand::StarRepository,
        ]
    );
}

/// The plugin step is optional: with the catalog unreachable it records why
/// and setup carries on, with the later steps still run.
#[tokio::test]
async fn a_failing_plugin_install_never_fails_setup() {
    let (_temp, context) = service_context_fixture();
    let mut actions = CliSetupActions::with_service_support(
        SetupEnvironment {
            platform: SetupPlatform::Linux,
            interactive: false,
        },
        NativeRuntimeConfigSelection::default(),
        context,
        Box::new(FakeServiceRunner),
        Box::new(SharedGhRunner::new([]).0),
    );
    let mut prompter = FakePrompter::default();
    let options = SetupOptions {
        skip_runtime: true,
        no_service: true,
        no_interactive: true,
        ..SetupOptions::default()
    };

    let plan = run_setup(
        options,
        SetupEnvironment {
            platform: SetupPlatform::Linux,
            interactive: false,
        },
        &mut prompter,
        &mut actions,
    )
    .await
    .expect("an unreachable plugin catalog must not fail setup");

    assert_eq!(
        plan.core_steps,
        vec![SetupStep::InstallPlugins(SetupPluginStore::Invoking)]
    );
    assert!(matches!(
        actions.plugins_outcome,
        crate::plugin_sets::SetupPluginsOutcome::Unavailable(_)
    ));
    assert!(prompter.prompts.is_empty());
}

/// `plugins install` uses the invoking shell's store, so when setup installed
/// into the service's store the retry command names that store.
#[test]
fn the_retry_hint_names_the_store_setup_installed_into() {
    use super::command::RetryStore;
    use std::path::PathBuf;

    assert!(
        RetryStore::Invoking
            .hint()
            .starts_with("Retry with `mesh-llm plugins install --set recommended`")
    );
    assert!(
        RetryStore::Service(PathBuf::from("/srv/mesh/plugins"))
            .hint()
            .starts_with(
                "Retry with `MESH_LLM_PLUGIN_DIR=/srv/mesh/plugins mesh-llm plugins install --set recommended`"
            )
    );
    assert!(
        RetryStore::Service(PathBuf::from("/Users/a b/it's"))
            .hint()
            .contains("`MESH_LLM_PLUGIN_DIR='/Users/a b/it'\\''s' mesh-llm")
    );
    assert!(
        RetryStore::UnknownService
            .hint()
            .contains("MESH_LLM_PLUGIN_DIR=<that store>")
    );
}
