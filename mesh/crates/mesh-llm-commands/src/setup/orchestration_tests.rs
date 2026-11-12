use super::test_support::{FakePrompter, RecordingActions};
use super::{SetupEnvironment, SetupOptions, SetupPlatform, SetupStep, run_setup};

#[tokio::test]
async fn run_setup_no_interactive_uses_guidance_step_and_hidden_star_without_prompts() {
    let options = SetupOptions {
        no_interactive: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: false,
    };
    let mut prompter = FakePrompter::default();
    let mut actions = RecordingActions::default();

    let plan = run_setup(options, environment, &mut prompter, &mut actions)
        .await
        .expect("setup should execute without prompts");

    assert_eq!(
        plan.core_steps,
        vec![
            SetupStep::InstallRuntime,
            SetupStep::PruneInactiveRuntimes,
            SetupStep::PrintServiceGuidance,
        ]
    );
    assert_eq!(actions.steps, plan.core_steps);
    assert_eq!(actions.github, vec![plan.github_star]);
    assert!(prompter.prompts.is_empty());
}

#[tokio::test]
async fn run_setup_skip_runtime_and_no_service_runs_no_hidden_core_side_effects() {
    let options = SetupOptions {
        skip_runtime: true,
        no_service: true,
        yes: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: true,
    };
    let mut prompter = FakePrompter::default();
    let mut actions = RecordingActions::default();

    let plan = run_setup(options, environment, &mut prompter, &mut actions)
        .await
        .expect("setup should execute without hidden work");

    assert!(plan.core_steps.is_empty());
    assert!(actions.steps.is_empty());
    assert_eq!(actions.github, vec![plan.github_star]);
    assert!(prompter.prompts.is_empty());
}

#[tokio::test]
async fn run_setup_service_failure_is_core_fatal_and_skips_github() {
    let options = SetupOptions {
        skip_runtime: true,
        service: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: true,
    };
    let mut prompter = FakePrompter::default();
    let mut actions = RecordingActions::failing_on(SetupStep::InstallService);

    let error = run_setup(options, environment, &mut prompter, &mut actions)
        .await
        .expect_err("service installation failure should stop setup");

    assert!(
        error
            .to_string()
            .contains("simulated step failure for InstallService")
    );
    assert_eq!(actions.steps, vec![SetupStep::InstallService]);
    assert!(actions.github.is_empty());
}

// Bootstrap scripts replace the host and then run `mesh-llm setup`, so setup
// owes installed plugins the same check and repairs `mesh-llm update` runs.
// It must run on every setup, after core steps and before the optional
// plugin phase, whatever the plugin flags say.
#[tokio::test]
async fn run_setup_checks_installed_plugins_after_core_steps_and_before_the_optional_phase() {
    for skip_plugin_install in [false, true] {
        let options = SetupOptions {
            no_interactive: true,
            skip_plugin_install,
            ..SetupOptions::default()
        };
        let environment = SetupEnvironment {
            platform: SetupPlatform::Linux,
            interactive: false,
        };
        let mut prompter = FakePrompter::default();
        let mut actions = RecordingActions::default();

        run_setup(options, environment, &mut prompter, &mut actions)
            .await
            .expect("setup should succeed");

        let checks = actions
            .order
            .iter()
            .filter(|action| **action == "plugin-check")
            .count();
        assert_eq!(checks, 1, "skip_plugin_install={skip_plugin_install}");
        let check = actions.order.iter().position(|a| *a == "plugin-check");
        let install = actions.order.iter().position(|a| *a == "plugin-install");
        let last_step = actions.order.iter().rposition(|a| *a == "step");
        assert!(last_step < check && check < install, "{:?}", actions.order);
    }
}

#[tokio::test]
async fn run_setup_does_not_check_plugins_when_core_setup_fails() {
    let options = SetupOptions {
        no_interactive: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: false,
    };
    let mut prompter = FakePrompter::default();
    let mut actions = RecordingActions::failing_on(SetupStep::InstallRuntime);

    run_setup(options, environment, &mut prompter, &mut actions)
        .await
        .expect_err("core failure is fatal");

    assert!(!actions.order.contains(&"plugin-check"));
}
