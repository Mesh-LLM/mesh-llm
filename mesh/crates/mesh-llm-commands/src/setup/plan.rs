#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetupPlan {
    pub runtime: SetupRuntimePlan,
    pub plugins: SetupPluginsPlan,
    pub service: SetupServicePlan,
    pub github_star: SetupGitHubStarPlan,
    pub core_steps: Vec<SetupStep>,
}

impl SetupPlan {
    pub fn new(
        runtime: SetupRuntimePlan,
        plugins: SetupPluginsPlan,
        service: SetupServicePlan,
        github_star: SetupGitHubStarPlan,
    ) -> Self {
        let core_steps = build_core_steps(runtime, plugins, service);
        Self {
            runtime,
            plugins,
            service,
            github_star,
            core_steps,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupRuntimePlan {
    InstallAndPrune,
    Skip,
}

/// Whether setup installs the catalog's default plugin set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupPluginsPlan {
    InstallDefaultSet,
    Skip,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupServicePlan {
    Install,
    Skip,
    PrintGuidance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupGitHubStarPlan {
    PromptIfEligible,
    Skip(SetupGitHubStarSkipReason),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupGitHubStarSkipReason {
    AutomaticYes,
    HiddenPrompt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupStep {
    InstallRuntime,
    PruneInactiveRuntimes,
    /// Non-fatal: the action reports a failure and setup carries on.
    InstallPlugins,
    InstallService,
    PrintServiceGuidance,
}

fn build_core_steps(
    runtime: SetupRuntimePlan,
    plugins: SetupPluginsPlan,
    service: SetupServicePlan,
) -> Vec<SetupStep> {
    let mut steps = Vec::new();
    if matches!(runtime, SetupRuntimePlan::InstallAndPrune) {
        steps.push(SetupStep::InstallRuntime);
        steps.push(SetupStep::PruneInactiveRuntimes);
    }
    // Before the service so a node it starts already finds the plugins.
    if matches!(plugins, SetupPluginsPlan::InstallDefaultSet) {
        steps.push(SetupStep::InstallPlugins);
    }
    match service {
        SetupServicePlan::Install => steps.push(SetupStep::InstallService),
        SetupServicePlan::PrintGuidance => steps.push(SetupStep::PrintServiceGuidance),
        SetupServicePlan::Skip => {}
    }
    steps
}
