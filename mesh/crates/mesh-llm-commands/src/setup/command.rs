use super::prompt::confirm_yes_no;
use super::service_paths::ServiceInstallContext;
use super::service_runner::{CliServiceCommandRunner, ServiceCommandRunner};
use super::summary::{
    print_runtime_install_result, print_service_install_result, print_setup_summary,
};
use super::{
    SetupActions, SetupConfirmPrompt, SetupEnvironment, SetupGitHubStarPlan, SetupOptions,
    SetupPlan, SetupPluginPlan, SetupPrompter, SetupStep,
    github::{SetupGitHubOutcome, execute_github_star_plan},
    github_runner::{GhCommandRunner, ProcessGhCommandRunner},
    plan_setup,
    service::{ServiceInstallReport, install_service},
};
use crate::runtime_native::{
    NativeRuntimeConfigSelection, SetupNativeRuntimeOptions, SetupNativeRuntimeOutcome,
    SetupNativeRuntimePruneResult, install_and_prune_native_runtime_for_setup,
};
use anyhow::{Result, anyhow};
use std::future::Future;
use std::io::Write;
use std::pin::Pin;

#[derive(Clone, Debug)]
pub struct SetupCommandArgs<'a> {
    pub options: SetupOptions,
    pub environment: SetupEnvironment,
    pub configured: NativeRuntimeConfigSelection<'a>,
}

pub async fn run_setup<P, A>(
    options: SetupOptions,
    environment: SetupEnvironment,
    prompter: &mut P,
    actions: &mut A,
) -> Result<SetupPlan>
where
    P: SetupPrompter,
    A: SetupActions,
    A::Error: Into<anyhow::Error>,
{
    let plan = plan_setup(&options, &environment, prompter).map_err(anyhow::Error::new)?;
    for step in plan.core_steps.iter().copied() {
        actions.run_step(step).await.map_err(Into::into)?;
    }
    // The host is already replaced: a failed check is reported, never fatal.
    if let Err(error) = actions.check_installed_plugins().await {
        let error: anyhow::Error = error.into();
        let mut err = mesh_llm_events::console_err();
        let _ = writeln!(err, "⚠️  Installed plugin check failed: {error:#}");
    }
    actions
        .handle_plugin_install(plan.plugins.clone(), prompter)
        .await
        .map_err(Into::into)?;
    actions
        .handle_github_star(plan.github_star, prompter)
        .await
        .map_err(Into::into)?;
    Ok(plan)
}

pub async fn run_setup_command(args: SetupCommandArgs<'_>) -> Result<()> {
    let mut prompter = CliSetupPrompter;
    let mut actions = CliSetupActions::new(args.environment, args.configured, args.options.verbose);
    let plan = run_setup(
        args.options.clone(),
        args.environment,
        &mut prompter,
        &mut actions,
    )
    .await?;
    print_setup_summary(&plan, &actions, args.options.verbose);
    Ok(())
}

struct CliSetupPrompter;

impl SetupPrompter for CliSetupPrompter {
    fn confirm(&mut self, prompt: SetupConfirmPrompt) -> Option<bool> {
        // The prompt's declared default reaches the terminal: the plugin-set
        // offer accepts on an empty reply, exactly as its plan promises.
        confirm_yes_no(prompt.message, prompt.default)
    }
}

pub(crate) struct CliSetupActions<'a> {
    environment: SetupEnvironment,
    configured: NativeRuntimeConfigSelection<'a>,
    pub(crate) runtime_outcome: Option<SetupNativeRuntimeOutcome>,
    service_context: Option<ServiceInstallContext>,
    service_runner: Box<dyn ServiceCommandRunner>,
    github_runner: Box<dyn GhCommandRunner>,
    pub(crate) service_outcome: SetupServiceOutcome,
    pub(crate) plugin_outcome: SetupPluginOutcome,
    /// The installed-plugin check refused: some plugin is not known to run on
    /// the host now installed.
    pub(crate) plugin_check_refused: bool,
    /// Whether `check_installed_plugins` consults the machine's plugin store.
    /// Tests that exercise other setup behavior turn it off so they never read
    /// the developer's real store or reach the plugin catalog.
    run_plugin_check: bool,
    pub(crate) github_outcome: SetupGitHubOutcome,
    verbose: bool,
}

impl<'a> CliSetupActions<'a> {
    pub(crate) fn new(
        environment: SetupEnvironment,
        configured: NativeRuntimeConfigSelection<'a>,
        verbose: bool,
    ) -> Self {
        Self::with_support(
            environment,
            configured,
            None,
            Box::new(CliServiceCommandRunner),
            Box::new(ProcessGhCommandRunner::default()),
            verbose,
        )
    }

    fn with_support(
        environment: SetupEnvironment,
        configured: NativeRuntimeConfigSelection<'a>,
        service_context: Option<ServiceInstallContext>,
        service_runner: Box<dyn ServiceCommandRunner>,
        github_runner: Box<dyn GhCommandRunner>,
        verbose: bool,
    ) -> Self {
        Self {
            environment,
            configured,
            runtime_outcome: None,
            service_context,
            service_runner,
            github_runner,
            service_outcome: SetupServiceOutcome::NotRequested,
            plugin_outcome: SetupPluginOutcome::NotRequested,
            plugin_check_refused: false,
            run_plugin_check: true,
            github_outcome: SetupGitHubOutcome::NotEvaluated,
            verbose,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_service_support(
        environment: SetupEnvironment,
        configured: NativeRuntimeConfigSelection<'a>,
        service_context: ServiceInstallContext,
        service_runner: Box<dyn ServiceCommandRunner>,
        github_runner: Box<dyn GhCommandRunner>,
    ) -> Self {
        let mut actions = Self::with_support(
            environment,
            configured,
            Some(service_context),
            service_runner,
            github_runner,
            false,
        );
        actions.run_plugin_check = false;
        actions
    }

    async fn install_runtime(&mut self) -> Result<()> {
        let outcome = install_and_prune_native_runtime_for_setup(SetupNativeRuntimeOptions {
            skip_runtime: false,
            requested_runtime: None,
            manifest_path: None,
            bundle_dirs: &[],
            cache_dir: None,
            configured: self.configured,
            progress: None,
        })
        .await?;
        if self.verbose {
            print_runtime_install_result(&outcome);
        }
        self.runtime_outcome = Some(outcome);
        Ok(())
    }

    fn report_runtime_prune(&self) -> Result<()> {
        let outcome = self
            .runtime_outcome
            .as_ref()
            .ok_or_else(|| anyhow!("setup runtime prune step ran before runtime install"))?;
        let mut err = mesh_llm_events::console_err();
        match &outcome.prune {
            SetupNativeRuntimePruneResult::Skipped => {}
            SetupNativeRuntimePruneResult::Pruned(plan) => {
                if self.verbose {
                    if plan.remove_dirs.is_empty() {
                        writeln!(err, "Native runtime cache is already clean")?;
                    } else {
                        writeln!(
                            err,
                            "Pruned {} inactive native runtime cache entr{}",
                            plan.remove_dirs.len(),
                            if plan.remove_dirs.len() == 1 {
                                "y"
                            } else {
                                "ies"
                            }
                        )?;
                    }
                }
            }
            SetupNativeRuntimePruneResult::Warning(warning) => {
                writeln!(
                    err,
                    "warning: native runtime installed, but cache pruning failed: {warning}"
                )?;
            }
        }
        Ok(())
    }

    fn install_service(&mut self) -> Result<()> {
        let context = match self.service_context.clone() {
            Some(context) => context,
            None => ServiceInstallContext::detect(self.environment.platform, true)?,
        };
        let report = install_service(&context, self.service_runner.as_mut())?;
        print_service_install_result(&report, self.verbose);
        self.service_outcome = SetupServiceOutcome::Installed(report);
        Ok(())
    }

    fn print_service_guidance(&self) {
        let mut err = mesh_llm_events::console_err();
        let _ = writeln!(
            err,
            "Service not installed. Run `mesh-llm setup --service` to enable it later."
        );
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SetupServiceOutcome {
    NotRequested,
    Installed(ServiceInstallReport),
    PrintedGuidance,
}

/// What the optional plugin-set phase did. A failed optional install is
/// `Incomplete`, never a setup failure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum SetupPluginOutcome {
    #[default]
    NotRequested,
    Skipped,
    Installed(usize),
    Incomplete,
}

impl SetupActions for CliSetupActions<'_> {
    type Error = anyhow::Error;
    type PluginInstallFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), Self::Error>> + 'a>>
    where
        Self: 'a;
    type GitHubStarFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), Self::Error>> + 'a>>
    where
        Self: 'a;
    type StepFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), Self::Error>> + 'a>>
    where
        Self: 'a;

    fn run_step(&mut self, step: SetupStep) -> Self::StepFuture<'_> {
        Box::pin(async move {
            match step {
                SetupStep::InstallRuntime => self.install_runtime().await,
                SetupStep::PruneInactiveRuntimes => self.report_runtime_prune(),
                SetupStep::InstallService => self.install_service(),
                SetupStep::PrintServiceGuidance => {
                    self.service_outcome = SetupServiceOutcome::PrintedGuidance;
                    self.print_service_guidance();
                    Ok(())
                }
            }
        })
    }

    fn check_installed_plugins(&mut self) -> Self::PluginInstallFuture<'_> {
        Box::pin(async move {
            // The host is already replaced, so a failure is reported with the
            // retry command rather than failing setup.
            if !self.run_plugin_check {
                return Ok(());
            }
            let host_version = mesh_llm_build_info::RELEASE_VERSION;
            if let Err(error) = crate::update::check_installed_plugins_for_host(host_version).await
            {
                self.plugin_check_refused = true;
                let mut err = mesh_llm_events::console_err();
                let _ = writeln!(
                    err,
                    "⚠️  Installed plugins are not all compatible with MeshLLM {host_version}: \
                     {error:#}. Setup continued; run `mesh-llm update` to retry."
                );
            }
            Ok(())
        })
    }

    fn handle_plugin_install<'a>(
        &'a mut self,
        plan: SetupPluginPlan,
        prompter: &'a mut dyn super::SetupPrompter,
    ) -> Self::PluginInstallFuture<'a> {
        Box::pin(async move {
            let optional = match &plan {
                SetupPluginPlan::Install(name) => {
                    crate::plugin_sets::OptionalPluginPlan::InstallSet(name.clone())
                }
                SetupPluginPlan::Offer => crate::plugin_sets::OptionalPluginPlan::Offer(
                    crate::plugin_sets::RECOMMENDED_SET.to_string(),
                ),
                SetupPluginPlan::InstallDefault => {
                    crate::plugin_sets::OptionalPluginPlan::InstallDefault(
                        crate::plugin_sets::RECOMMENDED_SET.to_string(),
                    )
                }
                SetupPluginPlan::Skip => crate::plugin_sets::OptionalPluginPlan::Skip,
            };
            let result = if matches!(plan, SetupPluginPlan::Skip) {
                // A skip prints its line; nothing here can fail.
                crate::plugin_sets::run_optional_plugin_phase(
                    &optional,
                    mesh_llm_build_info::RELEASE_VERSION,
                    &mut |_| false,
                )
                .await
                .unwrap_or(crate::plugin_sets::OptionalPluginResult::Skipped)
            } else {
                // The only reader of the terminal: the offer prints its lines
                // first, then the prompter answers with a Yes default (install
                // is preselected); an explicit `n` or an EOF declines.
                let mut prompt = |_set: &str| {
                    let reply = prompter.confirm(super::SetupConfirmPrompt {
                        kind: super::SetupPromptKind::InstallPlugins,
                        message: PLUGIN_SET_PROMPT,
                        default: super::SetupPromptDefault::Yes,
                    });
                    super::SetupPromptDefault::Yes.resolve(reply)
                };
                // The core steps already succeeded; optional selection and
                // download failures are reported as incomplete installation.
                crate::plugin_sets::run_optional_plugin_phase(
                    &optional,
                    mesh_llm_build_info::RELEASE_VERSION,
                    &mut prompt,
                )
                .await?
            };
            self.plugin_outcome = match result {
                crate::plugin_sets::OptionalPluginResult::Skipped => SetupPluginOutcome::Skipped,
                crate::plugin_sets::OptionalPluginResult::Installed(count) => {
                    SetupPluginOutcome::Installed(count)
                }
                crate::plugin_sets::OptionalPluginResult::Incomplete(_) => {
                    SetupPluginOutcome::Incomplete
                }
            };
            Ok(())
        })
    }

    fn handle_github_star<'a>(
        &'a mut self,
        plan: SetupGitHubStarPlan,
        prompter: &'a mut dyn super::SetupPrompter,
    ) -> Self::GitHubStarFuture<'a> {
        Box::pin(async move {
            self.github_outcome =
                execute_github_star_plan(plan, &mut *self.github_runner, prompter);
            Ok(())
        })
    }
}

/// The prompt message for the plugin-set offer; the offer lines above it
/// name the set.
const PLUGIN_SET_PROMPT: &str = "Install this plugin set?";
