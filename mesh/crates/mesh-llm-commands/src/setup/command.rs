use super::prompt::confirm_yes_no;
use super::service_paths::ServiceInstallContext;
use super::service_runner::{CliServiceCommandRunner, ServiceCommandRunner};
use super::summary::{
    print_runtime_install_result, print_service_install_result, print_setup_summary,
};
use super::{
    SetupActions, SetupConfirmPrompt, SetupEnvironment, SetupGitHubStarPlan, SetupOptions,
    SetupPlan, SetupPrompter, SetupStep,
    github::{SetupGitHubOutcome, execute_github_star_plan},
    github_runner::{GhCommandRunner, ProcessGhCommandRunner},
    plan_setup,
    service::{ServiceInstallReport, install_service},
};
use crate::plugin_sets::{DEFAULT_SET, SetupPluginsOutcome, install_default_set_for_setup};
use crate::runtime_native::{
    NativeRuntimeConfigSelection, SetupNativeRuntimeOptions, SetupNativeRuntimeOutcome,
    SetupNativeRuntimePruneResult, install_and_prune_native_runtime_for_setup,
};
use anyhow::{Result, anyhow};
use mesh_llm_plugin_manager::PluginInstallOptions;
use std::future::Future;
use std::io::Write;
use std::pin::Pin;

#[derive(Clone, Copy, Debug)]
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
    let plan = plan_setup(options, environment, prompter).map_err(anyhow::Error::new)?;
    for step in plan.core_steps.iter().copied() {
        actions.run_step(step).await.map_err(Into::into)?;
    }
    actions
        .handle_github_star(plan.github_star, prompter)
        .await
        .map_err(Into::into)?;
    Ok(plan)
}

pub async fn run_setup_command(args: SetupCommandArgs<'_>) -> Result<()> {
    let mut prompter = CliSetupPrompter;
    let mut actions = CliSetupActions::new(args.environment, args.configured, args.options.verbose);
    let plan = run_setup(args.options, args.environment, &mut prompter, &mut actions).await?;
    print_setup_summary(&plan, &actions, args.options.verbose);
    Ok(())
}

struct CliSetupPrompter;

impl SetupPrompter for CliSetupPrompter {
    fn confirm(&mut self, prompt: SetupConfirmPrompt) -> Option<bool> {
        confirm_yes_no(prompt.message)
    }
}

pub(crate) struct CliSetupActions<'a> {
    environment: SetupEnvironment,
    configured: NativeRuntimeConfigSelection<'a>,
    pub(crate) runtime_outcome: Option<SetupNativeRuntimeOutcome>,
    pub(crate) plugins_outcome: SetupPluginsOutcome,
    service_context: Option<ServiceInstallContext>,
    service_runner: Box<dyn ServiceCommandRunner>,
    github_runner: Box<dyn GhCommandRunner>,
    /// Where the plugin step reads the catalog; `None` reads the environment's.
    plugin_options: Option<PluginInstallOptions>,
    pub(crate) service_outcome: SetupServiceOutcome,
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
            plugins_outcome: SetupPluginsOutcome::NotRequested,
            service_context,
            service_runner,
            github_runner,
            plugin_options: None,
            service_outcome: SetupServiceOutcome::NotRequested,
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
        // A refused loopback port: the plugin step fails at once and tests
        // never reach the network or a real plugin store.
        actions.plugin_options = Some(PluginInstallOptions {
            store_root: std::env::temp_dir().join("mesh-llm-setup-test-unused-plugins"),
            install_root: std::env::temp_dir().join("mesh-llm-setup-test-unused-plugins/installed"),
            catalog_url: "http://127.0.0.1:9/plugins.jsonl".to_string(),
            target: mesh_llm_plugin_manager::PluginTarget::current()
                .expect("tests run on a supported plugin target"),
        });
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

    /// Install the default plugin set. Never fails setup: what went wrong is
    /// printed with the command that retries it, and kept for the summary.
    async fn install_plugins(&mut self) -> Result<()> {
        let options = match self.plugin_options.clone() {
            Some(options) => Ok(options),
            None => PluginInstallOptions::from_env(),
        };
        let outcome = match options {
            Ok(options) => {
                let mut progress = crate::plugin::CliPluginProgress::default();
                let outcome = install_default_set_for_setup(&options, &mut progress).await;
                progress.finish();
                outcome
            }
            Err(error) => SetupPluginsOutcome::Unavailable(format!("{error:#}")),
        };
        print_plugins_warnings(&outcome);
        self.plugins_outcome = outcome;
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

fn print_plugins_warnings(outcome: &SetupPluginsOutcome) {
    let mut err = mesh_llm_events::console_err();
    let retry = format!(
        "Retry with `mesh-llm plugins install --set {DEFAULT_SET}`, or pass --skip-plugin-install to setup to opt out."
    );
    match outcome {
        SetupPluginsOutcome::Unavailable(reason) => {
            let _ = writeln!(
                err,
                "warning: could not install the {DEFAULT_SET} plugins: {reason}. {retry}"
            );
        }
        SetupPluginsOutcome::Finished {
            unavailable,
            failed,
            ..
        } => {
            for (name, reason) in unavailable {
                let _ = writeln!(
                    err,
                    "warning: plugin {name} is not available on this platform: {reason}"
                );
            }
            for (name, reason) in failed {
                let _ = writeln!(err, "warning: could not install plugin {name}: {reason}");
            }
            if !failed.is_empty() {
                let _ = writeln!(err, "{retry}");
            }
        }
        SetupPluginsOutcome::NotRequested => {}
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SetupServiceOutcome {
    NotRequested,
    Installed(ServiceInstallReport),
    PrintedGuidance,
}

impl SetupActions for CliSetupActions<'_> {
    type Error = anyhow::Error;
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
                SetupStep::InstallPlugins => self.install_plugins().await,
                SetupStep::InstallService => self.install_service(),
                SetupStep::PrintServiceGuidance => {
                    self.service_outcome = SetupServiceOutcome::PrintedGuidance;
                    self.print_service_guidance();
                    Ok(())
                }
            }
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
