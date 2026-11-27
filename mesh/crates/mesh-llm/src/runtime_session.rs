//! CLI serving-session lifetime, including its startup hardware survey.

use mesh_llm_commands::usage_reporting::{self, ServeSession};
use mesh_llm_host_runtime::{RuntimeOptions, RuntimeSurface};
use std::time::Duration;

pub(super) async fn run(
    options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    warning: Option<String>,
    session: ServeSession,
) -> anyhow::Result<()> {
    // Report startup failures too, after the attempted native runtime load.
    let initialized = mesh_llm_host_runtime::initialize_host_runtime_for_options(&options).await;
    let mut hardware_profile = usage_reporting::report_hardware_profile();
    if let Err(error) = initialized {
        let probe_shutdown = hardware_profile
            .finish_within(Duration::from_secs(10))
            .await;
        session.finish(Err(&error));
        usage_reporting::shutdown().await;
        report_probe_shutdown(probe_shutdown);
        return Err(error);
    }
    super::install_cli_operational_audit_bridge();
    mesh_llm_tui::output::OutputManager::init_global(
        options.log_format,
        mesh_llm_host_runtime::console_session_mode_for_runtime_surface(explicit_surface),
    );
    mesh_llm_tui::install_terminal_panic_hook();

    let outcome =
        mesh_llm_host_runtime::run_runtime_initialized(options, explicit_surface, warning).await;
    let probe_shutdown = hardware_profile.shutdown().await;
    session.finish(outcome.as_ref().map(|_| ()));
    usage_reporting::shutdown().await;
    report_probe_shutdown(probe_shutdown);
    outcome
}

/// The survey is best-effort telemetry: failing to stop a probe tool is worth
/// reporting but must not turn a clean shutdown into a failed exit (#3067).
fn report_probe_shutdown(result: anyhow::Result<()>) {
    if let Err(error) = result {
        use std::io::Write;
        let _ = writeln!(
            mesh_llm_events::console_err(),
            "warning: could not stop the hardware survey cleanly: {error:#}"
        );
    }
}
