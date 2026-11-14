//! Open the web console at the end of an interactive Windows setup.
//!
//! When setup has just started the background service, the node it runs
//! serves the console on the default port. Someone who installed from
//! `install.ps1` would otherwise have to copy that address from the summary
//! into a browser, which is the step a person new to terminals is most likely
//! to miss. Setup waits for the console to answer, then opens it.

use super::command::SetupServiceOutcome;
use super::environment::{SetupEnvironment, SetupOptions, SetupPlatform};
use super::service::ServiceInstallStatus;
use crate::terminal::{style_muted, style_ok};
use std::io::Write;
use std::time::Duration;

/// The console the service's node serves on its default port.
pub(crate) const CONSOLE_URL: &str = "http://localhost:3131";
const CONSOLE_ADDR: (&str, u16) = ("127.0.0.1", 3131);
/// Long enough for a first start that loads the native runtime.
const CONSOLE_READY_TIMEOUT: Duration = Duration::from_secs(60);
const CONSOLE_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Whether setup should open the console: on Windows, when this run started
/// the service, and only when a person is answering the prompts. `--yes`,
/// `--no-interactive` and runs without a terminal never open a browser.
pub(crate) fn opens_console_after_setup(
    environment: SetupEnvironment,
    options: SetupOptions,
    service: &SetupServiceOutcome,
) -> bool {
    environment.platform == SetupPlatform::Windows
        && environment.prompts_visible(options)
        && matches!(
            service,
            SetupServiceOutcome::Installed(report)
                if report.status == ServiceInstallStatus::Started
        )
}

/// Wait for the console to accept connections, then open it in the default
/// browser. Never fails setup: on a timeout or an opener error it says where
/// the console is instead.
pub(crate) async fn open_console_when_ready() {
    let mut err = mesh_llm_events::console_err();
    let _ = writeln!(
        err,
        "{}",
        style_muted(&format!("Waiting for the console at {CONSOLE_URL}…"))
    );
    if !wait_for_console(CONSOLE_READY_TIMEOUT).await {
        let _ = writeln!(
            err,
            "The console did not answer within {} s. Open {CONSOLE_URL} once the node is up.",
            CONSOLE_READY_TIMEOUT.as_secs()
        );
        return;
    }
    match open_in_browser(CONSOLE_URL) {
        Ok(()) => {
            let _ = writeln!(err, "{} Opened the console at {CONSOLE_URL}", style_ok("✓"));
        }
        Err(_) => {
            let _ = writeln!(err, "Open the console at {CONSOLE_URL}");
        }
    }
}

async fn wait_for_console(timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if tokio::net::TcpStream::connect(CONSOLE_ADDR).await.is_ok() {
            return true;
        }
        if tokio::time::Instant::now() + CONSOLE_POLL_INTERVAL > deadline {
            return false;
        }
        tokio::time::sleep(CONSOLE_POLL_INTERVAL).await;
    }
}

/// Hand the URL to the default browser through the shell's URL protocol
/// handler, which opens no console window and needs no quoting.
#[cfg(windows)]
fn open_in_browser(url: &str) -> std::io::Result<()> {
    std::process::Command::new("rundll32.exe")
        .args(["url.dll,FileProtocolHandler", url])
        .spawn()
        .map(drop)
}

#[cfg(not(windows))]
fn open_in_browser(_url: &str) -> std::io::Result<()> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::service::ServiceInstallReport;

    fn interactive(platform: SetupPlatform) -> SetupEnvironment {
        SetupEnvironment {
            platform,
            interactive: true,
        }
    }

    fn installed(status: ServiceInstallStatus) -> SetupServiceOutcome {
        SetupServiceOutcome::Installed(ServiceInstallReport {
            status,
            summary: String::new(),
            messages: Vec::new(),
            service_file: std::path::PathBuf::new(),
            env_file: None,
            runner_file: None,
        })
    }

    #[test]
    fn opens_after_an_interactive_windows_setup_that_started_the_service() {
        assert!(opens_console_after_setup(
            interactive(SetupPlatform::Windows),
            SetupOptions::default(),
            &installed(ServiceInstallStatus::Started),
        ));
    }

    #[test]
    fn stays_closed_unless_the_service_is_running_now() {
        for service in [
            installed(ServiceInstallStatus::StartsAtLogon),
            installed(ServiceInstallStatus::NeedsManualStart),
            SetupServiceOutcome::NotRequested,
            SetupServiceOutcome::PrintedGuidance,
        ] {
            assert!(!opens_console_after_setup(
                interactive(SetupPlatform::Windows),
                SetupOptions::default(),
                &service,
            ));
        }
    }

    #[test]
    fn stays_closed_without_a_person_answering_the_prompts() {
        let started = installed(ServiceInstallStatus::Started);
        let unattended = [
            (
                SetupEnvironment {
                    platform: SetupPlatform::Windows,
                    interactive: false,
                },
                SetupOptions::default(),
            ),
            (
                interactive(SetupPlatform::Windows),
                SetupOptions {
                    yes: true,
                    ..SetupOptions::default()
                },
            ),
            (
                interactive(SetupPlatform::Windows),
                SetupOptions {
                    no_interactive: true,
                    ..SetupOptions::default()
                },
            ),
        ];
        for (environment, options) in unattended {
            assert!(!opens_console_after_setup(environment, options, &started));
        }
    }

    #[test]
    fn stays_closed_off_windows() {
        for platform in [SetupPlatform::Linux, SetupPlatform::MacOs] {
            assert!(!opens_console_after_setup(
                interactive(platform),
                SetupOptions::default(),
                &installed(ServiceInstallStatus::Started),
            ));
        }
    }
}
