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
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The console the service's node serves on its default port.
pub(crate) const CONSOLE_URL: &str = "http://localhost:3131";
/// The status the console's node serves, which names the answering instance.
const STATUS_URL: &str = "http://127.0.0.1:3131/api/status";
/// Long enough for a first start that loads the native runtime.
const CONSOLE_READY_TIMEOUT: Duration = Duration::from_secs(60);
const CONSOLE_POLL_INTERVAL: Duration = Duration::from_millis(500);
const STATUS_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

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

/// Wait for the node setup started to serve the console, then open it in the
/// default browser. Never fails setup: on a timeout or an opener error it says
/// where the console is instead.
///
/// `setup_started` ties readiness to that node: the console counts as ready
/// only once `/api/status` on its port names itself as an instance started
/// since then. A foreground node or another program already on port 3131
/// answers too, but not as an instance this run started, and the service's
/// node cannot bind there.
pub(crate) async fn open_console_when_ready(setup_started: SystemTime) {
    let mut err = mesh_llm_events::console_err();
    let _ = writeln!(
        err,
        "{}",
        style_muted(&format!("Waiting for the console at {CONSOLE_URL}…"))
    );
    let since_unix = unix_seconds(setup_started);
    if !wait_for_console(STATUS_URL, since_unix, CONSOLE_READY_TIMEOUT).await {
        let _ = writeln!(
            err,
            "The background service's node did not answer at {CONSOLE_URL} within {} s. If another program uses port 3131, stop it; then open {CONSOLE_URL} once the node is up.",
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

async fn wait_for_console(status_url: &str, since_unix: i64, timeout: Duration) -> bool {
    let Ok(client) = reqwest::Client::builder()
        .timeout(STATUS_REQUEST_TIMEOUT)
        .build()
    else {
        return false;
    };
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if served_by_instance_started_since(&client, status_url, since_unix).await {
            return true;
        }
        if tokio::time::Instant::now() + CONSOLE_POLL_INTERVAL > deadline {
            return false;
        }
        tokio::time::sleep(CONSOLE_POLL_INTERVAL).await;
    }
}

async fn served_by_instance_started_since(
    client: &reqwest::Client,
    status_url: &str,
    since_unix: i64,
) -> bool {
    let Ok(response) = client.get(status_url).send().await else {
        return false;
    };
    if !response.status().is_success() {
        return false;
    }
    response
        .json::<serde_json::Value>()
        .await
        .is_ok_and(|status| names_itself_started_since(&status, since_unix))
}

/// Whether a node's status names the answering instance (`is_self`) as one
/// started at or after `since_unix`. The node fills `local_instances` on its
/// first scan, a few seconds after it starts, so an early answer waits.
fn names_itself_started_since(status: &serde_json::Value, since_unix: i64) -> bool {
    status["local_instances"]
        .as_array()
        .is_some_and(|instances| {
            instances.iter().any(|instance| {
                instance["is_self"].as_bool() == Some(true)
                    && instance["started_at_unix"]
                        .as_i64()
                        .is_some_and(|started| started >= since_unix)
            })
        })
}

fn unix_seconds(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs() as i64)
}

/// Hand the URL to the default browser; an error means the shell did not take
/// it, and setup prints the address instead.
fn open_in_browser(url: &str) -> std::io::Result<()> {
    mesh_llm_system::browser::open_url(url)
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
    fn only_an_answering_instance_started_since_setup_counts() {
        let status = |is_self: bool, started: i64| {
            serde_json::json!({"local_instances": [
                {"pid": 1, "is_self": is_self, "started_at_unix": started}
            ]})
        };
        assert!(names_itself_started_since(&status(true, 1_000), 1_000));
        assert!(names_itself_started_since(&status(true, 1_005), 1_000));
        // A node already running before setup, as a foreground `serve`.
        assert!(!names_itself_started_since(&status(true, 999), 1_000));
        // A newer node, but not the one answering on the port.
        assert!(!names_itself_started_since(&status(false, 1_005), 1_000));
        // Before the node's first instance scan.
        assert!(!names_itself_started_since(
            &serde_json::json!({"local_instances": []}),
            1_000
        ));
        assert!(!names_itself_started_since(&serde_json::json!({}), 1_000));
    }

    /// Serves `body` with status 200 to every request on a free local port.
    async fn serve(body: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut request = [0u8; 1024];
                let _ = stream.read(&mut request).await;
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{address}/api/status")
    }

    #[tokio::test]
    async fn an_occupied_port_does_not_count_as_the_services_console() {
        // Another program answers on the port.
        let other = serve("ok").await;
        assert!(!wait_for_console(&other, 1_000, Duration::from_millis(600)).await);
        // An older node answers on the port.
        let older =
            serve(r#"{"local_instances":[{"pid":7,"is_self":true,"started_at_unix":999}]}"#).await;
        assert!(!wait_for_console(&older, 1_000, Duration::from_millis(600)).await);
        // The node setup started answers.
        let started =
            serve(r#"{"local_instances":[{"pid":8,"is_self":true,"started_at_unix":1000}]}"#).await;
        assert!(wait_for_console(&started, 1_000, Duration::from_millis(600)).await);
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
