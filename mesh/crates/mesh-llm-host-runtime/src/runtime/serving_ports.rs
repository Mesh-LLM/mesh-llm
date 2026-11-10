//! Fail fast when `serve`'s API or console port is already taken.
//!
//! The ports are bound late, after the mesh is up and its invite printed, so
//! a second node started on a machine whose node already holds them (the
//! background service `mesh-llm setup` starts, typically) used to get that far
//! and then die on a bare OS error. Checking first stops it before any of
//! that, and says what to do instead.

use anyhow::Result;

use super::RuntimeOptions;
use super::serving_surface::bind_runtime_tcp_listener;

/// Bind each serving port once and release it. A port in use ends startup
/// with [`port_in_use_message`]; any other bind error is left to the real
/// bind later, which reports it as before. Port 0 asks the OS for a free port
/// and is skipped.
pub(super) async fn ensure_serving_ports_free(options: &RuntimeOptions) -> Result<()> {
    for (port, label, flag) in [
        (options.port, "OpenAI-compatible API", "--port"),
        (options.console, "web console", "--console"),
    ] {
        if port == 0 {
            continue;
        }
        match bind_runtime_tcp_listener(port, options.listen_all, label).await {
            Ok(listener) => drop(listener),
            Err(error) if is_addr_in_use(&error) => {
                anyhow::bail!("{}", port_in_use_message(port, label, flag))
            }
            Err(_) => {}
        }
    }
    Ok(())
}

fn is_addr_in_use(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::AddrInUse)
    })
}

fn port_in_use_message(port: u16, label: &str, flag: &str) -> String {
    format!(
        "port {port} for the {label} is already in use. If a mesh-llm node is already \
         running on this machine (the background service that `mesh-llm setup` installs \
         starts one), load a model into it with `mesh-llm load <model>`, or stop it with \
         {STOP_HINT} before starting another one. Otherwise choose another port with {flag}."
    )
}

// On Windows the service's supervisor restarts a node that `mesh-llm stop`
// force-ends, so the service itself has to be ended. Dashes, not slashes:
// schtasks takes both, and the console's path redaction would turn `/End`
// and `/TN` into `[REDACTED_PATH]`.
#[cfg(windows)]
const STOP_HINT: &str = "`schtasks -End -TN mesh-llm` (the background service) or `mesh-llm stop`";
#[cfg(not(windows))]
const STOP_HINT: &str = "`mesh-llm stop`";

#[cfg(test)]
mod tests {
    use super::*;

    async fn held_port() -> (tokio::net::TcpListener, u16) {
        let listener = bind_runtime_tcp_listener(0, false, "test holder")
            .await
            .expect("bind a free port");
        let port = listener.local_addr().expect("holder address").port();
        (listener, port)
    }

    #[tokio::test]
    async fn a_held_api_port_stops_startup_with_what_to_do() {
        let (_holder, port) = held_port().await;
        let options = RuntimeOptions {
            port,
            console: 0,
            ..RuntimeOptions::default()
        };
        let error = ensure_serving_ports_free(&options)
            .await
            .expect_err("a held API port must stop startup");
        let message = error.to_string();
        assert!(message.contains(&format!("port {port} for the OpenAI-compatible API")));
        assert!(message.contains("mesh-llm load <model>"));
        assert!(message.contains(STOP_HINT));
        assert!(message.contains("--port"));
    }

    #[tokio::test]
    async fn a_held_console_port_names_the_console_flag() {
        let (_holder, port) = held_port().await;
        let options = RuntimeOptions {
            port: 0,
            console: port,
            ..RuntimeOptions::default()
        };
        let message = ensure_serving_ports_free(&options)
            .await
            .expect_err("a held console port must stop startup")
            .to_string();
        assert!(message.contains("web console"));
        assert!(message.contains("--console"));
    }

    #[tokio::test]
    async fn free_ports_pass_and_stay_free_for_the_real_bind() {
        let (holder, port) = held_port().await;
        drop(holder);
        let options = RuntimeOptions {
            port,
            console: 0,
            ..RuntimeOptions::default()
        };
        ensure_serving_ports_free(&options)
            .await
            .expect("a free port passes");
        bind_runtime_tcp_listener(port, false, "real bind")
            .await
            .expect("the check released the port");
    }
}
