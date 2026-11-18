//! Fail fast when `serve`'s API or console port is already taken.
//!
//! The ports are bound late, after the mesh is up and its invite printed, so
//! a second node started on a machine whose node already holds them (the
//! background service `mesh-llm setup` starts, typically) used to get that far
//! and then die on a bare OS error. Checking first stops it before any of
//! that, and says what to do instead: when the holder is a mesh-llm node that
//! names itself, the advice targets that node; otherwise it only says the
//! port is taken.

use std::time::Duration;

use anyhow::Result;

use super::RuntimeOptions;
#[cfg(not(target_os = "macos"))]
use super::serving_surface::bind_runtime_tcp_listener;

const STATUS_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

/// Check each serving port once. A port in use ends startup with
/// [`port_in_use_message`]; any other bind error is left to the real bind
/// later, which reports it as before. Port 0 asks the OS for a free port and
/// is skipped.
pub(super) async fn ensure_serving_ports_free(options: &RuntimeOptions) -> Result<()> {
    // Each probe releases its port, so two listeners on one port would pass
    // here and only collide once the node has started.
    if options.port != 0 && options.port == options.console {
        anyhow::bail!(
            "the API port and the console port are both {}; give them different ports with \
             --port and --console",
            options.port
        );
    }
    for (port, label, flag) in [
        (options.port, "OpenAI-compatible API", "--port"),
        (options.console, "web console", "--console"),
    ] {
        if port == 0 {
            continue;
        }
        if port_is_taken(port, options.listen_all, label).await {
            let holder = find_mesh_node_holding(port, flag == "--console", options.console).await;
            anyhow::bail!(
                "{}",
                port_in_use_message(port, label, flag, holder.as_ref())
            )
        }
    }
    Ok(())
}

/// Bind the port once and release it. On Windows a port held exclusively on
/// every address (a node started with `--listen-all`) fails a loopback bind
/// with access denied rather than address in use; access denied alone can
/// also mean a reserved port, so only a listener that answers counts as
/// taken then.
#[cfg(not(target_os = "macos"))]
async fn port_is_taken(port: u16, listen_all: bool, label: &str) -> bool {
    match bind_runtime_tcp_listener(port, listen_all, label).await {
        Ok(listener) => {
            drop(listener);
            false
        }
        Err(error) => is_addr_in_use(&error) || accepts_local_connections(port),
    }
}

/// On macOS a loopback bind succeeds beside a node listening on `0.0.0.0`,
/// and while the probe holds it, new connections to the port reach the probe
/// and are reset when it closes. Only a connection is tried there, which
/// leaves the running node's traffic alone. A node listening on every
/// address answers on loopback too, so it is seen as well.
#[cfg(target_os = "macos")]
async fn port_is_taken(port: u16, _listen_all: bool, _label: &str) -> bool {
    accepts_local_connections(port)
}

/// Whether something already accepts connections on the loopback port.
fn accepts_local_connections(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(300),
    )
    .is_ok()
}

#[cfg(not(target_os = "macos"))]
fn is_addr_in_use(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::AddrInUse)
    })
}

/// A mesh-llm process that named itself as the holder of a taken port.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PortHolder {
    pid: Option<u32>,
    /// The console port its status answered on, which `mesh-llm load
    /// --port` takes.
    management_port: u16,
    is_client: bool,
}

/// Ask the console that may belong to the holder who it is. A taken console
/// port is asked directly; for a taken API port, the consoles this node would
/// use and the default one are asked, and only one whose status reports that
/// API port counts, since an API port alone says nothing about which console
/// manages it.
async fn find_mesh_node_holding(
    port: u16,
    is_console_port: bool,
    console: u16,
) -> Option<PortHolder> {
    let client = reqwest::Client::builder()
        .timeout(STATUS_REQUEST_TIMEOUT)
        .build()
        .ok()?;
    let default_console = RuntimeOptions::default().console;
    let candidates: Vec<u16> = if is_console_port {
        vec![port]
    } else {
        let mut candidates = vec![console, default_console];
        candidates.dedup();
        candidates.retain(|candidate| *candidate != 0 && *candidate != port);
        candidates
    };
    for candidate in candidates {
        let url = format!("http://127.0.0.1:{candidate}/api/status");
        let Ok(response) = client.get(url).send().await else {
            continue;
        };
        let Ok(status) = response.json::<serde_json::Value>().await else {
            continue;
        };
        if let Some(holder) = holder_from_status(&status, port, is_console_port, candidate) {
            return Some(holder);
        }
    }
    None
}

/// The holder a node's `/api/status`, served on `management_port`, names for
/// `port`; `None` unless it is a mesh-llm status that accounts for the port.
fn holder_from_status(
    status: &serde_json::Value,
    port: u16,
    is_console_port: bool,
    management_port: u16,
) -> Option<PortHolder> {
    let api_port = status["api_port"].as_u64()?;
    if !is_console_port && api_port != u64::from(port) {
        return None;
    }
    let pid = status["local_instances"].as_array().and_then(|instances| {
        instances
            .iter()
            .find(|instance| instance["is_self"].as_bool() == Some(true))
            .and_then(|instance| instance["pid"].as_u64())
            .and_then(|pid| u32::try_from(pid).ok())
    });
    Some(PortHolder {
        pid,
        management_port,
        is_client: status["is_client"].as_bool() == Some(true),
    })
}

fn port_in_use_message(port: u16, label: &str, flag: &str, holder: Option<&PortHolder>) -> String {
    let Some(holder) = holder else {
        return format!(
            "port {port} for the {label} is already in use by another program. Stop it, or \
             choose another port with {flag}."
        );
    };
    let kind = if holder.is_client { "client" } else { "node" };
    let named = match holder.pid {
        Some(pid) => format!("the mesh-llm {kind} with pid {pid}"),
        None => format!("a mesh-llm {kind}"),
    };
    let stop = stop_hint(holder.pid);
    if holder.is_client {
        return format!(
            "port {port} for the {label} is held by {named} on this machine. Stop it{stop} \
             before starting a node, or choose another port with {flag}."
        );
    }
    format!(
        "port {port} for the {label} is held by {named} on this machine. Load a model into it \
         with `mesh-llm load <model> --port {}`, or stop it{stop} before starting another one. \
         Otherwise choose another port with {flag}.",
        holder.management_port
    )
}

// On Windows the service's supervisor restarts a node ended on its own, so
// the service itself has to be ended. Dashes, not slashes: schtasks and
// taskkill take both, and the console's path redaction would turn `/End`
// and `/TN` into `[REDACTED_PATH]`.
#[cfg(windows)]
fn stop_hint(pid: Option<u32>) -> String {
    const SERVICE: &str = "`schtasks -End -TN mesh-llm` if it is the background service";
    match pid {
        Some(pid) => format!(" ({SERVICE}, otherwise `taskkill -PID {pid}`)"),
        None => format!(" ({SERVICE})"),
    }
}

#[cfg(not(windows))]
fn stop_hint(pid: Option<u32>) -> String {
    match pid {
        Some(pid) => format!(" with `kill {pid}`"),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::serving_surface::bind_runtime_tcp_listener;

    async fn held_port() -> (tokio::net::TcpListener, u16) {
        let listener = bind_runtime_tcp_listener(0, false, "test holder")
            .await
            .expect("bind a free port");
        let port = listener.local_addr().expect("holder address").port();
        (listener, port)
    }

    /// Holds a free port and answers every request on it with `status`, as a
    /// node's console does.
    async fn status_server(status: serde_json::Value) -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (listener, port) = held_port().await;
        let body = status.to_string();
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
        port
    }

    fn node_status(api_port: u16, pid: u32, is_client: bool) -> serde_json::Value {
        serde_json::json!({
            "api_port": api_port,
            "is_client": is_client,
            "local_instances": [
                {"pid": pid + 1, "is_self": false},
                {"pid": pid, "is_self": true}
            ]
        })
    }

    #[tokio::test]
    async fn a_port_held_by_another_program_says_only_that_it_is_taken() {
        let (_holder, port) = held_port().await;
        let options = RuntimeOptions {
            port,
            console: 0,
            ..RuntimeOptions::default()
        };
        let message = ensure_serving_ports_free(&options)
            .await
            .expect_err("a held API port must stop startup")
            .to_string();
        assert!(message.contains(&format!("port {port} for the OpenAI-compatible API")));
        assert!(message.contains("another program"));
        assert!(message.contains("--port"));
        // Nothing here identifies a node to load into or stop.
        assert!(!message.contains("mesh-llm load"));
        assert!(!message.contains("mesh-llm stop"));
    }

    #[tokio::test]
    async fn a_console_port_held_by_a_node_points_at_that_node() {
        let console = status_server(node_status(9447, 4242, false)).await;
        let options = RuntimeOptions {
            port: 0,
            console,
            ..RuntimeOptions::default()
        };
        let message = ensure_serving_ports_free(&options)
            .await
            .expect_err("a held console port must stop startup")
            .to_string();
        assert!(message.contains(&format!("port {console} for the web console")));
        assert!(message.contains("pid 4242"));
        assert!(message.contains(&format!("mesh-llm load <model> --port {console}")));
        assert!(message.contains("--console"));
        assert!(!message.contains("mesh-llm stop"));
    }

    #[tokio::test]
    async fn an_api_port_is_tied_to_the_console_that_reports_it() {
        let (_api, api_port) = held_port().await;
        let console = status_server(node_status(api_port, 4242, false)).await;
        let options = RuntimeOptions {
            port: api_port,
            console,
            ..RuntimeOptions::default()
        };
        let message = ensure_serving_ports_free(&options)
            .await
            .expect_err("a held API port must stop startup")
            .to_string();
        assert!(message.contains("pid 4242"));
        assert!(message.contains(&format!("mesh-llm load <model> --port {console}")));
    }

    #[tokio::test]
    async fn a_console_reporting_another_api_port_does_not_claim_this_one() {
        let (_api, api_port) = held_port().await;
        let console = status_server(node_status(api_port.wrapping_add(1), 4242, false)).await;
        assert_eq!(find_mesh_node_holding(api_port, false, console).await, None);
    }

    #[test]
    fn a_status_names_its_holder_only_for_the_ports_it_accounts_for() {
        let node = node_status(9337, 7, false);
        let named = PortHolder {
            pid: Some(7),
            management_port: 3145,
            is_client: false,
        };
        assert_eq!(
            holder_from_status(&node, 9337, false, 3145),
            Some(named.clone())
        );
        assert_eq!(holder_from_status(&node, 9447, false, 3145), None);
        assert_eq!(holder_from_status(&node, 3145, true, 3145), Some(named));
        // A JSON answer that is not a mesh-llm status.
        assert_eq!(
            holder_from_status(&serde_json::json!({"ok": true}), 3145, true, 3145),
            None
        );
        // A node whose instance scan has not run yet is still a node.
        assert_eq!(
            holder_from_status(&serde_json::json!({"api_port": 9337}), 9337, false, 3131),
            Some(PortHolder {
                pid: None,
                management_port: 3131,
                is_client: false,
            })
        );
    }

    #[test]
    fn a_client_holding_the_port_is_never_offered_a_load() {
        let client = PortHolder {
            pid: Some(7),
            management_port: 3131,
            is_client: true,
        };
        let message = port_in_use_message(9337, "OpenAI-compatible API", "--port", Some(&client));
        assert!(message.contains("mesh-llm client with pid 7"));
        assert!(!message.contains("mesh-llm load"));
        assert!(!message.contains("mesh-llm stop"));
        assert!(message.contains("--port"));
    }

    #[test]
    fn the_stop_advice_names_the_holders_pid() {
        let node = PortHolder {
            pid: Some(7),
            management_port: 3131,
            is_client: false,
        };
        let message = port_in_use_message(9337, "OpenAI-compatible API", "--port", Some(&node));
        #[cfg(windows)]
        {
            assert!(message.contains("taskkill -PID 7"));
            assert!(message.contains("schtasks -End -TN mesh-llm"));
        }
        #[cfg(not(windows))]
        assert!(message.contains("kill 7"));
        assert!(!message.contains("mesh-llm stop"));
    }

    // Windows only: elsewhere a loopback bind beside a wildcard listener can
    // succeed (it does on macOS, which only connects), and Linux refuses it
    // with address in use already.
    #[cfg(windows)]
    #[tokio::test]
    async fn a_port_held_on_every_address_is_reported_for_a_loopback_node() {
        // A node started with --listen-all holds 0.0.0.0:P exclusively; on
        // Windows a loopback bind of P then fails with access denied rather
        // than address in use.
        let holder = crate::network::exclusive_listener::bind_exclusive_tcp_listener("0.0.0.0:0")
            .await
            .expect("bind a wildcard holder");
        let port = holder.local_addr().expect("holder address").port();
        let options = RuntimeOptions {
            port,
            console: 0,
            ..RuntimeOptions::default()
        };
        let message = ensure_serving_ports_free(&options)
            .await
            .expect_err("a port held on every address must stop startup")
            .to_string();
        assert!(message.contains(&format!("port {port} for the OpenAI-compatible API")));
        drop(holder);
    }

    #[tokio::test]
    async fn the_same_port_for_api_and_console_is_refused() {
        let options = RuntimeOptions {
            port: 9337,
            console: 9337,
            ..RuntimeOptions::default()
        };
        let message = ensure_serving_ports_free(&options)
            .await
            .expect_err("one port for both listeners must stop startup")
            .to_string();
        assert!(message.contains("both 9337"));
        assert!(message.contains("--port") && message.contains("--console"));
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
