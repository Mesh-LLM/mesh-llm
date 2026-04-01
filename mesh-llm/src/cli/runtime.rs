use anyhow::{Context, Result};
use clap::Subcommand;
use serde_json::Value;

#[derive(Subcommand, Debug)]
pub(crate) enum RuntimeCommand {
    /// Show locally served runtime status on a running mesh-llm instance.
    Status {
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Load a local-only model into a running mesh-llm instance.
    Load {
        /// Model name/path/url to load
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Unload a local runtime-loaded model from a running mesh-llm instance.
    #[command(alias = "drop")]
    Unload {
        /// Model name to unload
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
}

pub(crate) async fn run_drop(model_name: &str, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let encoded = percent_encode_path_segment(model_name);
    let url = format!("http://127.0.0.1:{port}/api/runtime/models/{encoded}");
    let resp = client
        .delete(&url)
        .send()
        .await
        .with_context(|| format!("Can't connect to mesh-llm on port {port}. Is it running?"))?;
    display_runtime_result(resp, model_name, "Unloaded").await
}

pub(crate) async fn run_load(model_name: &str, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/runtime/models");
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"model": model_name}))
        .send()
        .await
        .with_context(|| format!("Can't connect to mesh-llm on port {port}. Is it running?"))?;
    display_runtime_result(resp, model_name, "Loaded").await
}

async fn display_runtime_result(
    resp: reqwest::Response,
    model_name: &str,
    verb: &str,
) -> Result<()> {
    let action_inf = if verb == "Loaded" { "load" } else { "unload" };
    if resp.status().is_success() {
        eprintln!("✅ {verb} runtime model");
        eprintln!();
        eprintln!("Model: {model_name}");
        eprintln!("Scope: Local node");
    } else {
        eprintln!("❌ Failed to {action_inf} runtime model");
        eprintln!();
        eprintln!("Model: {model_name}");
        let reason = resp
            .json::<serde_json::Value>()
            .await
            .ok()
            .and_then(|v| v["error"].as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown error".to_string());
        eprintln!("Reason: {reason}");
    }
    Ok(())
}

/// Percent-encode a string for use as a URL path segment.
/// Unreserved characters (A-Z a-z 0-9 - _ . ~) are passed through unchanged;
/// all other bytes are encoded as %XX.
fn percent_encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            b => {
                out.push('%');
                out.push(
                    char::from_digit((b >> 4) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
                out.push(
                    char::from_digit((b & 0xf) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
            }
        }
    }
    out
}

pub(crate) async fn run_status(port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let runtime_body = fetch_runtime_payload(&client, port, "/api/runtime").await?;
    let processes_body = fetch_runtime_payload(&client, port, "/api/runtime/processes").await?;

    let models = runtime_body["models"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Invalid runtime status payload"))?;
    let processes = processes_body["processes"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Invalid runtime process payload"))?;

    println!("⚙️  Runtime");
    println!();

    let primary_model = runtime_body["primary_model"].as_str().unwrap_or("none");
    println!("🧠 Primary: {primary_model}");

    if models.is_empty() {
        println!("📦 Models served locally: 0");
        println!();
        println!("No local models are currently being served.");
        return Ok(());
    }

    println!("📦 Models served locally: {}", models.len());
    println!();

    println!(
        "{:<42} {:<8} {:<8} {:<10} {:<8} {:<6} {:<8}",
        "Model", "Role", "Backend", "State", "Pid", "Port", "Source"
    );
    for model in models {
        let name = model["name"].as_str().unwrap_or("unknown");
        let kind = display_runtime_role(model["kind"].as_str().unwrap_or("unknown"));
        let backend = display_backend_label(model["backend"].as_str().unwrap_or("unknown"));
        let status = display_runtime_state(model["status"].as_str().unwrap_or("unknown"));
        let pid = find_pid(processes, model)
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into());
        let port = model["port"]
            .as_u64()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into());
        let source = if model["startup_managed"].as_bool().unwrap_or(false) {
            "Startup"
        } else {
            "Runtime"
        };
        println!(
            "{:<42} {:<8} {:<8} {:<10} {:<8} {:<6} {:<8}",
            name, kind, backend, status, pid, port, source
        );
    }

    Ok(())
}

fn display_runtime_role(value: &str) -> &'static str {
    match value {
        "primary" => "Primary",
        "runtime" => "Runtime",
        _ => "Unknown",
    }
}

fn display_runtime_state(value: &str) -> &'static str {
    match value {
        "ready" => "Ready",
        "starting" => "Starting",
        "stopped" => "Stopped",
        _ => "Unknown",
    }
}

fn display_backend_label(value: &str) -> &'static str {
    match value {
        "llama" => "Llama",
        _ => "Unknown",
    }
}

async fn fetch_runtime_payload(client: &reqwest::Client, port: u16, path: &str) -> Result<Value> {
    let url = format!("http://127.0.0.1:{port}{path}");
    client
        .get(&url)
        .send()
        .await
        .with_context(|| {
            format!("Can't connect to mesh-llm console on port {port}. Is it running?")
        })?
        .error_for_status()?
        .json::<Value>()
        .await
        .map_err(Into::into)
}

fn find_pid(processes: &[Value], model: &Value) -> Option<u64> {
    let name = model["name"].as_str()?;
    let kind = model["kind"].as_str()?;
    processes
        .iter()
        .find(|process| {
            process["name"].as_str() == Some(name) && process["kind"].as_str() == Some(kind)
        })
        .and_then(|process| process["pid"].as_u64())
}
