//! Two actual Mesh hosts and production payment engines with explicit offline wallet/backend fixtures.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::Duration,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    mesh_bin: PathBuf,
    /// Fresh owner-only output directory; contains private invitation/config/log files.
    #[arg(long)]
    output_dir: PathBuf,
}
struct Running(Child);
impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[derive(Clone, Copy)]
struct Ports {
    api: u16,
    console: u16,
}
fn port() -> Result<u16> {
    Ok(
        std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?
            .local_addr()?
            .port(),
    )
}
fn private_write(path: &Path, content: &str) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    use std::io::Write;
    let mut file = options.open(path)?;
    file.write_all(content.as_bytes())?;
    file.sync_all()?;
    Ok(())
}
fn quote(path: &Path) -> String {
    serde_json::to_string(&path.to_string_lossy()).unwrap()
}
fn package_plugins(directory: &Path, bins: &Path) -> Result<Vec<(String, PathBuf)>> {
    let packages = directory.join("packages");
    std::fs::create_dir(&packages)?;
    let mut archives = Vec::new();
    for name in [
        "pay-stablecoin",
        "demo-wallet-fixture",
        "demo-backend-fixture",
    ] {
        let stage = packages.join(name);
        std::fs::create_dir(&stage)?;
        std::fs::copy(bins.join(name), stage.join(name))?;
        private_write(
            &stage.join("plugin.toml"),
            &format!(
                "name = \"{name}\"\nversion = \"{}\"\n",
                env!("CARGO_PKG_VERSION")
            ),
        )?;
        let output = Command::new(bins.join(name)).arg("--manifest").output()?;
        ensure!(
            output.status.success(),
            "manifest generation failed for {name}"
        );
        let manifest = String::from_utf8(output.stdout)?;
        serde_json::from_str::<Value>(&manifest)?;
        private_write(&stage.join("plugin-manifest.json"), &manifest)?;
        let archive = packages.join(format!("{name}.tar.gz"));
        let output = Command::new("tar")
            .arg("-C")
            .arg(&packages)
            .arg("-czf")
            .arg(&archive)
            .arg(name)
            .output()?;
        ensure!(
            output.status.success(),
            "archive creation failed for {name}"
        );
        archives.push((name.to_owned(), archive));
    }
    Ok(archives)
}
fn install_plugins(
    mesh: &Path,
    directory: &Path,
    archives: &[(String, PathBuf)],
    include_backend: bool,
) -> Result<()> {
    std::fs::create_dir(directory)?;
    let bootstrap = directory.join("install-config.toml");
    private_write(&bootstrap, "version = 1\n")?;
    for (name, archive) in archives {
        if name == "demo-backend-fixture" && !include_backend {
            continue;
        }
        let output = Command::new(mesh)
            .args(["plugins", "install", "--archive"])
            .arg(archive)
            .args(["--name", name, "--version", env!("CARGO_PKG_VERSION")])
            .env("MESH_LLM_CONFIG", &bootstrap)
            .env("MESH_LLM_PLUGIN_DIR", directory.join("installed-plugins"))
            .env("MESH_LLM_NODE_KEY_PATH", directory.join("node.key"))
            .env("MESH_LLM_RUNTIME_ROOT", directory.join("runtime"))
            .env("MESH_LLM_NO_UPDATE_CHECK", "1")
            .stdin(Stdio::null())
            .output()?;
        let log = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        private_write(&directory.join(format!("install-{name}.log")), &log)?;
        ensure!(
            output.status.success(),
            "install {name} failed; inspect the private install log"
        );
    }
    Ok(())
}
fn config(directory: &Path, chain: &Path, backend: Option<u16>, seller: &str) -> Result<()> {
    let seller_mode = backend.is_some();
    let payments = json!({"data_dir":directory.join("payments"),"wallet_binding":"wallet",
        "prices":if seller_mode {json!({"mesh-payment-fixture":{"input_per_million":"1000000","output_per_million":"1000000"}})} else {json!({})},
        "admission_fee":"3","buying":{"enabled":!seller_mode,"peers":if seller_mode {vec![]} else {vec![seller]},
            "maximum_price":{"input_per_million":"1000000","output_per_million":"1000000"},
            "max_payment":"100","max_fee":"0","daily_budget":"1000"}});
    private_write(
        &directory.join("payments.json"),
        &serde_json::to_string_pretty(&payments)?,
    )?;
    let mut config = format!(
        r#"version = 1

[[plugin]]
name = "demo-wallet-fixture"
[plugin.settings]
mode = "offline-fixture"
chain_path = {}
address = "0x{}"

[[plugin]]
name = "pay-stablecoin"
[plugin.settings]
config_path = {}

[[plugin.operation_bindings]]
name = "wallet"
provider = "demo-wallet-fixture"
profile = "offline"
setup_operation = "wallet_open"
setup_input_json = "{{}}"
operations = ["wallet_balance", "wallet_estimate_payment", "wallet_pay", "wallet_lookup", "wallet_verify_payment", "wallet_list_attempts"]

[plugin.openai_exchange_grant]
endpoints = ["chat_completions", "completions", "responses"]
phases = ["request_received", "backend_selected", "exchange_finished"]
request_body = true
effective_request_body = true
admission = true
read_identity_bundle = true
deadline_ms = 30000
max_body_bytes = 65536
max_queue_bytes = 2097152
max_in_flight = 32
failure_policy = "required"
"#,
        quote(chain),
        (if seller_mode { "22" } else { "11" }).repeat(20),
        quote(&directory.join("payments.json"))
    );
    if let Some(port) = backend {
        config.push_str(&format!(
            "\n[[plugin]]\nname = \"demo-backend-fixture\"\nurl = \"http://127.0.0.1:{port}/v1\"\n"
        ));
    } else {
        config.push_str("\n[runtime]\nmode = \"on_demand\"\n");
    }
    private_write(&directory.join("config.toml"), &config)
}
fn start(
    mesh: &Path,
    directory: &Path,
    ports: Ports,
    invitation: Option<&Path>,
) -> Result<Running> {
    let log_path = directory.join("host.log");
    if !log_path.exists() {
        private_write(&log_path, "")?;
    }
    let log = OpenOptions::new().append(true).open(log_path)?;
    let mut command = Command::new(mesh);
    command
        .arg("serve")
        .args([
            "--no-default-plugins",
            "--log-format",
            "json",
            "--disable-iroh-relays",
            "--bind-ip",
            "127.0.0.1",
            "--mesh-discovery-mode",
            "mdns",
        ])
        .arg("--port")
        .arg(ports.api.to_string())
        .arg("--console")
        .arg(ports.console.to_string())
        .env("MESH_LLM_CONFIG", directory.join("config.toml"))
        .env("MESH_LLM_PLUGIN_DIR", directory.join("installed-plugins"))
        .env("MESH_LLM_RUNTIME_ROOT", directory.join("runtime"))
        .env("MESH_LLM_NODE_KEY_PATH", directory.join("node.key"))
        .env("MESH_LLM_NO_UPDATE_CHECK", "1")
        .env("MESH_LLM_NO_RUNTIME_DOWNLOAD", "1")
        .env_remove("MESH_LLM_JOIN")
        .env_remove("MESH_LLM_JOIN_FILE")
        .env_remove("MESH_LLM_EPHEMERAL_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
    if let Some(path) = invitation {
        command
            .args(["--max-vram", "0", "--no-enumerate-host"])
            .arg("--join-file")
            .arg(path);
    }
    Ok(Running(command.spawn().context("start Mesh smoke node")?))
}
async fn get(client: &reqwest::Client, port: u16, path: &str) -> Result<Value> {
    Ok(client
        .get(format!("http://127.0.0.1:{port}{path}"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}
async fn status(client: &reqwest::Client, port: u16) -> Result<Value> {
    let response: Value = client
        .post(format!(
            "http://127.0.0.1:{port}/api/plugins/pay-stablecoin/tools/control"
        ))
        .json(&json!({"command":"status"}))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    if let Some(data) = response
        .get("structuredContent")
        .or_else(|| response.get("structured_content"))
    {
        return Ok(data.clone());
    }
    if let Some(text) = response["content"]
        .as_array()
        .and_then(|xs| xs.first())
        .and_then(|x| x["text"].as_str())
    {
        return Ok(serde_json::from_str(text)?);
    }
    Ok(response)
}
async fn wait<F, Fut>(label: &str, child: &mut Running, mut action: F) -> Result<Value>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<Value>>,
{
    let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
    loop {
        if let Some(exit) = child.0.try_wait()? {
            anyhow::bail!(
                "Mesh exited ({exit}) while waiting for {label}; inspect private host logs"
            );
        }
        if let Ok(value) = action().await {
            return Ok(value);
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "timeout waiting for {label}; inspect private host logs"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
async fn cleared(client: &reqwest::Client, port: u16, child: &mut Running) -> Result<Value> {
    wait("cleared payment debt", child, || async {
        let value = status(client, port).await?;
        ensure!(
            value["active"].as_array().is_some_and(Vec::is_empty),
            "settlement pending"
        );
        Ok(value)
    })
    .await
}
async fn buyer_ready(
    client: &reqwest::Client,
    ports: Ports,
    child: &mut Running,
) -> Result<String> {
    let state = wait("buyer payment plugin", child, || async {
        let state = status(client, ports.console).await?;
        ensure!(state["buying_enabled"] == true, "buyer plugin not ready");
        ensure!(
            state["node"]
                .as_str()
                .is_some_and(|id| id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit())),
            "buyer full identity not ready"
        );
        Ok(state)
    })
    .await?;
    wait("remote fixture model", child, || async {
        let value = get(client, ports.api, "/v1/models").await?;
        ensure!(
            value["data"]
                .as_array()
                .is_some_and(|xs| xs.iter().any(|x| x["id"] == "mesh-payment-fixture")),
            "model not advertised"
        );
        Ok(value)
    })
    .await?;
    Ok(state["node"]
        .as_str()
        .context("missing buyer identity")?
        .to_owned())
}
async fn execute(args: Args) -> Result<()> {
    let mesh = args.mesh_bin.canonicalize()?;
    ensure!(!args.output_dir.exists(), "--output-dir must be fresh");
    std::fs::create_dir_all(&args.output_dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&args.output_dir, std::fs::Permissions::from_mode(0o700))?;
    }
    let directory = args.output_dir.canonicalize()?;
    let bins = std::env::current_exe()?
        .parent()
        .context("binary directory missing")?
        .to_path_buf();
    for binary in [
        "pay-stablecoin",
        "demo-wallet-fixture",
        "demo-backend-fixture",
    ] {
        ensure!(
            bins.join(binary).is_file(),
            "build missing companion binary {binary}"
        );
    }
    let chain = directory.join("OFFLINE-fixture.sqlite3");
    let seller_dir = directory.join("seller");
    let buyer_dir = directory.join("buyer");
    let seller_ports = Ports {
        api: port()?,
        console: port()?,
    };
    let buyer_ports = Ports {
        api: port()?,
        console: port()?,
    };
    let archives = package_plugins(&directory, &bins)?;
    install_plugins(&mesh, &seller_dir, &archives, true)?;
    install_plugins(&mesh, &buyer_dir, &archives, false)?;
    config(&seller_dir, &chain, Some(port()?), "")?;
    let mut seller = start(&mesh, &seller_dir, seller_ports, None)?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .pool_max_idle_per_host(0)
        .timeout(Duration::from_secs(45))
        .build()?;
    let identity = wait("seller identity", &mut seller, || {
        get(&client, seller_ports.console, "/api/status")
    })
    .await?;
    let invitation = identity["token"]
        .as_str()
        .context("status missing private invite")?;
    private_write(&directory.join("invite.token"), invitation)?;
    let payment_status = wait("seller payment plugin", &mut seller, || async {
        let s = status(&client, seller_ports.console).await?;
        ensure!(
            s["node"]
                .as_str()
                .is_some_and(|id| id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit())),
            "payment plugin full identity not ready"
        );
        Ok(s)
    })
    .await?;
    let seller_id = payment_status["node"]
        .as_str()
        .context("payment plugin status missing full node identity")?;
    let public_wallet = client
        .post(format!(
            "http://127.0.0.1:{}/api/plugins/demo-wallet-fixture/tools/wallet_pay",
            seller_ports.console
        ))
        .json(&json!({}))
        .send()
        .await?;
    ensure!(
        public_wallet.status().as_u16() == 404,
        "wallet spending operation was exposed outside its host binding"
    );
    config(&buyer_dir, &chain, None, seller_id)?;
    let mut buyer = start(
        &mesh,
        &buyer_dir,
        buyer_ports,
        Some(&directory.join("invite.token")),
    )?;
    let buyer_id = buyer_ready(&client, buyer_ports, &mut buyer).await?;
    for number in 0..2 {
        let streaming = number == 1;
        let mut request = json!({"model":"mesh-payment-fixture","messages":[{"role":"user","content":"test"}],"max_tokens":4,"stream":streaming});
        if streaming {
            request["stream_options"] = json!({"include_usage":true});
        }
        let response = client
            .post(format!(
                "http://127.0.0.1:{}/v1/chat/completions",
                buyer_ports.api
            ))
            .json(&request)
            .send()
            .await?;
        let code = response.status();
        let text = response.text().await?;
        if !code.is_success() {
            private_write(&directory.join("inference-error.json"), &text)?;
        }
        ensure!(
            code.is_success(),
            "fixture inference rejected (HTTP {code}); inspect private logs"
        );
        let usage = if streaming {
            ensure!(
                text.contains("[DONE]") && text.contains("OFFLINE FIXTURE RESPONSE"),
                "incomplete fixture SSE stream"
            );
            text.lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                .find_map(|chunk| chunk.get("usage").filter(|u| u.is_object()).cloned())
                .context("stream omitted terminal usage")?
        } else {
            let body: Value = serde_json::from_str(&text)?;
            body["usage"].clone()
        };
        ensure!(
            usage["prompt_tokens"] == 1 && usage["completion_tokens"] == 4,
            "wrong fixture usage"
        );
        cleared(&client, buyer_ports.console, &mut buyer).await?;
        cleared(&client, seller_ports.console, &mut seller).await?;
        if number == 0 {
            drop(buyer);
            buyer = start(
                &mesh,
                &buyer_dir,
                buyer_ports,
                Some(&directory.join("invite.token")),
            )?;
            let restarted_id = buyer_ready(&client, buyer_ports, &mut buyer).await?;
            ensure!(
                restarted_id == buyer_id,
                "buyer payment identity changed across restart"
            );
        }
    }
    let connection = rusqlite::Connection::open(&chain)?;
    let mut statement = connection.prepare("SELECT attempt FROM fixture_payments")?;
    let encoded = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let attempts = encoded
        .iter()
        .map(|s| serde_json::from_str::<wallet_contract::PaymentAttempt>(s))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut amounts = attempts
        .iter()
        .map(|a| a.request.obligation.amount)
        .collect::<Vec<_>>();
    amounts.sort_unstable();
    ensure!(
        amounts == vec![2, 3, 5],
        "expected admission3, first remainder2, second usage5; got {amounts:?}"
    );
    private_write(
        &directory.join("result.json"),
        &serde_json::to_string_pretty(&json!({
        "offline":true,"real_mesh_hosts":2,"inference_requests":2,"buffered_requests":1,"streaming_requests":1,"fabricated_payments":3,
        "amounts_atomic":amounts,"settlement_total_atomic":10,"debt_cleared":true,
        "buyer_restart_identity_preserved":true,
        "wallet_spending_not_http_exposed":true,"mainnet_transactions_sent":0}))?,
    )?;
    println!(
        "PASS: actual two-peer Mesh IPC/routing, admission and exact usage settlement; offline fixtures only. Report: {}",
        directory.join("result.json").display()
    );
    Ok(())
}
#[tokio::main]
async fn main() -> Result<()> {
    execute(Args::parse()).await
}
