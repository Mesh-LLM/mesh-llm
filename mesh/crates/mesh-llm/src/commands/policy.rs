use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use mesh_llm_cli::{Cli, PolicyCommand};
use mesh_llm_system::local_http::local_http_client_builder;
use reqwest::header::AUTHORIZATION;
use serde_json::json;
use std::io::Write;

pub(super) async fn dispatch_policy_command(command: &PolicyCommand, cli: &Cli) -> Result<()> {
    let _ = cli;
    match command {
        PolicyCommand::Set {
            ensure_models,
            drain_models,
            expires_in_secs,
            policy_gen,
            payload,
            signature,
            owner_key,
            port,
            json,
        } => {
            run_policy_set(
                ensure_models,
                drain_models,
                *expires_in_secs,
                *policy_gen,
                payload.as_deref(),
                signature.as_deref(),
                owner_key.as_deref(),
                *port,
                *json,
            )
            .await
        }
        PolicyCommand::Get { port, json } => run_policy_get(*port, *json).await,
        PolicyCommand::Clear { port, json } => run_policy_clear(*port, *json).await,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_policy_set(
    ensure_models: &[String],
    drain_models: &[String],
    expires_in_secs: Option<u64>,
    policy_gen: Option<u32>,
    payload: Option<&str>,
    signature: Option<&str>,
    owner_key: Option<&std::path::Path>,
    port: u16,
    json_output: bool,
) -> Result<()> {
    let token = mesh_llm_commands::owner_control::owner_control_token(port).await?;
    let client = local_http_client_builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/runtime/control/mesh-policy");

    let (payload_b64, signature_b64) = if let (Some(p), Some(s)) = (payload, signature) {
        // Pre-signed mode: use the values directly.
        (p.to_string(), s.to_string())
    } else {
        // Auto-sign mode: build the payload from the model lists, load the
        // owner keypair, sign, and base64-encode.
        build_and_sign_policy(
            ensure_models,
            drain_models,
            expires_in_secs,
            policy_gen,
            owner_key,
        )?
    };

    let body = json!({
        "policy_payload_bytes": payload_b64,
        "signature": signature_b64,
    });
    let response = client
        .post(&url)
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .json(&body)
        .send()
        .await
        .with_context(|| format!("Can't connect to mesh-llm on port {port}. Is it running?"))?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    print_policy_response("Set mesh policy", &response, json_output)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn build_and_sign_policy(
    ensure_models: &[String],
    drain_models: &[String],
    expires_in_secs: Option<u64>,
    policy_gen: Option<u32>,
    owner_key: Option<&std::path::Path>,
) -> Result<(String, String)> {
    use mesh_llm_identity::{
        CryptoError, KeystoreInfo, OwnerKeypair, default_keystore_path, keystore_metadata,
        load_keystore,
    };
    use mesh_llm_protocol::proto::node::{MeshModelIntent, MeshModelIntentAction, MeshPolicyPayload};
    use prost::Message;

    let keystore_path = match owner_key {
        Some(p) => p.to_path_buf(),
        None => default_keystore_path().context("Could not determine default keystore path")?,
    };
    // Respect encrypted keystores: check MESH_LLM_OWNER_PASSPHRASE first,
    // then prompt interactively (same unlock flow as the runtime).
    let passphrase: Option<String> = {
        let info: KeystoreInfo = keystore_metadata(&keystore_path)
            .with_context(|| format!("Could not read keystore metadata from {}", keystore_path.display()))?;
        if info.encrypted {
            if let Ok(p) = std::env::var("MESH_LLM_OWNER_PASSPHRASE") {
                Some(p)
            } else if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
                let prompt = format!(
                    "Enter owner keystore passphrase for {}: ",
                    keystore_path.display()
                );
                Some(rpassword::prompt_password(&prompt)
                    .context("Failed to read passphrase")?)
            } else {
                return Err(CryptoError::MissingPassphrase.into());
            }
        } else {
            None
        }
    };
    let keypair: OwnerKeypair =
        load_keystore(&keystore_path, passphrase.as_deref())
            .with_context(|| format!("Failed to load owner keypair from {}", keystore_path.display()))?;

    let issued = now_ms();
    let expires = expires_in_secs
        .map(|s| issued + s * 1000)
        .unwrap_or(0);

    let mut model_intents = Vec::new();
    for m in ensure_models {
        model_intents.push(MeshModelIntent {
            canonical_model_ref: m.clone(),
            action: MeshModelIntentAction::Ensure as i32,
        });
    }
    for m in drain_models {
        model_intents.push(MeshModelIntent {
            canonical_model_ref: m.clone(),
            action: MeshModelIntentAction::Drain as i32,
        });
    }

    let payload = MeshPolicyPayload {
        issued_at_unix_ms: issued,
        expires_at_unix_ms: expires,
        owner_id: keypair.owner_id(),
        owner_sign_public_key: hex::encode(keypair.verifying_key().as_bytes()),
        policy_gen: policy_gen.unwrap_or(1),
        model_intents,
    };

    let payload_bytes = payload.encode_to_vec();
    let signature_bytes = keypair.sign_bytes(&payload_bytes);

    let payload_b64 = BASE64_STANDARD.encode(&payload_bytes);
    let signature_b64 = BASE64_STANDARD.encode(signature_bytes);

    Ok((payload_b64, signature_b64))
}

async fn run_policy_get(port: u16, json_output: bool) -> Result<()> {
    let client = local_http_client_builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/runtime/control/mesh-policy");
    let token = mesh_llm_commands::owner_control::owner_control_token(port).await?;
    let response = client
        .get(&url)
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Can't connect to mesh-llm on port {port}: {e}"))?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    print_policy_response("Get mesh policy", &response, json_output)
}

async fn run_policy_clear(port: u16, json_output: bool) -> Result<()> {
    let token = mesh_llm_commands::owner_control::owner_control_token(port).await?;
    let client = local_http_client_builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/runtime/control/mesh-policy");
    let response = client
        .delete(&url)
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Can't connect to mesh-llm on port {port}: {e}"))?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    print_policy_response("Clear mesh policy", &response, json_output)
}

fn print_policy_response(
    title: &str,
    body: &serde_json::Value,
    json_output: bool,
) -> Result<()> {
    if !json_output {
        let mut out = mesh_llm_events::console_out();
        writeln!(out, "🔐 {title}")?;
        writeln!(out)?;
    }
    let mut out = mesh_llm_events::machine_out();
    writeln!(out, "{}", serde_json::to_string_pretty(body)?)?;
    Ok(())
}
