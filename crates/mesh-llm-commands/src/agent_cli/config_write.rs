//! Opt-in config writers for independently operated agent harnesses.
use anyhow::{Context, Result, bail};
use mesh_llm_cli::agent_config::AgentConfigArgs;
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};

mod storage;

/// Configure a harness without starting it, Mesh, plugins, or credential discovery.
pub async fn run(args: &AgentConfigArgs, hermes: bool) -> Result<()> {
    let target = super::normalize_mesh_host(&args.host)?;
    let url = url::Url::parse(&target.api_base_url)?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
    {
        bail!("Use an HTTP(S) endpoint without embedded credentials");
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let body: Value = client
        .get(&target.api_models_url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let inventory = super::model_inventory::ModelInventory::from_response(&body);
    if inventory.names.is_empty() {
        bail!("Mesh has no available models; start Mesh first");
    }
    if !matches!(args.model.as_str(), "auto" | "mesh") && !inventory.names.contains(&args.model) {
        bail!("Requested model is not advertised by Mesh");
    }
    inventory.report_fallbacks();
    let served = if matches!(args.model.as_str(), "auto" | "mesh") {
        inventory
            .names
            .iter()
            .map(|id| inventory.context_limit(id))
            .min()
            .unwrap_or(8192)
    } else {
        inventory.context_limit(&args.model)
    };
    let context = args.context_length.unwrap_or(served).min(served);
    let path = args
        .config_path
        .clone()
        .map(Ok)
        .unwrap_or_else(|| default_path(hermes))?;
    let original = storage::read(&path)?;
    let updated = merge(
        original.as_deref(),
        hermes,
        &target.api_base_url,
        &args.model,
        context,
    )?;
    if original.is_some() {
        writeln!(
            mesh_llm_events::console_err(),
            "Writing harness config: comments/formatting will be normalized; an exact sibling backup will be retained."
        )?;
    }
    storage::save(&path, original.as_deref(), &updated)?;
    writeln!(
        mesh_llm_events::console_out(),
        "Mesh provider saved to {}. Default model and permissions unchanged. Select {} in your agent. Context budget: {context}; rerun when serving limits change. Comments/formatting normalized; existing file backed up alongside it.",
        path.display(),
        if hermes {
            format!("provider mesh, model {}", args.model)
        } else {
            format!("mesh/{}", args.model)
        }
    )?;
    Ok(())
}

fn default_path(hermes: bool) -> Result<PathBuf> {
    let home = dirs::home_dir().context("Cannot locate home; use --config-path")?;
    if hermes {
        Ok(std::env::var_os("HERMES_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".hermes"))
            .join("config.yaml"))
    } else {
        Ok(std::env::var_os("OPENCLAW_CONFIG_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("OPENCLAW_STATE_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| home.join(".openclaw"))
                    .join("openclaw.json")
            }))
    }
}

fn object(value: &mut Value) -> Result<&mut serde_json::Map<String, Value>> {
    value
        .as_object_mut()
        .context("Expected a config mapping; refusing to replace malformed settings")
}

fn child<'a>(value: &'a mut Value, key: &str) -> Result<&'a mut Value> {
    Ok(object(value)?.entry(key).or_insert_with(|| json!({})))
}

fn reject_includes(value: &Value) -> Result<()> {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if matches!(key.as_str(), "$include" | "<<") {
                    bail!("Included/merged configs are unsupported; edit the provider manually");
                }
                reject_includes(value)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                reject_includes(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn merge(
    original: Option<&[u8]>,
    hermes: bool,
    base: &str,
    model: &str,
    context: u32,
) -> Result<Vec<u8>> {
    let mut config: Value = match original {
        None => json!({}),
        Some(bytes) if hermes => serde_yaml::from_slice(bytes)
            .map_err(|_| anyhow::anyhow!("Invalid Hermes YAML config"))?,
        Some(bytes) => json5::from_str(std::str::from_utf8(bytes)?)
            .map_err(|_| anyhow::anyhow!("Invalid OpenClaw JSON5 config"))?,
    };
    reject_includes(&config)?;
    if hermes && config.get("custom_providers").is_some() {
        bail!(
            "Legacy custom_providers present; migrate to Hermes providers mapping before writing"
        );
    }
    let provider = if hermes {
        json!({"name":"Mesh", "base_url":base, "api_key":"mesh", "api_mode":"chat_completions", "models":{model:{"context_length":context}}, "context_length":context})
    } else {
        json!({"baseUrl":base,"apiKey":"mesh","api":"openai-completions","models":[{"id":model,"name":model,"input":["text"],"contextWindow":context,"maxTokens":(context/4).min(4096)}]})
    };
    let providers = if hermes {
        child(&mut config, "providers")?
    } else {
        child(child(&mut config, "models")?, "providers")?
    };
    let map = object(providers)?;
    if let Some(existing) = map.get("mesh") {
        if existing != &provider {
            bail!("A different mesh provider already exists; refusing to overwrite it");
        }
    } else {
        map.insert("mesh".into(), provider);
    }
    if !hermes {
        // An explicit model allowlist must include this connection, without changing the default.
        if let Some(allowlist) = config.pointer_mut("/agents/defaults/models") {
            object(allowlist)?
                .entry(format!("mesh/{model}"))
                .or_insert_with(|| json!({}));
        }
    }
    if hermes {
        Ok(serde_yaml::to_string(&config)?.into_bytes())
    } else {
        Ok(serde_json::to_vec_pretty(&config)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_defaults_and_permissions_and_refuses_conflicts() {
        for hermes in [true, false] {
            let input = br#"{"model":"existing","tools":{"deny":["exec"]},"agents":{"defaults":{"models":{"other/model":{}}}}}"#;
            let bytes = merge(
                Some(input),
                hermes,
                "http://localhost:9337/v1",
                "auto",
                8192,
            )
            .unwrap();
            let value: Value = if hermes {
                serde_yaml::from_slice(&bytes).unwrap()
            } else {
                serde_json::from_slice(&bytes).unwrap()
            };
            assert_eq!(value["model"], "existing");
            assert_eq!(value["tools"]["deny"], json!(["exec"]));
            assert!(
                merge(
                    Some(&bytes),
                    hermes,
                    "http://localhost:9337/v1",
                    "auto",
                    8192
                )
                .is_ok()
            );
            assert!(
                merge(
                    Some(&bytes),
                    hermes,
                    "http://localhost:9338/v1",
                    "auto",
                    8192
                )
                .is_err()
            );
        }
    }
}
