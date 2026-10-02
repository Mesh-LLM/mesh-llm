//! Isolated client configuration serialization; no client execution or credentials.
use crate::command::DynResult;
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Serialize)]
struct PiConfig<'a> {
    providers: BTreeMap<&'static str, PiProvider<'a>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PiProvider<'a> {
    api: &'static str,
    api_key: &'static str,
    base_url: &'a str,
    compat: PiCompatibility,
    models: [PiModel<'a>; 1],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PiCompatibility {
    supports_store: bool,
    supports_developer_role: bool,
    supports_usage_in_streaming: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PiModel<'a> {
    id: &'a str,
    name: &'a str,
    context_window: u32,
    max_tokens: u32,
}

#[derive(Serialize)]
struct GooseProvider<'a> {
    name: &'static str,
    engine: &'static str,
    display_name: &'static str,
    description: &'static str,
    api_key_env: &'static str,
    base_url: &'a str,
    models: [GooseModel<'a>; 1],
    timeout_seconds: u32,
    supports_streaming: bool,
    requires_auth: bool,
}

#[derive(Serialize)]
struct GooseModel<'a> {
    name: &'a str,
    context_limit: u32,
}

fn write_json(path: &Path, config: &impl Serialize) -> DynResult<()> {
    let mut bytes = serde_json::to_vec_pretty(config)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}

fn pi(base: &str, model: &str, output: &Path) -> DynResult<()> {
    let provider = PiProvider {
        api: "openai-completions",
        api_key: "mesh",
        base_url: base.trim_end_matches('/'),
        compat: PiCompatibility {
            supports_store: false,
            supports_developer_role: false,
            supports_usage_in_streaming: true,
        },
        models: [PiModel {
            id: model,
            name: model,
            context_window: 32768,
            max_tokens: 4096,
        }],
    };
    write_json(
        output,
        &PiConfig {
            providers: BTreeMap::from([("mesh", provider)]),
        },
    )
}

fn goose(base: &str, model: &str, provider: &Path, config: &Path) -> DynResult<()> {
    let settings = GooseProvider {
        name: "mesh",
        engine: "openai",
        display_name: "mesh-llm",
        description: "Distributed LLM inference via mesh-llm",
        api_key_env: "",
        base_url: base.trim_end_matches('/'),
        models: [GooseModel {
            name: model,
            context_limit: 32768,
        }],
        timeout_seconds: 600,
        supports_streaming: true,
        requires_auth: false,
    };
    let yaml = format!(
        "GOOSE_PROVIDER: mesh\nGOOSE_MODEL: {}\nGOOSE_MODE: auto\nGOOSE_DISABLE_KEYRING: true\n",
        serde_json::to_string(model)?
    );
    write_json(provider, &settings)?;
    fs::write(config, yaml)?;
    Ok(())
}

#[derive(Serialize)]
struct OpenCodeConfig<'a> {
    #[serde(rename = "$schema")]
    schema: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<BTreeMap<&'static str, OpenCodeProvider<'a>>>,
    permission: BTreeMap<&'static str, &'static str>,
}

#[derive(Serialize)]
struct OpenCodeProvider<'a> {
    npm: &'static str,
    name: &'static str,
    options: OpenCodeOptions<'a>,
    models: BTreeMap<&'a str, OpenCodeModel<'a>>,
}

#[derive(Serialize)]
struct OpenCodeOptions<'a> {
    #[serde(rename = "baseURL")]
    base_url: &'a str,
}

#[derive(Serialize)]
struct OpenCodeModel<'a> {
    name: &'a str,
    limit: OpenCodeLimit,
}

#[derive(Serialize)]
struct OpenCodeLimit {
    context: u32,
    output: u32,
}

fn opencode(provider: Option<(&str, &str)>) -> DynResult<()> {
    let provider = provider
        .map(|(base, model)| -> DynResult<_> {
            let normalized_base = base.trim_end_matches('/');
            if normalized_base.is_empty()
                || model.is_empty()
                || base.len() > 65536
                || model.len() > 65536
            {
                return Err(
                    "OpenCode base URL and model must be nonempty and at most 64 KiB".into(),
                );
            }
            Ok(BTreeMap::from([(
                "mesh",
                OpenCodeProvider {
                    npm: "@ai-sdk/openai-compatible",
                    name: "mesh-llm",
                    options: OpenCodeOptions {
                        base_url: normalized_base,
                    },
                    models: BTreeMap::from([(
                        model,
                        OpenCodeModel {
                            name: model,
                            limit: OpenCodeLimit {
                                context: 32768,
                                output: 4096,
                            },
                        },
                    )]),
                },
            )]))
        })
        .transpose()?;
    let config = OpenCodeConfig {
        schema: "https://opencode.ai/config.json",
        provider,
        permission: BTreeMap::from([
            ("bash", "allow"),
            ("read", "allow"),
            ("grep", "allow"),
            ("glob", "allow"),
            ("edit", "allow"),
            ("webfetch", "deny"),
            ("websearch", "deny"),
            ("question", "deny"),
            ("todowrite", "deny"),
        ]),
    };
    crate::repository::check_report::CheckReport::success(format!(
        "{}\n",
        serde_json::to_string(&config)?
    ))
    .emit()
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [client] if client == "opencode" => opencode(None),
        [client, base, model] if client == "opencode" => opencode(Some((base, model))),
        [client, base, model, path] if client == "pi" => pi(base, model, Path::new(path)),
        [client, base, model, provider, config] if client == "goose" =>
            goose(base, model, Path::new(provider), Path::new(config)),
        _ => Err("usage: automation agent-client-config {pi BASE MODEL JSON | goose BASE MODEL PROVIDER_JSON CONFIG_YAML | opencode [BASE MODEL]}".into()),
    }
}
