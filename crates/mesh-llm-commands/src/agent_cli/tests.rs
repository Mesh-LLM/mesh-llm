use super::{
    DEFAULT_MESH_MCP_URL, OPENCODE_DEFAULT_CONTEXT_LIMIT, OPENCODE_INSTALL_HINT,
    OPENCODE_OUTPUT_LIMIT, build_mesh_provider_spec_for_test, build_opencode_launch_spec,
    build_opencode_launch_spec_with_limits, build_pi_provider_config,
    build_pi_provider_config_with_limits, cleanup_mesh_child, configure_opencode_launch_command,
    merge_context_lengths, merge_goose_mcp_config, mesh_mcp_claude_config_json,
    normalize_opencode_host, opencode_missing_binary_guidance, pi_missing_binary_guidance,
    resolve_opencode_config_path_from_home, write_opencode_config_for_test,
    write_pi_config_for_test, write_pi_config_to_path,
};

const LOCAL_OPENCODE_HOST: &str = "127.0.0.1:9337";

fn write_config(
    config_path: &std::path::Path,
    models: &[String],
    host: &str,
) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()
        .expect("test runtime")
        .block_on(write_opencode_config_for_test(config_path, models, host))
}

#[test]
fn opencode_launch_spec_uses_mesh_provider_and_v1_base_url() {
    let spec = build_opencode_launch_spec(
        &[
            "GLM-4.7-Flash-Q4_K_M".to_string(),
            "bartowski/DeepSeek-R1.gguf".to_string(),
        ],
        "GLM-4.7-Flash-Q4_K_M",
        "http://127.0.0.1:9337/v1",
    );
    let config: serde_json::Value =
        serde_json::from_str(&spec.config_content).expect("valid OpenCode config JSON");

    assert_eq!(spec.provider_id, "mesh");
    assert_eq!(spec.api_key_env, "OPENAI_API_KEY");
    assert_eq!(spec.api_key_value, "dummy");
    assert_eq!(config["$schema"], "https://opencode.ai/config.json");
    assert_eq!(
        config["provider"]["mesh"]["npm"],
        "@ai-sdk/openai-compatible"
    );
    assert_eq!(config["provider"]["mesh"]["name"], "mesh-llm");
    assert_eq!(
        config["provider"]["mesh"]["options"]["baseURL"],
        "http://127.0.0.1:9337/v1"
    );
    // apiKey should NOT be in persisted config (handled at runtime via env var)
    assert!(
        config["provider"]["mesh"]["options"]
            .get("apiKey")
            .is_none(),
        "apiKey should not be in options for persisted config"
    );
    assert_eq!(
        config["provider"]["mesh"]["models"]["GLM-4.7-Flash-Q4_K_M"]["name"],
        "GLM-4.7-Flash-Q4_K_M"
    );
    assert_eq!(
        config["provider"]["mesh"]["models"]["bartowski/DeepSeek-R1.gguf"]["name"],
        "bartowski/DeepSeek-R1.gguf"
    );
    assert_eq!(
        config["provider"]["mesh"]["models"]
            .as_object()
            .map(|m| m.len()),
        Some(2)
    );
    assert_eq!(config["mcp"]["mesh"]["type"], "remote");
    assert_eq!(config["mcp"]["mesh"]["enabled"], true);
    assert_eq!(config["mcp"]["mesh"]["url"], DEFAULT_MESH_MCP_URL);
}

#[test]
fn claude_mcp_config_points_at_mesh_mcp_http_endpoint() {
    let config = mesh_mcp_claude_config_json("http://127.0.0.1:3131/mcp").unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&config).unwrap();

    assert_eq!(
        parsed["mcpServers"]["mesh"]["type"],
        serde_json::json!("http")
    );
    assert_eq!(
        parsed["mcpServers"]["mesh"]["url"],
        serde_json::json!("http://127.0.0.1:3131/mcp")
    );
}

#[test]
fn goose_mcp_merge_preserves_existing_extensions() {
    let mut config: serde_yaml::Value = serde_yaml::from_str(
        r#"
extensions:
  developer:
    enabled: true
GOOSE_PROVIDER: mesh
"#,
    )
    .unwrap();
    let path = std::path::Path::new("/tmp/goose/config.yaml");

    merge_goose_mcp_config(&mut config, "http://127.0.0.1:3131/mcp", path).unwrap();
    let extensions = config
        .get("extensions")
        .and_then(serde_yaml::Value::as_mapping)
        .unwrap();

    assert!(extensions.contains_key("developer"));
    let mesh = extensions
        .get("mesh")
        .and_then(serde_yaml::Value::as_mapping)
        .unwrap();
    assert_eq!(
        mesh.get("type").and_then(serde_yaml::Value::as_str),
        Some("streamable_http")
    );
    assert_eq!(
        mesh.get("uri").and_then(serde_yaml::Value::as_str),
        Some("http://127.0.0.1:3131/mcp")
    );
}

#[test]
fn opencode_launch_spec_uses_mesh_prefixed_model() {
    let spec = build_opencode_launch_spec(
        &[
            "GLM-4.7-Flash-Q4_K_M".to_string(),
            "bartowski/DeepSeek-R1.gguf".to_string(),
        ],
        "bartowski/DeepSeek-R1.gguf",
        "http://127.0.0.1:8080/v1",
    );

    assert_eq!(spec.provider_id, "mesh");
    assert_eq!(spec.model, "mesh/bartowski/DeepSeek-R1.gguf");
}

#[test]
fn opencode_launch_command_uses_persisted_config_instead_of_env_blob() {
    let spec = build_opencode_launch_spec(
        &["GLM-4.7-Flash-Q4_K_M".to_string()],
        "GLM-4.7-Flash-Q4_K_M",
        "http://127.0.0.1:9337/v1",
    );
    let mut command = std::process::Command::new("opencode");

    configure_opencode_launch_command(&mut command, &spec);

    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let envs = command
        .get_envs()
        .filter_map(|(key, value)| {
            value.map(|value| {
                (
                    key.to_string_lossy().into_owned(),
                    value.to_string_lossy().into_owned(),
                )
            })
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(args, vec!["-m", "mesh/GLM-4.7-Flash-Q4_K_M"]);
    assert_eq!(
        envs.get("OPENAI_API_KEY").map(String::as_str),
        Some("dummy")
    );
    assert!(
        !envs.contains_key("OPENCODE_CONFIG_CONTENT"),
        "interactive launch should use the persisted opencode config"
    );
}

#[test]
fn opencode_install_hint_mentions_official_install_url() {
    assert!(OPENCODE_INSTALL_HINT.contains("https://opencode.ai/install"));
    assert_eq!(
        OPENCODE_INSTALL_HINT,
        "curl -fsSL https://opencode.ai/install | bash"
    );
}

#[test]
fn opencode_missing_binary_reports_official_install_hint() {
    let spec = build_opencode_launch_spec(
        &[
            "GLM-4.7-Flash-Q4_K_M".to_string(),
            "bartowski/DeepSeek-R1.gguf".to_string(),
        ],
        "GLM-4.7-Flash-Q4_K_M",
        "http://127.0.0.1:9337/v1",
    );
    let lines =
        opencode_missing_binary_guidance("GLM-4.7-Flash-Q4_K_M", LOCAL_OPENCODE_HOST, &spec);

    assert_eq!(lines[0], "opencode not found in PATH");
    assert_eq!(lines[1], OPENCODE_INSTALL_HINT);
    assert_eq!(lines[2], "Then rerun through mesh-llm:");
    assert_eq!(
        lines[3],
        "  mesh-llm opencode --host 127.0.0.1:9337 --model GLM-4.7-Flash-Q4_K_M"
    );
    assert_eq!(
        lines[4],
        "mesh-llm writes the mesh provider into your OpenCode config before launching."
    );
}

#[test]
fn pi_missing_binary_guidance_quotes_model_argument() {
    let lines = pi_missing_binary_guidance("mesh/Qwen's 3.6 27B");

    assert_eq!(lines[0], "pi not found in PATH.");
    assert_eq!(
        lines[1],
        "Install: npm install -g @mariozechner/pi-coding-agent"
    );
    assert_eq!(lines[2], "Or run manually:");
    assert_eq!(lines[3], "  pi --model 'mesh/Qwen'\"'\"'s 3.6 27B'");
}

#[test]
fn test_write_creates_new_config_file() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    assert!(!config_path.exists());

    let models = vec!["qwen2.5-3b".to_string(), "glm-4.7-flash".to_string()];

    let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

    assert!(
        result.is_ok(),
        "write_opencode_config should succeed on new file"
    );
    assert!(config_path.exists(), "config file should be created");

    let content = std::fs::read_to_string(&config_path).expect("failed to read config");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

    assert_eq!(parsed["$schema"], "https://opencode.ai/config.json");
    assert!(parsed["provider"]["mesh"].is_object());
    assert_eq!(parsed["mcp"]["mesh"]["type"], "remote");
    assert_eq!(parsed["mcp"]["mesh"]["url"], "http://127.0.0.1:3131/mcp");
    assert_eq!(parsed["mcp"]["mesh"]["enabled"], true);
}

#[test]
fn test_write_merges_with_existing_providers() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    let existing_config = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "anthropic": {
                "npm": "@ai-sdk/anthropic",
                "name": "Anthropic",
                "options": {
                    "apiKey": "{env:ANTHROPIC_API_KEY}"
                },
                "models": {
                    "claude-3-sonnet": { "name": "claude-3-sonnet" }
                }
            },
            "openai": {
                "npm": "@ai-sdk/openai",
                "name": "OpenAI",
                "options": {
                    "apiKey": "{env:OPENAI_API_KEY}"
                },
                "models": {
                    "gpt-4o": { "name": "gpt-4o" }
                }
            }
        }
    });

    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&existing_config).unwrap(),
    )
    .expect("failed to write initial config");

    let models = vec!["qwen2.5-3b".to_string()];

    let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

    assert!(result.is_ok(), "merge should succeed");

    let content = std::fs::read_to_string(&config_path).expect("failed to read config");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

    assert_eq!(parsed["$schema"], "https://opencode.ai/config.json");
    assert!(
        parsed["provider"]["anthropic"].is_object(),
        "anthropic provider should be preserved"
    );
    assert!(
        parsed["provider"]["openai"].is_object(),
        "openai provider should be preserved"
    );
    assert!(
        parsed["provider"]["mesh"].is_object(),
        "mesh provider should be added"
    );
    assert_eq!(
        parsed["provider"]["anthropic"]["name"], "Anthropic",
        "anthropic name should be unchanged"
    );
}

#[test]
fn test_write_overwrites_mesh_provider() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    let existing_config = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "mesh": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "mesh-llm-old",
                "options": {
                    "baseURL": "http://127.0.0.1:8080/v1",
                    "apiKey": "{env:OPENAI_API_KEY}"
                },
                "models": {
                    "old-model": { "name": "old-model" }
                }
            }
        }
    });

    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&existing_config).unwrap(),
    )
    .expect("failed to write initial config");

    let models = vec!["qwen2.5-3b".to_string(), "deepseek-r1".to_string()];

    let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

    assert!(result.is_ok(), "overwrite should succeed");

    let content = std::fs::read_to_string(&config_path).expect("failed to read config");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

    assert_eq!(
        parsed["provider"]["mesh"]["name"], "mesh-llm",
        "mesh name should be updated"
    );
    assert_eq!(
        parsed["provider"]["mesh"]["options"]["baseURL"], "http://127.0.0.1:9337/v1",
        "baseURL should be updated to new port"
    );
    assert!(
        parsed["provider"]["mesh"]["models"]["old-model"].is_null(),
        "old model should be removed"
    );
    assert_eq!(
        parsed["provider"]["mesh"]["models"]["qwen2.5-3b"]["name"], "qwen2.5-3b",
        "new model should be present"
    );
    assert_eq!(
        parsed["provider"]["mesh"]["models"]["deepseek-r1"]["name"], "deepseek-r1",
        "second new model should be present"
    );
}

#[test]
fn test_build_mesh_provider_spec_generates_correct_format() {
    let models = vec![
        "Qwen2.5-3B-Q4_K_M".to_string(),
        "bartowski/GLM-4.7-Flash-Q4_K_M".to_string(),
    ];
    let spec = build_mesh_provider_spec_for_test(&models, LOCAL_OPENCODE_HOST);

    assert!(spec.is_object(), "should return a JSON object");

    assert_eq!(
        spec["npm"], "@ai-sdk/openai-compatible",
        "npm package should match opencode format"
    );
    assert_eq!(spec["name"], "mesh-llm", "name field should be mesh-llm");
    assert!(spec["options"].is_object(), "options should be an object");
    assert_eq!(
        spec["options"]["baseURL"], "http://127.0.0.1:9337/v1",
        "baseURL should include /v1 suffix and correct port"
    );
    // apiKey is not persisted in config (handled at runtime via env var)
    assert!(
        spec["options"].get("apiKey").is_none(),
        "apiKey should not be in options for persisted config"
    );
    assert!(spec["models"].is_object(), "models should be an object");
    assert_eq!(
        spec["models"]["Qwen2.5-3B-Q4_K_M"]["name"], "Qwen2.5-3B-Q4_K_M",
        "model name should match input"
    );
    assert_eq!(
        spec["models"]["bartowski/GLM-4.7-Flash-Q4_K_M"]["name"], "bartowski/GLM-4.7-Flash-Q4_K_M",
        "model with slash in name should work correctly"
    );
}

#[test]
fn test_write_handles_empty_models_list() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    let models: Vec<String> = vec![];

    let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

    assert!(result.is_ok(), "should succeed with empty models list");
    assert!(config_path.exists(), "config file should still be created");

    let content = std::fs::read_to_string(&config_path).expect("failed to read config");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

    assert!(
        parsed["provider"]["mesh"]["models"].is_object(),
        "models field should exist even when empty"
    );
    assert_eq!(
        parsed["provider"]["mesh"]["models"]
            .as_object()
            .map(|m| m.len())
            .unwrap_or(0),
        0,
        "models object should be empty"
    );
}

#[test]
fn test_write_handles_special_characters_in_model_names() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    let models = vec![
        "model-with-dashes".to_string(),
        "model_with_underscores".to_string(),
        "ModelWithCamelCase".to_string(),
        "bartowski/model-v2.5-Q4_K_M.gguf".to_string(),
        "1-model-starting-with-number".to_string(),
    ];

    let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

    assert!(
        result.is_ok(),
        "should succeed with special character model names"
    );

    let content = std::fs::read_to_string(&config_path).expect("failed to read config");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

    for model in &models {
        assert!(
            !parsed["provider"]["mesh"]["models"][model].is_null(),
            "model '{}' should be present in config",
            model
        );
        assert_eq!(
            parsed["provider"]["mesh"]["models"][model]["name"], *model,
            "model name should match exactly"
        );
    }
}

#[test]
fn test_write_preserves_existing_file_schema() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    let existing_config = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "$customField": "preserve-me",
        "provider": {}
    });

    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&existing_config).unwrap(),
    )
    .expect("failed to write initial config");

    let models = vec!["qwen".to_string()];

    let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

    assert!(result.is_ok());

    let content = std::fs::read_to_string(&config_path).expect("failed to read config");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

    assert_eq!(
        parsed["$schema"], "https://opencode.ai/config.json",
        "schema should be preserved"
    );
    assert_eq!(
        parsed["$customField"], "preserve-me",
        "custom fields at root level should be preserved"
    );
}

#[test]
fn pi_provider_config_lists_all_mesh_models_with_models_key_last() {
    let models = vec!["Qwen 3.6 27B".to_string(), "Qwen 3.5 4B".to_string()];
    let provider = build_pi_provider_config(&models, "http://localhost:9337/v1");

    assert_eq!(provider["api"], "openai-completions");
    assert_eq!(provider["apiKey"], "mesh");
    assert_eq!(provider["baseUrl"], "http://localhost:9337/v1");
    assert_eq!(provider["compat"]["supportsStore"], false);
    assert_eq!(provider["compat"]["supportsDeveloperRole"], false);
    assert_eq!(provider["compat"]["supportsUsageInStreaming"], true);
    assert_eq!(provider["models"].as_array().map(Vec::len), Some(2));
    assert_eq!(provider["models"][0]["id"], "Qwen 3.6 27B");
    assert_eq!(provider["models"][0]["name"], "Qwen 3.6 27B");
    assert_eq!(provider["models"][1]["id"], "Qwen 3.5 4B");
    assert_eq!(provider["models"][1]["name"], "Qwen 3.5 4B");

    let key_order: Vec<&str> = provider
        .as_object()
        .expect("provider is object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(key_order.last(), Some(&"models"));
}

#[test]
fn pi_provider_config_includes_context_window_and_max_tokens_when_known() {
    let models = vec![
        "Qwen3.6-27B-UD-Q4_K_XL".to_string(),
        "Qwen3.5-4B-UD-Q4_K_XL".to_string(),
        "Unknown-Model".to_string(),
    ];
    let mut context_lengths = std::collections::HashMap::new();
    context_lengths.insert("Qwen3.6-27B-UD-Q4_K_XL".to_string(), Some(262144));
    context_lengths.insert("Qwen3.5-4B-UD-Q4_K_XL".to_string(), Some(65536));
    context_lengths.insert("Unknown-Model".to_string(), None);

    let provider = build_pi_provider_config_with_limits(
        &models,
        "http://carrack.patio51.com:9337/v1",
        &context_lengths,
    );

    assert_eq!(provider["models"][0]["contextWindow"], 262144);
    assert_eq!(provider["models"][0]["maxTokens"], 262144);
    assert_eq!(provider["models"][1]["contextWindow"], 65536);
    assert_eq!(provider["models"][1]["maxTokens"], 65536);
    assert!(
        provider["models"][2]["contextWindow"].is_null(),
        "model with unknown context_length should omit contextWindow"
    );
    assert!(
        provider["models"][2]["maxTokens"].is_null(),
        "model with unknown context_length should omit maxTokens"
    );

    let key_order: Vec<&str> = provider
        .as_object()
        .expect("provider is object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(key_order.last(), Some(&"models"));
}

#[test]
fn pi_write_creates_provider_and_preserves_other_providers() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("models.json");
    let existing_config = serde_json::json!({
        "providers": {
            "anthropic": {
                "api": "anthropic",
                "apiKey": "preserve-me",
                "models": [{ "id": "claude" }]
            }
        }
    });
    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&existing_config).unwrap(),
    )
    .expect("failed to write initial config");

    let models = vec!["Qwen 3.6 27B".to_string(), "Qwen 3.5 4B".to_string()];
    write_pi_config_to_path(&config_path, &models, "http://localhost:9337/v1")
        .expect("pi write should succeed");

    let content = std::fs::read_to_string(&config_path).expect("failed to read config");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

    assert_eq!(parsed["providers"]["anthropic"]["apiKey"], "preserve-me");
    assert_eq!(parsed["providers"]["mesh"]["api"], "openai-completions");
    assert_eq!(
        parsed["providers"]["mesh"]["baseUrl"],
        "http://localhost:9337/v1"
    );
    assert_eq!(
        parsed["providers"]["mesh"]["models"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert!(
        !parsed["providers"]["mesh"]["models"]
            .as_array()
            .expect("models is array")
            .iter()
            .any(|model| model["id"] == "auto"),
        "pi --write should list mesh models, not add a synthetic auto model"
    );
}

#[test]
fn pi_write_uses_normalized_remote_host_as_base_url() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("models.json");
    let models = vec![
        "Qwen3.5-4B-UD-Q4_K_XL".to_string(),
        "Qwen3.6-27B-UD-Q4_K_XL".to_string(),
    ];

    write_pi_config_for_test(
        &config_path,
        &models,
        "https://carrack.patio51.com:9443/custom/path",
    )
    .expect("pi write should succeed with a full remote URL");

    let content = std::fs::read_to_string(&config_path).expect("failed to read config");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

    assert_eq!(
        parsed["providers"]["mesh"]["baseUrl"],
        "https://carrack.patio51.com:9443/v1"
    );
    assert_eq!(parsed["providers"]["mesh"]["models"][0]["id"], models[0]);
    assert_eq!(parsed["providers"]["mesh"]["models"][1]["id"], models[1]);

    let key_order: Vec<&str> = parsed["providers"]["mesh"]
        .as_object()
        .expect("provider is object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(key_order.last(), Some(&"models"));
}

#[test]
fn pi_write_rejects_invalid_json_without_clobbering_config() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("models.json");
    std::fs::write(&config_path, "not-json").expect("failed to write invalid config");

    let err = write_pi_config_to_path(
        &config_path,
        &["Qwen 3.6 27B".to_string()],
        "http://localhost:9337/v1",
    )
    .expect_err("invalid JSON should fail");

    assert!(err.to_string().contains("Failed to parse"));
    assert_eq!(
        std::fs::read_to_string(&config_path).expect("failed to reread config"),
        "not-json"
    );
}

#[test]
fn pi_write_rejects_non_object_providers() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("models.json");
    std::fs::write(&config_path, r#"{"providers": []}"#)
        .expect("failed to write invalid providers config");

    let err = write_pi_config_to_path(
        &config_path,
        &["Qwen 3.6 27B".to_string()],
        "http://localhost:9337/v1",
    )
    .expect_err("array providers should fail");

    assert!(err.to_string().contains("providers"));
    assert!(err.to_string().contains("object"));
}

#[test]
fn opencode_write_rejects_non_object_provider() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"provider": []}"#)
        .expect("failed to write invalid provider config");

    let result = write_config(&config_path, &["qwen".to_string()], LOCAL_OPENCODE_HOST);

    let err = result.expect_err("array provider should fail");
    assert!(err.to_string().contains("provider"));
    assert!(err.to_string().contains("object"));
}

#[test]
fn test_build_opencode_launch_spec_with_limits_includes_context_length() {
    let mut context_lengths = std::collections::HashMap::new();
    context_lengths.insert("Qwen3.5-27B".to_string(), Some(262144));
    context_lengths.insert("Gemma-7B".to_string(), Some(8192));
    context_lengths.insert("Llama-3B".to_string(), None);

    let models = vec![
        "Qwen3.5-27B".to_string(),
        "Gemma-7B".to_string(),
        "Llama-3B".to_string(),
    ];

    let spec = build_opencode_launch_spec_with_limits(
        &models,
        "Qwen3.5-27B",
        "http://127.0.0.1:9337/v1",
        DEFAULT_MESH_MCP_URL,
        &context_lengths,
    );
    let config: serde_json::Value = serde_json::from_str(&spec.config_content).expect("valid JSON");

    assert_eq!(
        config["provider"]["mesh"]["models"]["Qwen3.5-27B"]["name"],
        "Qwen3.5-27B"
    );
    assert_eq!(
        config["provider"]["mesh"]["models"]["Qwen3.5-27B"]["limit"]["context"],
        262144
    );
    assert_eq!(
        config["provider"]["mesh"]["models"]["Qwen3.5-27B"]["limit"]["output"],
        OPENCODE_OUTPUT_LIMIT
    );

    assert_eq!(
        config["provider"]["mesh"]["models"]["Gemma-7B"]["name"],
        "Gemma-7B"
    );
    assert_eq!(
        config["provider"]["mesh"]["models"]["Gemma-7B"]["limit"]["context"],
        8192
    );
    assert_eq!(
        config["provider"]["mesh"]["models"]["Gemma-7B"]["limit"]["output"],
        OPENCODE_OUTPUT_LIMIT
    );

    assert_eq!(
        config["provider"]["mesh"]["models"]["Llama-3B"]["name"],
        "Llama-3B"
    );
    assert_eq!(
        config["provider"]["mesh"]["models"]["Llama-3B"]["limit"]["context"],
        OPENCODE_DEFAULT_CONTEXT_LIMIT
    );
    assert_eq!(
        config["provider"]["mesh"]["models"]["Llama-3B"]["limit"]["output"],
        OPENCODE_OUTPUT_LIMIT
    );
}

#[test]
fn opencode_host_normalization_defaults_bare_host_ports_and_management_lookup() {
    let target = normalize_opencode_host("mesh.example.com").expect("valid host");

    assert_eq!(target.api_base_url, "http://mesh.example.com:9337/v1");
    assert_eq!(
        target.api_models_url,
        "http://mesh.example.com:9337/v1/models"
    );
    assert_eq!(
        target.management_models_url,
        "http://mesh.example.com:3131/api/models"
    );
    assert_eq!(target.mcp_url, "http://mesh.example.com:3131/mcp");
    assert!(!target.auto_start_local_mesh);
}

#[test]
fn opencode_host_normalization_treats_bare_port_as_loopback_api_port() {
    let target = normalize_opencode_host("9443").expect("valid port-only host");

    assert_eq!(target.api_base_url, "http://127.0.0.1:9443/v1");
    assert_eq!(target.api_models_url, "http://127.0.0.1:9443/v1/models");
    assert_eq!(
        target.management_models_url,
        "http://127.0.0.1:3131/api/models"
    );
    assert_eq!(target.mcp_url, "http://127.0.0.1:3131/mcp");
    assert!(target.auto_start_local_mesh);
    assert_eq!(target.local_port, Some(9443));
}

#[test]
fn opencode_host_normalization_defaults_scheme_loopback_to_mesh_ports() {
    let localhost = normalize_opencode_host("http://localhost").expect("valid localhost URL");
    let loopback = normalize_opencode_host("http://127.0.0.1").expect("valid loopback URL");

    assert_eq!(localhost.api_base_url, "http://localhost:9337/v1");
    assert_eq!(localhost.api_models_url, "http://localhost:9337/v1/models");
    assert_eq!(
        localhost.management_models_url,
        "http://localhost:3131/api/models"
    );
    assert!(localhost.auto_start_local_mesh);
    assert_eq!(localhost.local_port, Some(9337));

    assert_eq!(loopback.api_base_url, "http://127.0.0.1:9337/v1");
    assert_eq!(
        loopback.management_models_url,
        "http://127.0.0.1:3131/api/models"
    );
    assert!(loopback.auto_start_local_mesh);
    assert_eq!(loopback.local_port, Some(9337));
}

#[test]
fn opencode_host_normalization_uses_management_port_for_explicit_loopback_api_urls() {
    let localhost = normalize_opencode_host("http://localhost:9337").expect("valid localhost URL");
    let loopback = normalize_opencode_host("http://127.0.0.1:9443").expect("valid loopback URL");

    assert_eq!(localhost.api_base_url, "http://localhost:9337/v1");
    assert_eq!(
        localhost.management_models_url,
        "http://localhost:3131/api/models"
    );
    assert!(localhost.auto_start_local_mesh);
    assert_eq!(localhost.local_port, Some(9337));

    assert_eq!(loopback.api_base_url, "http://127.0.0.1:9443/v1");
    assert_eq!(
        loopback.management_models_url,
        "http://127.0.0.1:3131/api/models"
    );
    assert!(loopback.auto_start_local_mesh);
    assert_eq!(loopback.local_port, Some(9443));
}

#[test]
fn opencode_host_validation_mentions_opencode_host() {
    let err = normalize_opencode_host("   ").expect_err("empty host should fail");

    assert!(err.to_string().contains("OpenCode host"));
    assert!(!err.to_string().contains("mesh host"));
}

#[test]
fn opencode_host_normalization_does_not_auto_start_https_loopback() {
    let target = normalize_opencode_host("https://localhost:9337").expect("valid HTTPS URL");

    assert_eq!(target.api_base_url, "https://localhost:9337/v1");
    assert_eq!(
        target.management_models_url,
        "https://localhost:9337/api/models"
    );
    assert!(!target.auto_start_local_mesh);
    assert_eq!(target.local_port, Some(9337));
}

#[test]
fn merge_context_lengths_uses_runtime_process_when_api_models_missing() {
    let models = serde_json::json!({
        "mesh_models": [
            { "name": "ModelA", "context_length": null },
            { "name": "ModelB", "context_length": 8192 },
        ]
    });
    let processes = serde_json::json!({
        "processes": [
            { "name": "ModelA", "context_length": 16384 },
            { "name": "ModelB", "context_length": null },
            { "name": "ModelC", "context_length": 32768 },
        ]
    });

    let result = merge_context_lengths(&models, &processes);

    assert_eq!(result.get("ModelA"), Some(&Some(16384)));
    assert_eq!(result.get("ModelB"), Some(&Some(8192)));
    assert_eq!(result.get("ModelC"), Some(&Some(32768)));
}

#[test]
fn merge_context_lengths_api_models_only() {
    let models = serde_json::json!({
        "mesh_models": [
            { "name": "ModelA", "context_length": 4096 },
            { "name": "ModelB", "context_length": 8192 },
        ]
    });
    let processes = serde_json::json!({ "processes": [] });

    let result = merge_context_lengths(&models, &processes);

    assert_eq!(result.get("ModelA"), Some(&Some(4096)));
    assert_eq!(result.get("ModelB"), Some(&Some(8192)));
    assert_eq!(result.get("ModelC"), None);
}

#[test]
fn merge_context_lengths_runtime_process_only() {
    let models = serde_json::json!({ "mesh_models": [] });
    let processes = serde_json::json!({
        "processes": [
            { "name": "ModelX", "context_length": 65536 },
        ]
    });

    let result = merge_context_lengths(&models, &processes);

    assert_eq!(result.get("ModelX"), Some(&Some(65536)));
}

#[test]
fn merge_context_lengths_runtime_process_trumps_api_models() {
    let models = serde_json::json!({
        "mesh_models": [
            { "name": "Qwen3-8B", "context_length": 32768 },
        ]
    });
    let processes = serde_json::json!({
        "processes": [
            { "name": "Qwen3-8B", "context_length": 16384 },
        ]
    });

    let result = merge_context_lengths(&models, &processes);

    assert_eq!(result.get("Qwen3-8B"), Some(&Some(16384)));
}

#[test]
fn merge_context_lengths_falls_back_to_metadata_when_runtime_null() {
    let models = serde_json::json!({
        "mesh_models": [
            { "name": "ModelA", "context_length": 4096 },
        ]
    });
    let processes = serde_json::json!({
        "processes": [
            { "name": "ModelA", "context_length": null },
        ]
    });

    let result = merge_context_lengths(&models, &processes);

    assert_eq!(result.get("ModelA"), Some(&Some(4096)));
}

#[test]
fn context_length_lookup_is_best_effort_and_returns_empty_map_on_failure() {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(50))
        .build()
        .expect("client should build");

    let context_lengths = tokio::runtime::Runtime::new()
        .expect("test runtime")
        .block_on(super::fetch_model_context_lengths(
            &client,
            "http://127.0.0.1:9/api/models",
        ));

    assert!(context_lengths.is_empty());
}

#[test]
fn opencode_host_normalization_preserves_full_url_origin() {
    let target =
        normalize_opencode_host("https://mesh.example.com:9443/custom/path").expect("valid URL");

    assert_eq!(target.api_base_url, "https://mesh.example.com:9443/v1");
    assert_eq!(
        target.management_models_url,
        "https://mesh.example.com:9443/api/models"
    );
    assert!(!target.auto_start_local_mesh);
}

#[test]
fn opencode_host_normalization_marks_loopback_targets_for_auto_start() {
    let localhost = normalize_opencode_host("127.0.0.1").expect("valid loopback host");
    let remote = normalize_opencode_host("https://mesh.example.com").expect("valid host");

    assert!(localhost.auto_start_local_mesh);
    assert_eq!(localhost.local_port, Some(9337));
    assert!(!remote.auto_start_local_mesh);
}

#[test]
fn resolve_opencode_config_path_accepts_jsonc_only_configs() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_dir = temp_dir.path().join(".config").join("opencode");
    std::fs::create_dir_all(&config_dir).expect("failed to create config dir");
    let jsonc_path = config_dir.join("opencode.jsonc");
    std::fs::write(&jsonc_path, "{/* comments */}").expect("failed to write jsonc config");

    let resolved =
        resolve_opencode_config_path_from_home(temp_dir.path()).expect("jsonc should resolve");

    assert_eq!(resolved, jsonc_path);
}

#[test]
fn opencode_write_accepts_jsonc_config_with_comments_and_trailing_commas() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = temp_dir.path().join("opencode.jsonc");
    std::fs::write(
        &config_path,
        r#"{
              // Existing OpenCode setting
              "$schema": "https://opencode.ai/config.json",
              "theme": "opencode",
            }"#,
    )
    .expect("failed to write jsonc config");

    write_config(
        &config_path,
        &["Qwen3.5-27B".to_string()],
        LOCAL_OPENCODE_HOST,
    )
    .expect("jsonc config should be updated");

    let content = std::fs::read_to_string(&config_path).expect("failed to read config");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("written JSON");
    assert_eq!(parsed["theme"], "opencode");
    assert!(parsed["provider"]["mesh"].is_object());
}

#[test]
fn cleanup_mesh_child_stops_spawned_process() {
    let mut child = Some(
        std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("failed to spawn test child"),
    );

    cleanup_mesh_child(&mut child);

    assert!(child.is_some());
    let status = child
        .as_mut()
        .expect("child handle retained")
        .try_wait()
        .expect("wait should succeed");
    assert!(status.is_some(), "child should be exited after cleanup");
}
