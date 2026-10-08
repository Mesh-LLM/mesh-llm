use serde_json::{Value, json};
use std::{fs, process::Command};

fn execute(args: &[&str]) {
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "agent-client-config"])
        .args(args)
        .env("OPENAI_API_KEY", "fixture-secret-must-not-appear")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    assert!(result.stderr.is_empty());
}

#[test]
fn pi_config_preserves_literal_model_and_tool_compatibility() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("models with spaces.json");
    let model = "model\"\\\n雪:$literal";
    execute(&[
        "pi",
        "http://127.0.0.1:9337/v1///",
        model,
        output.to_str().unwrap(),
    ]);
    let bytes = fs::read(&output).unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        parsed,
        json!({"providers": {"mesh": {
            "api": "openai-completions", "apiKey": "mesh", "baseUrl": "http://127.0.0.1:9337/v1",
            "compat": {"supportsStore": false, "supportsDeveloperRole": false, "supportsUsageInStreaming": true},
            "models": [{"id": model, "name": model, "contextWindow": 32768, "maxTokens": 4096}]
        }}})
    );
    assert!(
        !String::from_utf8(bytes)
            .unwrap()
            .contains("fixture-secret-must-not-appear")
    );
}

#[test]
fn goose_provider_and_yaml_preserve_literal_model_and_auth_settings() {
    let directory = tempfile::tempdir().unwrap();
    let provider = directory.path().join("provider.json");
    let config = directory.path().join("config.yaml");
    let model = "model\"\\\n雪: # not a yaml mapping";
    execute(&[
        "goose",
        "http://localhost/v1/",
        model,
        provider.to_str().unwrap(),
        config.to_str().unwrap(),
    ]);
    let parsed: Value = serde_json::from_slice(&fs::read(provider).unwrap()).unwrap();
    assert_eq!(
        parsed,
        json!({"name": "mesh", "engine": "openai", "display_name": "mesh-llm",
        "description": "Distributed LLM inference via mesh-llm", "api_key_env": "",
        "base_url": "http://localhost/v1", "models": [{"name": model, "context_limit": 32768}],
        "timeout_seconds": 600, "supports_streaming": true, "requires_auth": false})
    );
    let yaml = fs::read_to_string(config).unwrap();
    let lines = yaml.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0], "GOOSE_PROVIDER: mesh");
    let quoted = lines[1].strip_prefix("GOOSE_MODEL: ").unwrap();
    assert_eq!(serde_json::from_str::<String>(quoted).unwrap(), model);
    assert_eq!(lines[2], "GOOSE_MODE: auto");
    assert_eq!(lines[3], "GOOSE_DISABLE_KEYRING: true");
    assert!(!yaml.contains("fixture-secret-must-not-appear"));
}

#[test]
fn malformed_config_command_fails_without_output_or_files() {
    let directory = tempfile::tempdir().unwrap();
    for args in [vec!["pi", "base"], vec!["other", "base", "model", "path"]] {
        let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(directory.path())
            .args(["automation", "agent-client-config"])
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
}

fn opencode_config(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "agent-client-config", "opencode"])
        .args(args)
        .env("OPENAI_API_KEY", "fixture-secret-must-not-appear")
        .env("PATH", "")
        .output()
        .unwrap()
}

#[test]
fn opencode_config_preserves_provider_model_limits_permissions_and_literal_identity() {
    let model = "model\"\\\n雪:$literal";
    let result = opencode_config(&["http://127.0.0.1:9337/v1///", model]);
    assert!(result.status.success(), "{result:?}");
    assert!(result.stderr.is_empty());
    let document: Value = serde_json::from_slice(&result.stdout).unwrap();
    let permission = json!({"bash":"allow","read":"allow","grep":"allow","glob":"allow","edit":"allow","webfetch":"deny","websearch":"deny","question":"deny","todowrite":"deny"});
    assert_eq!(
        document,
        json!({
            "$schema":"https://opencode.ai/config.json", "permission":permission,
            "provider":{"mesh":{"npm":"@ai-sdk/openai-compatible","name":"mesh-llm",
                "options":{"baseURL":"http://127.0.0.1:9337/v1"},
                "models":{model:{"name":model,"limit":{"context":32768,"output":4096}}}}}
        })
    );
    assert!(
        !String::from_utf8(result.stdout)
            .unwrap()
            .contains("fixture-secret-must-not-appear")
    );
    let external = opencode_config(&[]);
    assert!(external.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&external.stdout).unwrap(),
        json!({"$schema":"https://opencode.ai/config.json","permission":permission})
    );
}

#[test]
fn malformed_opencode_config_inputs_fail_without_config_output() {
    for args in [
        vec!["base"],
        vec!["", "model"],
        vec!["/", "model"],
        vec!["///", "model"],
        vec!["base", ""],
    ] {
        let result = opencode_config(&args);
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
}
