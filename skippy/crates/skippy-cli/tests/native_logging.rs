//! Native logging smoke tests use a matching locally built runtime bundle.
#![cfg(feature = "dynamic-native-runtime")]

use std::{path::PathBuf, process::Command};

fn runtime_bundle() -> Option<PathBuf> {
    std::env::var_os("SKIPPY_TEST_RUNTIME_BUNDLE").map(PathBuf::from)
}

#[test]
fn native_failure_replays_diagnostics_as_jsonl_events() {
    let Some(bundle) = runtime_bundle() else {
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let model = directory.path().join("invalid.gguf");
    std::fs::write(&model, b"invalid model data").unwrap();
    let mut config = skippy_config::example_config();
    config["model_path"] = serde_json::json!(model);
    let path = directory.path().join("stage.json");
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_skippy"))
        .arg("--runtime-bundle")
        .arg(bundle)
        .args(["--output", "jsonl", "serve", "--config"])
        .arg(path)
        .args(["--stage-transport", "binary", "--worker-only"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let events: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events.last().unwrap()["type"], "error");
    let diagnostics: String = events
        .iter()
        .filter(|event| event["type"] == "native_log")
        .map(|event| event["data"]["message"].as_str().unwrap())
        .collect();
    assert!(diagnostics.contains("gguf"), "{events:?}");
}

/// Supply a small GGUF via SKIPPY_TEST_MODEL to exercise startup and generation.
#[tokio::test]
async fn successful_serving_is_quiet_unless_debug_is_requested() {
    let Some(bundle) = runtime_bundle() else {
        return;
    };
    let Some(model) = std::env::var_os("SKIPPY_TEST_MODEL") else {
        return;
    };
    for debug in [false, true] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let mut command = Command::new(env!("CARGO_BIN_EXE_skippy"));
        command
            .arg("--runtime-bundle")
            .arg(&bundle)
            .args(["--output", "jsonl", "serve", "--model-path"])
            .arg(&model)
            .args([
                "--bind-addr",
                &address.to_string(),
                "--ctx-size",
                "512",
                "--n-gpu-layers",
                "0",
                "--generation-concurrency",
                "1",
                "--threads",
                "2",
                "--threads-batch",
                "2",
                "--kv-cache-type-k",
                "f16",
                "--kv-cache-type-v",
                "f16",
                "--kv-cache-offload=false",
                "--prefix-cache",
                "off",
                "--native-mtp=false",
                "--compact=false",
                "--temperature",
                "0",
                "--request-max-tokens",
                "4",
            ]);
        if debug {
            command.arg("--debug");
        }
        let mut child = ChildGuard(
            command
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let stdout = read_stream(child.0.stdout.take().unwrap());
        let stderr = read_stream(child.0.stderr.take().unwrap());
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();
        let base = format!("http://{address}/v1");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let model_id = loop {
            if let Ok(response) = client.get(format!("{base}/models")).send().await
                && response.status().is_success()
            {
                let models: serde_json::Value = response.json().await.unwrap();
                break models["data"][0]["id"].as_str().unwrap().to_owned();
            }
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "server exited before readiness"
            );
            assert!(
                std::time::Instant::now() < deadline,
                "server did not become ready"
            );
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        };
        let response = client.post(format!("{base}/chat/completions"))
            .json(&serde_json::json!({"model": model_id, "messages": [{"role": "user", "content": "Hello"}]}))
            .send().await.unwrap();
        assert!(
            response.status().is_success(),
            "{}",
            response.text().await.unwrap()
        );
        let body: serde_json::Value = response.json().await.unwrap();
        assert!(body["usage"]["completion_tokens"].as_u64().unwrap() <= 4);
        child.stop();
        let output = String::from_utf8(stdout.join().unwrap()).unwrap();
        assert!(stderr.join().unwrap().is_empty());
        let events: Vec<serde_json::Value> = output
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let diagnostics: String = events
            .iter()
            .filter(|event| event["type"] == "native_log")
            .map(|event| event["data"]["message"].as_str().unwrap())
            .collect();
        assert_eq!(!diagnostics.is_empty(), debug, "{output}");
        if debug {
            assert!(diagnostics.contains("llama_model_loader"), "{output}");
        }
    }
}

fn read_stream(
    mut stream: impl std::io::Read + Send + 'static,
) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stream.read_to_end(&mut bytes).unwrap();
        bytes
    })
}

struct ChildGuard(std::process::Child);
impl ChildGuard {
    fn stop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.stop();
    }
}
