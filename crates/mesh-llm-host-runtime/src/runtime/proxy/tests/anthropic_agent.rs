// Runs the installed Claude CLI, a real host ingress and a deterministic OpenAI
// upstream. This tests client/tool interoperability, not model quality.
#[tokio::test]
async fn claude_cli_executes_read_tool_through_host_ingress() {
    let fixture = tempfile::tempdir().unwrap();
    let fixture = fixture.path().canonicalize().unwrap();
    std::fs::write(fixture.join("marker.txt"), "mesh-anthropic-agent-marker").unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let marker_path = fixture.join("marker.txt").to_string_lossy().into_owned();
    let upstream = tokio::spawn(async move {
        let mut turn = 0;
        while turn < 2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let raw = read_raw_http_request(&mut stream).await;
            let raw = String::from_utf8(raw).unwrap();
            if !raw.starts_with("POST /v1/chat/completions") {
                stream
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .unwrap();
                stream.shutdown().await.unwrap();
                continue;
            }
            let request: serde_json::Value =
                serde_json::from_str(raw.split_once("\r\n\r\n").unwrap().1).unwrap();
            assert_eq!(request["stream"], true);
            assert_eq!(request["stream_options"]["include_usage"], true);
            let (delta, finish) = if turn == 0 {
                assert!(
                    request["tools"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|tool| tool["function"]["name"] == "Read")
                );
                (
                    json!({"role":"assistant","tool_calls":[{"index":0,"id":"call_read_marker","type":"function","function":{"name":"Read","arguments":json!({"file_path":marker_path}).to_string()}}]}),
                    "tool_calls",
                )
            } else {
                assert!(
                    request["messages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|message| message["role"] == "tool"
                            && message.to_string().contains("mesh-anthropic-agent-marker")),
                    "tool result did not round-trip: {request}"
                );
                (
                    json!({"role":"assistant","content":"mesh-agent-success"}),
                    "stop",
                )
            };
            let chunk = json!({"id":"chat-agent","model":"test","choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
            let usage = json!({"id":"chat-agent","model":"test","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}});
            let body = format!("data: {chunk}\n\ndata: {usage}\n\ndata: [DONE]\n\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
            turn += 1;
        }
    });
    let (addr, proxy) = spawn_api_proxy_test_harness(local_targets(&[("test", port)])).await;
    let mut command = tokio::process::Command::new(
        std::env::var("MESH_CLAUDE_BIN").unwrap_or_else(|_| "claude".into()),
    );
    command
        .current_dir(&fixture)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("CLAUDE_CONFIG_DIR", fixture.join("config"))
        .env("ANTHROPIC_BASE_URL", format!("http://{addr}"))
        .env("ANTHROPIC_API_KEY", "local-harness-only")
        .env("DISABLE_AUTOUPDATER", "1")
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .args([
            "--bare",
            "--print",
            "--model",
            "test",
            "--tools",
            "Read",
            "--allowedTools",
            "Read",
            "--strict-mcp-config",
            "--mcp-config",
            r#"{"mcpServers":{}}"#,
            "--no-session-persistence",
            "--output-format",
            "json",
            "--system-prompt",
            "Use Read to read marker.txt, then report its contents.",
            "Read marker.txt.",
        ])
        .arg("--debug-file")
        .arg(fixture.join("debug.log"))
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(60), command.output()).await;
    proxy.abort();
    match output {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(output.status.success(), "Claude failed: {stdout}\n{stderr}");
            assert!(stdout.contains("mesh-agent-success"), "{stdout}\n{stderr}");
            tokio::time::timeout(Duration::from_secs(2), upstream)
                .await
                .unwrap()
                .unwrap();
        }
        error => {
            upstream.abort();
            let debug = std::fs::read_to_string(fixture.join("debug.log")).unwrap_or_default();
            panic!("Claude CLI did not complete: {error:?}\n{debug}");
        }
    }
}

#[cfg(feature = "claude-live-model-integration")]
#[tokio::test]
async fn claude_cli_round_trips_through_host_ingress_and_live_claude_model() {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY is required for the live Claude model gate");
    let live_model = std::env::var("MESH_LIVE_CLAUDE_MODEL")
        .expect("MESH_LIVE_CLAUDE_MODEL is required for the live Claude model gate");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let bridge = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let raw = String::from_utf8(read_raw_http_request(&mut stream).await).unwrap();
        assert!(raw.starts_with("POST /v1/chat/completions"), "{raw}");
        let request: serde_json::Value =
            serde_json::from_str(raw.split_once("\r\n\r\n").unwrap().1).unwrap();
        let mut system = Vec::new();
        let mut messages = Vec::new();
        for message in request["messages"].as_array().unwrap() {
            let role = message["role"].as_str().unwrap_or_default();
            let content = message["content"].as_str().unwrap_or_default();
            if role == "system" {
                system.push(content.to_string());
            } else if role == "user" || role == "assistant" {
                messages.push(json!({"role": role, "content": content}));
            }
        }
        let mut body = json!({
            "model": live_model,
            "max_tokens": 64,
            "messages": messages
        });
        if !system.is_empty() {
            body["system"] = json!(system.join("\n"));
        }
        let response = reqwest::Client::new()
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let response: serde_json::Value = response.json().await.unwrap();
        assert!(status.is_success(), "live Anthropic request failed: {response}");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks
                    .iter()
                    .find(|block| block["type"] == "text")
            })
            .and_then(|block| block["text"].as_str())
            .expect("live Claude response must contain text");
        let prompt_tokens = response["usage"]["input_tokens"].as_u64().unwrap_or(0);
        let completion_tokens = response["usage"]["output_tokens"].as_u64().unwrap_or(0);
        let chunk = json!({
            "id": "chat-live-claude",
            "model": "live-claude",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
        });
        let usage = json!({
            "id": "chat-live-claude",
            "model": "live-claude",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        });
        let response_body = format!("data: {chunk}\n\ndata: {usage}\n\ndata: [DONE]\n\n");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
            response_body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
    });

    let fixture = tempfile::tempdir().unwrap();
    let (addr, proxy) = spawn_api_proxy_test_harness(local_targets(&[("live-claude", port)])).await;
    let mut command = tokio::process::Command::new(
        std::env::var("MESH_CLAUDE_BIN").unwrap_or_else(|_| "claude".into()),
    );
    command
        .current_dir(fixture.path())
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("CLAUDE_CONFIG_DIR", fixture.path().join("config"))
        .env("ANTHROPIC_BASE_URL", format!("http://{addr}"))
        .env("ANTHROPIC_API_KEY", "mesh-live-gate")
        .env("DISABLE_AUTOUPDATER", "1")
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .args([
            "--bare",
            "--print",
            "--model",
            "live-claude",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            r#"{"mcpServers":{}}"#,
            "--no-session-persistence",
            "--output-format",
            "json",
            "Reply with exactly MESH-LIVE-CLAUDE and nothing else.",
        ])
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(120), command.output())
        .await
        .expect("live Claude Code invocation timed out")
        .expect("Claude Code failed to launch");
    proxy.abort();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "Claude failed: {stdout}\n{stderr}");
    assert!(stdout.contains("MESH-LIVE-CLAUDE"), "{stdout}\n{stderr}");
    bridge.await.unwrap();
}
