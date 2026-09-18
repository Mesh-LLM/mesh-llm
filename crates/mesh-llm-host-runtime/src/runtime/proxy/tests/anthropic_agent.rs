// Runs the installed Claude CLI, a real host ingress and a deterministic OpenAI
// upstream. This tests client/tool interoperability, not model quality.
#[tokio::test]
#[ignore = "requires the Claude Code executable; run explicitly in the agent harness lane"]
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
        .env("DISABLE_PROMPT_CACHING", "1")
        .env("CLAUDE_CODE_DISABLE_THINKING", "1")
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
