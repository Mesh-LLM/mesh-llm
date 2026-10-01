use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    process::Command,
};
type TestResult = Result<(), Box<dyn std::error::Error>>;

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
}

#[test]
fn split_probe_cli_preserves_dense_recurrent_and_transient_decisions() -> TestResult {
    let root = tempfile::tempdir()?;
    let payloads = command()
        .args(["automation", "split-probe", "prefix-payloads", "fixture"])
        .arg(root.path())
        .arg("unique")
        .output()?;
    assert!(payloads.status.success());
    let first: serde_json::Value =
        serde_json::from_slice(&fs::read(root.path().join("prompt-1.json"))?)?;
    let repeat: serde_json::Value =
        serde_json::from_slice(&fs::read(root.path().join("prompt-2.json"))?)?;
    assert_eq!(first, repeat);
    for (kind, cached, expected) in [
        ("kv-dense", [0, 99, 90, 199, 190, 299], 0),
        ("kv-recurrent", [0, 32, 0, 64, 0, 96], 0),
        ("kv-dense", [0; 6], 75),
    ] {
        for (index, (prompt, cached)) in [100, 100, 200, 200, 300, 300]
            .into_iter()
            .zip(cached)
            .enumerate()
        {
            fs::write(
                root.path().join(format!("response-{}.json", index + 1)),
                serde_json::to_vec(
                    &serde_json::json!({"object":"chat.completion","choices":[{}],"usage":{"prompt_tokens":prompt,"prompt_tokens_details":{"cached_tokens":cached}}}),
                )?,
            )?;
        }
        let result = command()
            .args(["automation", "split-probe", "prefix-verify"])
            .arg(root.path())
            .args(["6", kind])
            .output()?;
        assert_eq!(result.status.code(), Some(expected));
    }
    Ok(())
}

#[test]
fn split_reconciler_cli_verifies_the_persisted_fixture() -> TestResult {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/migration/split_evidence");
    let mut invocation = command();
    invocation.args(["automation", "split-evidence"]);
    for name in [
        "seed-status",
        "seed-stages",
        "seed-models",
        "worker-status",
        "worker-stages",
        "worker-models",
    ] {
        invocation
            .arg(format!("--{name}"))
            .arg(fixture.join(format!("{name}.json")));
    }
    let result = invocation
        .args(["--model-label", "dense", "--verify"])
        .arg(fixture.join("expected-ready.json"))
        .output()?;
    assert!(result.status.success(), "{:?}", result.stderr);
    Ok(())
}

#[test]
fn smoke_inputs_cli_accepts_canonical_and_rejects_wrong_backend() -> TestResult {
    let root = tempfile::tempdir()?;
    fs::create_dir_all(root.path().join("native-runtimes/fixture"))?;
    fs::write(root.path().join("mesh-llm"), b"host")?;
    fs::write(root.path().join("host-imports.json"), b"{}")?;
    fs::write(root.path().join("product-manifest.json"),br#"{"schema_version":2,"contract":"mesh-llm-product-v2","mesh_version":"1.0.0","backend":"cpu","host":{"path":"mesh-llm"},"runtime":{"path":"native-runtimes/fixture"}}"#)?;
    let result = command()
        .args(["automation", "smoke-inputs"])
        .arg(root.path())
        .args(["mesh-llm", "cpu"])
        .output()?;
    assert!(result.status.success(), "{:?}", result.stderr);
    assert_eq!(
        result.stdout,
        b"1.0.0\tcpu\tmesh-llm\tnative-runtimes/fixture\n"
    );
    let rejected = command()
        .args(["automation", "smoke-inputs"])
        .arg(root.path())
        .args(["mesh-llm", "metal"])
        .output()?;
    assert!(!rejected.status.success());
    Ok(())
}

#[test]
fn xet_cli_verifies_isolated_fixture() -> TestResult {
    let root = tempfile::tempdir()?;
    let artifact = root.path().join("fixture.gguf");
    fs::write(&artifact, b"GGUF")?;
    let output = root.path().join("output.json");
    fs::write(
        &output,
        serde_json::to_vec(&serde_json::json!({"path":artifact}))?,
    )?;
    let result = command()
        .args(["automation", "hf-xet-smoke"])
        .arg(output)
        .arg(root.path())
        .output()?;
    assert!(result.status.success(), "{:?}", result.stderr);
    assert!(String::from_utf8(result.stdout)?.contains("(4 bytes)"));
    Ok(())
}

#[test]
fn observation_cli_accepts_stream_and_rejects_incomplete_stream() -> TestResult {
    for (bytes, accepted) in [
        (b"data: {\"id\":\"a\"}\ndata: [DONE]\n".as_slice(), true),
        (b"data: {\"id\":\"a\"}\n".as_slice(), false),
    ] {
        let mut child = command()
            .args(["automation", "smoke-observation", "stream"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()?;
        child
            .stdin
            .take()
            .ok_or("missing stdin")?
            .write_all(bytes)?;
        assert_eq!(child.wait_with_output()?.status.success(), accepted);
    }
    Ok(())
}

fn fixture_audio() -> Vec<u8> {
    let mut bytes = b"RIFF".to_vec();
    bytes.extend(236_u32.to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16_u32.to_le_bytes());
    bytes.extend(1_u16.to_le_bytes());
    bytes.extend(1_u16.to_le_bytes());
    bytes.extend(1000_u32.to_le_bytes());
    bytes.extend(2000_u32.to_le_bytes());
    bytes.extend(2_u16.to_le_bytes());
    bytes.extend(16_u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(200_u32.to_le_bytes());
    for _ in 0..100 {
        bytes.extend(2_i16.to_le_bytes());
    }
    bytes
}

fn serve(listener: TcpListener, replies: Vec<(&'static str, Vec<u8>)>) -> std::io::Result<()> {
    for (content_type, body) in replies {
        let (mut stream, _) = listener.accept()?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let count = stream.read(&mut buffer)?;
            if count == 0 {
                return Err(std::io::Error::other("incomplete request"));
            }
            request.extend_from_slice(&buffer[..count]);
            if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_lowercase()
                            .strip_prefix("content-length:")
                            .and_then(|value| value.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )?;
        stream.write_all(&body)?;
    }
    Ok(())
}

#[test]
fn workload_cli_executes_all_six_classes_against_local_wire_fixtures() -> TestResult {
    let numeric = br#"{"object":"list","model":"fixture","usage":{"prompt_tokens":3},"data":[{"object":"embedding","index":0,"embedding":[1,0]},{"object":"embedding","index":1,"embedding":[1,0]},{"object":"embedding","index":2,"embedding":[0,1]}]}"#;
    let encoded = br#"{"object":"list","model":"fixture","data":[{"object":"embedding","index":0,"embedding":"AACAPwAAAAA="}]}"#;
    let cases = [
        ("embedding",vec![("application/json",numeric.to_vec()),("application/json",encoded.to_vec())]),
        ("rerank",vec![("application/json",br#"{"results":[{"index":0,"relevance_score":2,"document":"related"},{"index":1,"relevance_score":1,"document":"unrelated"}],"usage":{"prompt_tokens":3}}"#.to_vec())]),
        ("encoder_decoder",vec![("application/json",br#"{"choices":[{"text":"Das Haus"}],"usage":{"completion_tokens":2}}"#.to_vec())]),
        ("ocr",vec![("application/json",br#"{"choices":[{"message":{"content":"fixture"}}]}"#.to_vec())]),
        ("speech_synthesis",vec![("audio/wav",fixture_audio())]),
        ("speech_recognition",vec![("application/json",br#"{"text":"fixture"}"#.to_vec())]),
    ];
    let media = tempfile::NamedTempFile::new()?;
    fs::write(media.path(), b"fixture")?;
    for (class, replies) in cases {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}/v1", listener.local_addr()?);
        let server = std::thread::spawn(move || serve(listener, replies));
        let result = command()
            .args([
                "automation",
                "workload-smoke",
                "--base-url",
                &url,
                "--model",
                "fixture",
                "--class",
                class,
                "--media-path",
            ])
            .arg(media.path())
            .output()?;
        server.join().map_err(|_| "fixture server panicked")??;
        assert!(result.status.success(), "{class}: {:?}", result.stderr);
        assert_eq!(
            String::from_utf8(result.stdout)?,
            format!("OpenAI HTTP {class} smoke passed: model=fixture\n")
        );
    }
    Ok(())
}
