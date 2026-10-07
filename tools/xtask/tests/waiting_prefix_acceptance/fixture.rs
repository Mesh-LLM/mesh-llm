//! Local executable fixture for retained A/B ownership; no native model execution.
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn option(arguments: &[String], name: &str) -> Result<String> {
    Ok(arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .ok_or("missing fixture option")?[1]
        .clone())
}

fn request(stream: &mut TcpStream) -> Result<(String, Value)> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(2)))?;
    let mut bytes = Vec::new();
    let mut buffer = [0; 4096];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err("incomplete fixture request".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > 1024 * 1024 {
            return Err("fixture request exceeds 1 MiB".into());
        }
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let headers = std::str::from_utf8(&bytes[..end])?;
            let length = headers
                .lines()
                .find_map(|line| {
                    line.split_once(':')
                        .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                        .map(|(_, value)| value.trim().parse::<usize>())
                })
                .transpose()?
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                return Ok((
                    headers.lines().next().ok_or("missing request line")?.into(),
                    if length == 0 {
                        Value::Null
                    } else {
                        serde_json::from_slice(&bytes[end + 4..end + 4 + length])?
                    },
                ));
            }
        }
    }
}

fn response(stream: &mut TcpStream, kind: &str, body: &[u8]) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    Ok(())
}

fn telemetry(tokens: u64, request_id: u64) -> Result<()> {
    let mut stderr = std::io::stderr().lock();
    for (event, attributes) in [
        (
            "stage.openai_generation_summary",
            json!({"skippy.request_id":request_id.to_string(),"skippy.kv.status":"hit","skippy.kv.matched_prefix_tokens":10,
            "llama_stage.completion_token_count":tokens,"skippy.kv.suffix_prefill_tokens":3}),
        ),
        (
            "stage.openai_kv_capacity_decision",
            json!({"skippy.kv.capacity_status":"evicted","skippy.kv.capacity_evicted_tokens":2}),
        ),
        (
            "stage.openai_kv_record_decision",
            json!({"skippy.kv.decision":"proactive_eviction","skippy.kv.proactive_evicted_tokens":1}),
        ),
    ] {
        serde_json::to_writer(&mut stderr, &json!({"event":event,"attributes":attributes}))?;
        stderr.write_all(b"\n")?;
    }
    stderr.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.first().map(String::as_str) != Some("serve-openai") {
        return Err("expected native serve-openai command".into());
    }
    if arguments.len() == 2 && arguments[1] == "--help" {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(b"inert legacy serve-openai help\n")?;
        stdout.flush()?;
        return Ok(());
    }
    if option(&arguments, "--telemetry-level")? != "debug"
        || std::env::var("SKIPPY_TELEMETRY_STDERR")? != "1"
    {
        return Err("missing telemetry configuration".into());
    }
    let config = std::path::PathBuf::from(option(&arguments, "--config")?);
    let stage: Value = serde_json::from_slice(&std::fs::read(&config)?)?;
    let directory = config.parent().ok_or("missing fixture output directory")?;
    if arguments
        .iter()
        .any(|argument| argument == "--metrics-otlp-grpc")
    {
        std::fs::write(
            directory.join("fixture.metrics.json"),
            serde_json::to_vec(&json!({
            "run_id":stage["run_id"],"otlp_grpc":option(&arguments,"--metrics-otlp-grpc")?}))?,
        )?;
    }
    std::fs::write(
        directory.join("fixture.pid"),
        std::process::id().to_string(),
    )?;
    if stage["model_id"] == "early-exit" {
        std::process::exit(7);
    }
    let address = option(&arguments, "--bind-addr")?;
    std::fs::write(directory.join("fixture.address"), &address)?;
    let listener = TcpListener::bind(address)?;
    let mut request_id = 0_u64;
    for incoming in listener.incoming() {
        let mut stream = incoming?;
        let (line, body) = request(&mut stream)?;
        if line.starts_with("GET /v1/models ") {
            let late = (stage["model_id"] == "late-failure"
                && directory
                    .file_name()
                    .is_some_and(|name| name == "round-2-new"))
                || (stage["model_id"] == "manual-late-failure"
                    && directory
                        .file_name()
                        .is_some_and(|name| name == "round-2-old"));
            let model = if stage["model_id"] == "wrong-model" || late {
                json!("other-model")
            } else {
                stage["model_id"].clone()
            };
            response(
                &mut stream,
                "application/json",
                &serde_json::to_vec(&json!({"data":[{"id":model}]}))?,
            )?;
        } else {
            if !line.starts_with("POST /v1/chat/completions ") {
                return Err("unexpected fixture route".into());
            }
            if stage["model_id"] == "stall" {
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
            let tokens = body["max_tokens"]
                .as_u64()
                .ok_or("missing request output tokens")?;
            request_id += 1;
            telemetry(tokens, request_id)?;
            let data = format!(
                "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                json!({"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}),
                json!({"usage":{"prompt_tokens":40,"completion_tokens":tokens,"prompt_tokens_details":{"cached_tokens":30}}})
            );
            response(&mut stream, "text/event-stream", data.as_bytes())?;
        }
    }
    Ok(())
}
