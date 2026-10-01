use std::{
    io::{Read, Write},
    net::TcpStream,
    time::Duration,
};

pub struct Context<'a> {
    pub api: u16,
    pub scenario: &'a str,
    pub model: &'a str,
    pub headless: bool,
}

pub fn serve(
    mut stream: TcpStream,
    chats: &mut u32,
    context: Context<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let mut bytes = Vec::new();
    let mut buffer = [0; 4096];
    let header_end = loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 || bytes.len() > 16384 {
            return Err("incomplete request".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(position) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let header = std::str::from_utf8(&bytes[..header_end])?.to_owned();
    let length = header
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(str::trim)
                .map(str::to_owned)
        })
        .map_or(Ok(0), |value| value.parse::<usize>())?;
    while bytes.len() < header_end + length {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err("short request body".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let mut code = 200;
    let body = if header.starts_with("GET /api/status ") {
        serde_json::json!({"api_port":context.api,"local_instances":[{"pid": if context.scenario == "ownership" { 1 } else { std::process::id() },"is_self":true}],
            "token":"sdk-fixture-invite", "llama_ready": context.scenario != "timeout", "release_attestation":{"status": if context.scenario == "attestation" { "invalid" } else { "missing" }}}).to_string()
    } else if header.starts_with("GET /v1/models ") {
        if context.headless && context.scenario == "headless" {
            code = 500;
        }
        let id = header
            .lines()
            .find_map(|line| line.strip_prefix("x-request-id: "));
        if let Some(id) = id {
            println!(
                "{}",
                serde_json::json!({"request_id":id,"source":"direct_http","route":"models","method":"GET","request_kind":"model_listing","status_code":200,"event":"request_completed","outcome":"completed"})
            );
            std::io::stdout().flush()?;
        }
        serde_json::json!({"data":[{"id":"chosen"}]}).to_string()
    } else if header.starts_with("POST /v1/chat/completions ") {
        *chats += 1;
        let payload: serde_json::Value = serde_json::from_slice(&bytes[header_end..])?;
        let stream_request = payload.get("stream") == Some(&serde_json::Value::Bool(true));
        let expected = if *chats == 3 { "auto" } else { "chosen" };
        if context.scenario == "cancel" {
            let root = std::path::PathBuf::from(
                std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR").ok_or("native")?,
            );
            std::fs::write(root.join("chat.armed"), b"armed")?;
            while !super::signals::stopped() {
                std::thread::park_timeout(Duration::from_millis(1));
            }
            return Ok(());
        }
        if payload.get("model").and_then(serde_json::Value::as_str) != Some(expected)
            || payload
                .get("max_tokens")
                .and_then(serde_json::Value::as_u64)
                != Some(4)
            || context.headless
            || context.model.is_empty()
        {
            return Err("bad smoke payload".into());
        }
        match context.scenario {
            "transfer" => {
                code = 500;
                "failed".into()
            }
            "malformed" => "{".into(),
            "oversized" => "x".repeat(1_048_577),
            "stream" if stream_request => "data: {}\n\n".into(),
            "auto" if expected == "auto" => "{}".into(),
            _ if stream_request => "data: {\"role\":\"assistant\"}\n\ndata: [DONE]\n\n".into(),
            _ => {
                "{\"object\":\"chat.completion\",\"choices\":[{\"message\":{\"content\":\"hi\"}}]}"
                    .into()
            }
        }
    } else {
        return Err("unexpected smoke endpoint".into());
    };
    let response = format!(
        "HTTP/1.1 {code} OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    match stream.write_all(response.as_bytes()) {
        Ok(()) => {
            if context.scenario == "headless-spawn" && *chats == 3 {
                std::fs::remove_file(std::env::current_exe()?)?;
            }
            Ok(())
        }
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}
