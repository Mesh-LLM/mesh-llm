#[path = "../tests/migration_lifecycle/signals.rs"]
#[expect(
    dead_code,
    reason = "fixture reuses signal ownership but not stdout closure"
)]
mod signals;
use std::{
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    path::PathBuf,
    time::Duration,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.first().is_some_and(|argument| argument == "exec") {
        if std::env::var("MESH_LOGS_E2E_PERSISTED_REQUEST_ID")? != "request-fixture" {
            return Err("browser child received wrong durable identity".into());
        }
        let root = PathBuf::from(
            std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR").ok_or("fixture root")?,
        );
        std::fs::write(root.join("browser-ran"), b"checked")?;
        if root.join("browser-fail").exists() {
            std::process::exit(23);
        }
        return Ok(());
    }
    signals::install()?;
    let value = |flag| {
        arguments
            .windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].as_str())
            .ok_or("missing flag")
    };
    let api = value("--port")?.parse::<u16>()?;
    let console = value("--console")?.parse::<u16>()?;
    let state = PathBuf::from(value("--config")?)
        .parent()
        .ok_or("state")?
        .to_owned();
    let fail_open = value("--config")?.ends_with("fail-open.toml");
    let count_path = state.join("launch-count");
    let count = std::fs::read_to_string(&count_path)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0)
        + 1;
    std::fs::write(&count_path, count.to_string())?;
    let listeners = [
        TcpListener::bind((Ipv4Addr::LOCALHOST, api))?,
        TcpListener::bind((Ipv4Addr::LOCALHOST, console))?,
    ];
    for listener in &listeners {
        listener.set_nonblocking(true)?;
    }
    while !signals::stopped() {
        for listener in &listeners {
            match listener.accept() {
                Ok((stream, _)) => respond(stream, &state, fail_open)?,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => (),
                Err(error) => return Err(error.into()),
            }
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    Ok(())
}

fn respond(
    mut stream: TcpStream,
    state: &std::path::Path,
    fail_open: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut buffer = [0; 1024];
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Ok(());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(index) = bytes.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            break index + 4;
        }
        if bytes.len() > 16384 {
            return Err("request limit".into());
        }
    };
    let header = String::from_utf8(bytes[..header_end].to_vec())?;
    let content_length = header
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .map(str::to_owned)
        })
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    while bytes.len() < header_end + content_length {
        let mut buffer = [0; 1024];
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err("short body".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let hostile = header.to_ascii_lowercase().contains("attacker.example");
    let root = PathBuf::from(
        std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR").ok_or("fixture root")?,
    );
    let (code, content_type, body) = if hostile {
        (
            403,
            "application/json",
            "{\"error\":{\"code\":\"forbidden\"}}".to_owned(),
        )
    } else if fail_open && header.starts_with("GET /api/logs/requests") {
        (
            503,
            "application/json",
            "{\"error\":{\"code\":\"unavailable\"}}".to_owned(),
        )
    } else if fail_open && header.starts_with("POST /v1/chat/completions ") {
        (
            200,
            "application/json",
            "{\"object\":\"chat.completion\",\"choices\":[{\"message\":{\"content\":\"ok\"}}]}"
                .to_owned(),
        )
    } else if header.starts_with("POST /api/logs/requests/request-fixture/delete ") {
        let payload: serde_json::Value = serde_json::from_slice(&bytes[header_end..])?;
        std::fs::remove_file(state.join("durable"))?;
        (200,"application/json",serde_json::json!({"requestId":"request-fixture","operationId":if root.join("wrong-delete").exists(){serde_json::json!("wrong")}else{payload["operationId"].clone()}}).to_string())
    } else if header.starts_with("POST /v1/chat/completions ") {
        std::fs::write(state.join("durable"), b"request-fixture")?;
        (
            400,
            "application/json",
            "{\"error\":\"no model\"}".to_owned(),
        )
    } else if header.starts_with("GET /api/logs/requests?limit=") {
        let body = if root.join("private-leak").exists() {
            "{\"items\":[{\"requestId\":\"request-fixture\",\"content\":\"QA_PRIVATE_MARKER_NOT_FOR_PERSISTENCE\"}]}"
        } else if state.join("durable").exists() {
            "{\"items\":[{\"requestId\":\"request-fixture\"}]}"
        } else {
            "{\"items\":[]}"
        };
        (200, "application/json", body.to_owned())
    } else if header.starts_with("GET /api/logs/requests/request-fixture ") {
        (
            if state.join("durable").exists() {
                200
            } else {
                404
            },
            "application/json",
            "{\"requestId\":\"request-fixture\"}".to_owned(),
        )
    } else if header.starts_with("GET /api/logs/events?") {
        (
            200,
            "text/event-stream",
            "event: replay_gap\ndata: {\"authoritative\":\"/api/logs/requests\"}\n\n".to_owned(),
        )
    } else {
        (200, "application/json", "{}".to_owned())
    };
    write!(
        stream,
        "HTTP/1.1 {code} OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    Ok(())
}
