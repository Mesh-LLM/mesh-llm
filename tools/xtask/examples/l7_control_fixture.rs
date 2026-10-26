#[path = "../tests/migration_lifecycle/signals.rs"]
#[expect(dead_code, reason = "shared fixture signal ownership")]
mod signals;
use std::{
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    time::Duration,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let value = |flag| {
        arguments
            .windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].as_str())
            .ok_or("flag")
    };
    if arguments.iter().any(|argument| argument == "--version") {
        println!("control fixture 1");
        return Ok(());
    }
    if arguments.iter().any(|argument| argument == "--help") {
        println!("--owner-key --headless --client serve");
        return Ok(());
    }
    if arguments.iter().any(|argument| argument == "auth") {
        std::fs::write(value("--owner-key")?, b"fixture owner")?;
        return Ok(());
    }
    let root = PathBuf::from(
        std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR").ok_or("fixture root")?,
    );
    if arguments.iter().any(|argument| argument == "runtime") {
        if value("--endpoint")? != "control://target" {
            return Err("wrong explicit endpoint".into());
        }
        if arguments.iter().any(|argument| argument == "scan-refresh") {
            let inventory = if root.join("unsorted-scan").exists() {
                vec![
                    serde_json::json!({"canonical_model_ref":"z","metadata":{}}),
                    serde_json::json!({"canonical_model_ref":"a","metadata":{}}),
                ]
            } else {
                vec![
                    serde_json::json!({"canonical_model_ref":"a","metadata":{"architecture":"fixture"}}),
                ]
            };
            println!(
                "{}",
                serde_json::json!({"disposition":"executed","target_node_id":"target","inventory":inventory})
            );
        } else {
            println!("{{\"config\":{{}}}}");
        }
        return Ok(());
    }
    signals::install()?;
    let home = PathBuf::from(std::env::var_os("HOME").ok_or("home")?);
    let label = home
        .parent()
        .and_then(|path| path.file_name())
        .ok_or("label")?
        .to_string_lossy()
        .into_owned();
    if arguments.iter().any(|argument| argument == "--join") && value("--join")? != "fixture-invite"
    {
        return Err("wrong join token".into());
    }
    let wrong = arguments
        .windows(2)
        .any(|pair| pair[0] == "--owner-key" && pair[1].ends_with("wrong-owner.json"));
    std::fs::write(
        root.join(format!("{label}.live")),
        std::process::id().to_string(),
    )?;
    let listeners = [
        TcpListener::bind((Ipv4Addr::LOCALHOST, value("--port")?.parse::<u16>()?))?,
        TcpListener::bind((Ipv4Addr::LOCALHOST, value("--console")?.parse::<u16>()?))?,
    ];
    for listener in &listeners {
        listener.set_nonblocking(true)?;
    }
    while !signals::stopped() {
        for listener in &listeners {
            match listener.accept() {
                Ok((stream, _)) => respond(stream, &root, &label, wrong)?,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => (),
                Err(error) => return Err(error.into()),
            }
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    std::fs::remove_file(root.join(format!("{label}.live")))?;
    Ok(())
}

fn respond(
    mut stream: TcpStream,
    root: &Path,
    label: &str,
    wrong: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut bytes = Vec::new();
    let end = loop {
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
            return Err("request bound".into());
        }
    };
    let header = String::from_utf8(bytes[..end].to_vec())?;
    let length = header
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .map(str::to_owned)
        })
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    while bytes.len() < end + length {
        let mut buffer = [0; 1024];
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err("short body".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let path = header.split_whitespace().nth(1).ok_or("path")?;
    let mut code = 200;
    let body = if path == "/api/status" {
        let mut status = serde_json::json!({"token":"fixture-invite","peers":[{"node_id":"peer"}],"node_id":label});
        if root.join("control-leak").exists() {
            status["nested"] = serde_json::json!({"control_endpoint":"control://private"});
        }
        status
    } else if path == "/api/runtime/control-bootstrap" {
        serde_json::json!({"enabled":true,"requires_explicit_remote_endpoint":true,"endpoint":if label=="released-server"{"control://legacy"}else{"control://target"}})
    } else if path == "/api/runtime/control/scan-refresh" {
        if !wrong || root.join("allow-wrong-owner").exists() {
            serde_json::json!({"disposition":"executed"})
        } else {
            code = 403;
            serde_json::json!({"error":{"code":"unauthorized_owner","message":"owner mismatch"}})
        }
    } else if path.starts_with("/api/runtime/control/") {
        let payload: serde_json::Value = serde_json::from_slice(&bytes[end..])?;
        if payload["endpoint"] == "control://legacy" {
            code = 503;
            serde_json::json!({"error":{"code":"control_unsupported"}})
        } else {
            serde_json::json!({"accepted":true,"model":"qa.invalid/model@main:missing.gguf","instance_id":null})
        }
    } else if path == "/v1/models" {
        serde_json::json!({"data":[{"id":"fixture-model"}]})
    } else {
        serde_json::json!({"object":"chat.completion","choices":[{"message":{"content":"hello"}}]})
    };
    let body = serde_json::to_vec(&body)?;
    write!(
        stream,
        "HTTP/1.1 {code} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    Ok(())
}
