//! Inert local host protocol fixture, owned by the existing l7 example target.
use std::{
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    path::Path,
    time::Duration,
};
pub(super) fn selected(args: &[String]) -> bool {
    args.windows(2)
        .find(|pair| pair[0] == "--config")
        .is_some_and(|pair| {
            std::fs::read_to_string(&pair[1])
                .is_ok_and(|text| text.contains("manual-smoke-fixture"))
        })
}
pub(super) fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let option = |name| {
        args.windows(2)
            .find(|pair| pair[0] == name)
            .map(|pair| pair[1].as_str())
            .ok_or("fixture option missing")
    };
    let root = std::path::PathBuf::from(
        std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR").ok_or("fixture root absent")?,
    );
    let config = std::fs::read_to_string(option("--config")?)?;
    std::fs::write(root.join("applied.toml"), &config)?;
    if config.contains("early-exit") {
        std::process::exit(23);
    }
    crate::signals::install()?;
    let listeners = [
        TcpListener::bind((Ipv4Addr::LOCALHOST, option("--port")?.parse::<u16>()?))?,
        TcpListener::bind((Ipv4Addr::LOCALHOST, option("--console")?.parse::<u16>()?))?,
    ];
    for listener in &listeners {
        listener.set_nonblocking(true)?;
    }
    while !crate::signals::stopped() {
        for listener in &listeners {
            match listener.accept() {
                Ok((stream, _)) => respond(stream, &root, &config)?,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => (),
                Err(error) => return Err(error.into()),
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    std::fs::write(root.join("stopped"), "graceful")?;
    Ok(())
}
fn respond(
    mut stream: TcpStream,
    root: &Path,
    config: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let mut bytes = Vec::new();
    let end = loop {
        let mut buffer = [0; 1024];
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Ok(());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > 65536 {
            return Err("fixture request bound".into());
        }
        if let Some(end) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let header = std::str::from_utf8(&bytes[..end])?;
    let path = header
        .split_whitespace()
        .nth(1)
        .ok_or("fixture path")?
        .to_owned();
    let length = header
        .lines()
        .find_map(|line| {
            line.to_lowercase()
                .strip_prefix("content-length:")
                .map(str::trim)
                .map(str::parse::<usize>)
        })
        .transpose()?
        .unwrap_or(0);
    if length > 65536 {
        return Err("fixture body bound".into());
    }
    while bytes.len() - end < length {
        let mut buffer = [0; 1024];
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err("fixture partial body".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let value = match path.as_str() {
        "/api/status" => serde_json::json!({"llama_ready":true}),
        "/v1/models" => serde_json::json!({"data":[{"id":"inert-model"}]}),
        "/v1/chat/completions" => {
            let request: serde_json::Value = serde_json::from_slice(&bytes[end..end + length])?;
            std::fs::write(root.join("request.json"), serde_json::to_vec(&request)?)?;
            if config.contains("held-chat") {
                while !crate::signals::stopped() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                return Ok(());
            }
            if config.contains("error-chat") {
                serde_json::json!({"error":{"message":"inert refusal"}})
            } else {
                serde_json::json!({"choices":[{"message":{"content":"Hello local fixture"}}]})
            }
        }
        _ => return Err("unexpected smoke path".into()),
    };
    let body = serde_json::to_vec(&value)?;
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    Ok(())
}
