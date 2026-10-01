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
    signals::install()?;
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let value = |flag| {
        arguments
            .windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].as_str())
            .ok_or("fixture flag missing")
    };
    let home = PathBuf::from(std::env::var_os("HOME").ok_or("home")?);
    let label = home
        .parent()
        .and_then(|path| path.file_name())
        .ok_or("node label")?
        .to_string_lossy()
        .into_owned();
    let root = PathBuf::from(
        std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR").ok_or("fixture root")?,
    );
    if label != "seed" && value("--join")? != "fixture-invite" {
        return Err("wrong invite".into());
    }
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
                Ok((stream, _)) => respond(stream, &root, &label)?,
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
) -> Result<(), Box<dyn std::error::Error>> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut bytes = Vec::new();
    loop {
        let mut buffer = [0; 1024];
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Ok(());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
            break;
        }
        if bytes.len() > 16384 {
            return Err("request limit".into());
        }
    }
    let header = String::from_utf8(bytes)?;
    let live: Vec<String> = std::fs::read_dir(root)?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.strip_suffix(".live"))
                .map(str::to_owned)
        })
        .collect();
    let recovered = !live.iter().any(|name| name == "worker-1");
    let body = if header.starts_with("GET /api/status ") {
        serde_json::json!({"token":"fixture-invite","node_id":label,"peers":live.iter().filter(|node|*node!=label).collect::<Vec<_>>()})
    } else if header.starts_with("GET /api/runtime/stages ") {
        let downstream = if recovered { "worker-2" } else { "worker-1" };
        let run = if recovered && !root.join("stale-run").exists() {
            "replacement"
        } else {
            "initial"
        };
        serde_json::json!({"topologies":[{"run_id":run,"stages":[{"stage_index":0,"node_id":"seed"},{"stage_index":1,"node_id":downstream}]}]})
    } else if header.starts_with("POST /v1/chat/completions ") {
        serde_json::json!({"object":"chat.completion","choices":[{"message":{"content":"hello"}}]})
    } else {
        serde_json::json!({"data":[{"id":"fixture-model"}]})
    };
    let body = serde_json::to_vec(&body)?;
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    Ok(())
}
