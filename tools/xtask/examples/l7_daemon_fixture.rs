#[path = "l7_daemon_fixture/competitive_fixture.rs"]
mod competitive_fixture;
#[path = "l7_daemon_fixture/hf_mtp_compose_fixture.rs"]
mod hf_mtp_compose_fixture;
#[path = "l7_daemon_fixture/lightning_peer_fixture.rs"]
mod lightning_peer_fixture;
#[path = "l7_daemon_fixture/manual_smoke_fixture.rs"]
mod manual_smoke_fixture;
#[path = "../tests/migration_lifecycle/signals.rs"]
#[expect(dead_code, reason = "shared fixture signal ownership")]
mod signals;
use std::{
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    path::PathBuf,
    time::Duration,
};

#[path = "l7_daemon_fixture/hf_certification_fixture.rs"]
mod hf_certification_fixture;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.first().is_some_and(|v| {
        v == "compose-mtp"
            || (v == "validate-mtp-attach" && !arguments.iter().any(|a| a == "--projector"))
    }) {
        return hf_mtp_compose_fixture::run(&arguments);
    }
    if arguments.first().is_some_and(|value| {
        ["validate-projector", "validate-mtp-attach"].contains(&value.as_str())
    }) {
        return hf_certification_fixture::run(&arguments);
    }
    if let [verb, rest @ ..] = arguments.as_slice()
        && verb == "lightning-peer"
    {
        return lightning_peer_fixture::run(rest);
    }
    if arguments.iter().any(|argument| argument == "--version") {
        if std::path::Path::new("version-oversized").exists() {
            std::io::stdout().write_all(&vec![b'v'; 65537])?;
            return Ok(());
        }
        if std::path::Path::new("version-held").exists() {
            crate::signals::install()?;
            while !crate::signals::stopped() {
                std::thread::sleep(Duration::from_millis(5));
            }
            return Ok(());
        }
        println!("fixture version 1");

        return Ok(());
    }
    if manual_smoke_fixture::selected(&arguments) {
        return manual_smoke_fixture::run(&arguments);
    }
    if competitive_fixture::selected(&arguments) {
        return competitive_fixture::run(&arguments);
    }
    if arguments.iter().any(|argument| argument == "auth") {
        if let Some(root) = std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")
            && PathBuf::from(root).join("owner-unavailable").exists()
        {
            std::process::exit(23);
        }
        let key = arguments
            .windows(2)
            .find(|pair| pair[0] == "--owner-key")
            .ok_or("key")?;
        std::fs::write(&key[1], b"inert owner fixture")?;
        return Ok(());
    }
    let value = |flag| {
        arguments
            .windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].as_str())
            .ok_or("flag")
    };
    if let Ok(config) = value("--config")
        && std::fs::read_to_string(config)?.contains("fail_fast")
    {
        std::process::exit(23);
    }
    if let Ok(config) = value("--config")
        && std::fs::read_to_string(config)?.contains("on_demand")
        && let Some(root) = std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")
        && PathBuf::from(root).join("on-demand-usage").exists()
    {
        println!("usage: on-demand requires model");
        return Ok(());
    }
    signals::install()?;
    let root = PathBuf::from(
        std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR").ok_or("fixture root")?,
    );
    let api = value("--port")?.parse::<u16>()?;
    let console = value("--console")?.parse::<u16>()?;
    let listeners = [
        TcpListener::bind((Ipv4Addr::LOCALHOST, api))?,
        TcpListener::bind((Ipv4Addr::LOCALHOST, console))?,
    ];
    for listener in &listeners {
        listener.set_nonblocking(true)?;
    }
    let mut intents = Vec::new();
    while !signals::stopped() {
        for listener in &listeners {
            match listener.accept() {
                Ok((stream, _)) => respond(stream, &root, &mut intents)?,
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
    root: &std::path::Path,
    intents: &mut Vec<serde_json::Value>,
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
    let path = header.split_whitespace().nth(1).ok_or("request path")?;
    let body = if path == "/api/runtime/control-bootstrap" {
        serde_json::json!({"endpoint":"control://fixture"})
    } else if let Some(operation) = path.strip_prefix("/api/runtime/control/") {
        let desired = match operation {
            "load-model" | "ensure-model" => "present",
            "unload-model" => "absent",
            "drain-model" => "draining",
            _ => return Err("operation".into()),
        };
        let id = format!("intent-{}", intents.len());
        intents.push(serde_json::json!({"intent_id":if root.join("wrong-intent").exists(){"wrong"}else{&id},"model_ref":"qa.invalid/model@main:missing.gguf","source":"owner_lifecycle","desired_state":desired}));
        serde_json::json!({"accepted":true,"intent_id":id,"accepted_state":desired,"model":"qa.invalid/model@main:missing.gguf","instance_id":null})
    } else if path == "/api/runtime/intents" {
        serde_json::json!({"intents":intents})
    } else if path.starts_with("/api/runtime/activity") {
        let mode = if header.starts_with("PUT ") {
            "active"
        } else {
            "auto"
        };
        let mut activity = serde_json::json!({"effective_state":"accepting","override_mode":mode,"detector_category":"unavailable"});
        if root.join("private-activity").exists() {
            activity["raw_window_title"] = serde_json::json!("private");
        }
        activity
    } else if path == "/v1/models" {
        serde_json::json!({"data":[]})
    } else {
        serde_json::json!({"ready":true})
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
