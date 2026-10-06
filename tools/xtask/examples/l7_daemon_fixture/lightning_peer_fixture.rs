//! Inert HTTP/argv narrative, explicitly not mesh/payment/runtime qualification.
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    path::PathBuf,
    time::{Duration, Instant},
};
pub(super) fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let [mode, rest @ ..] = args else {
        return Err("fixture mode".into());
    };
    if rest == ["--version"] {
        println!("mesh-llm inert-{mode}");
        return Ok(());
    }
    let value = |key| {
        rest.windows(2)
            .find(|p| p[0] == key)
            .map(|p| p[1].as_str())
            .ok_or("fixture argv")
    };
    let config = PathBuf::from(value("--config")?);
    let root = config.parent().ok_or("fixture root")?;
    // Persist only invite presence, never the synthetic or product invite string.
    let mut recorded = rest.to_vec();
    if let Some(i) = recorded.iter().position(|v| v == "--join") {
        recorded[i + 1] = "<redacted>".into();
    }
    std::fs::write(root.join("inert-argv.json"), serde_json::to_vec(&recorded)?)?;
    let console: u16 = value("--console")?.parse()?;
    let api: u16 = value("--port")?.parse()?;
    let quic: u16 = value("--bind-port")?.parse()?;
    if console == api || api == quic || quic == console {
        return Err("fixture distinct ports".into());
    }
    if value("--bind-ip")? != "127.0.0.1" {
        return Err("fixture loopback".into());
    }
    let provider = rest.first().is_some_and(|s| s == "serve");
    if !provider && rest.first().is_none_or(|s| s != "client") {
        return Err("fixture surface".into());
    }
    let joined = if provider {
        None
    } else {
        Some(
            value("--join")?
                .strip_prefix("inert:")
                .ok_or("fixture invite")?
                .parse::<u16>()?,
        )
    };
    let a = TcpListener::bind((Ipv4Addr::LOCALHOST, console))?;
    let b = TcpListener::bind((Ipv4Addr::LOCALHOST, api))?;
    a.set_nonblocking(true)?;
    b.set_nonblocking(true)?;
    super::signals::install()?;
    let until = Instant::now() + Duration::from_secs(45);
    let mut paid = false;
    while !super::signals::stopped() && Instant::now() < until {
        for listener in [&a, &b] {
            let (mut socket, _) = match listener.accept() {
                Ok(v) => v,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(e) => return Err(e.into()),
            };
            socket.set_nonblocking(false)?;
            socket.set_read_timeout(Some(Duration::from_secs(1)))?;
            socket.set_write_timeout(Some(Duration::from_secs(1)))?;
            let (path, body) = read(&mut socket)?;
            let (status, value) = match path.as_str() {
                "/api/status" => (200, json!({"token":format!("inert:{console}")})),
                "/v1/models" => (200, json!({"data":[{"id":"Payment-Compatibility-Smoke"}]})),
                "/api/wallet" => {
                    if body["command"] != "set_pricing"
                        || body["model"] != "Payment-Compatibility-Smoke"
                        || body["value"]["minimum_invoice_msat"] != 1000
                    {
                        return Err("fixture pricing".into());
                    }
                    paid = true;
                    (200, json!({"ok":true}))
                }
                "/api/fixture-price" => (200, json!({"paid":paid})),
                "/v1/chat/completions" => {
                    if body["model"] != "Payment-Compatibility-Smoke"
                        || body["max_tokens"] != 8
                        || body["stream"] != false
                        || body["messages"][0]["content"] != "Say hello."
                    {
                        return Err("fixture request projection".into());
                    }
                    std::fs::write(root.join("received-inference"), b"observed")?;
                    if mode == "hold" {
                        while !super::signals::stopped() && Instant::now() < until {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        return Ok(());
                    }
                    let charged = if let Some(port) = joined {
                        let mut peer = TcpStream::connect_timeout(
                            &format!("127.0.0.1:{port}").parse()?,
                            Duration::from_secs(1),
                        )?;
                        peer.set_read_timeout(Some(Duration::from_secs(1)))?;
                        peer.set_write_timeout(Some(Duration::from_secs(1)))?;
                        peer.write_all(b"GET /api/fixture-price HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
                        let mut bytes = Vec::new();
                        peer.take(65536).read_to_end(&mut bytes)?;
                        let end = bytes
                            .windows(4)
                            .position(|w| w == b"\r\n\r\n")
                            .ok_or("price response")?;
                        serde_json::from_slice::<Value>(&bytes[end + 4..])?["paid"] == true
                    } else {
                        paid
                    };
                    if charged && mode != "bypass" {
                        (402, json!({"error":{"message":"free-only policy"}}))
                    } else {
                        (
                            200,
                            json!({"choices":[{"message":{"content":"hello"}}],"usage":{"completion_tokens":if mode=="missing-usage"{0}else{1}}}),
                        )
                    }
                }
                _ => return Err("unexpected fixture endpoint".into()),
            };
            let bytes = serde_json::to_vec(&value)?;
            write!(
                socket,
                "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            )?;
            socket.write_all(&bytes)?;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(())
}
fn read(socket: &mut TcpStream) -> Result<(String, Value), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    let mut buf = [0; 4096];
    loop {
        let n = socket.read(&mut buf)?;
        if n == 0 {
            return Err("fixture incomplete request".into());
        }
        bytes.extend_from_slice(&buf[..n]);
        if bytes.len() > 65536 {
            return Err("fixture request bound".into());
        }
        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let text = std::str::from_utf8(&bytes[..end])?;
            let path = text
                .lines()
                .next()
                .ok_or("request line")?
                .split_whitespace()
                .nth(1)
                .ok_or("request path")?
                .to_owned();
            let length = text
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|v| v.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                return Ok((
                    path,
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
