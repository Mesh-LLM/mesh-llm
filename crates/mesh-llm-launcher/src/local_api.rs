use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
use std::time::Duration;

#[derive(Debug, Deserialize)]
pub(crate) struct PairingSessionsResponse {
    pub(crate) sessions: Vec<PairingSession>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PairingSession {
    pub(crate) id: String,
    pub(crate) direction: String,
    pub(crate) peer_name: String,
    pub(crate) status: String,
}

pub(crate) fn mesh_peer_count(port: u16) -> Option<usize> {
    let response = request(port, "GET", "/api/status").ok()?;
    let payload = serde_json::from_str::<serde_json::Value>(&response).ok()?;
    payload.get("peers")?.as_array().map(Vec::len)
}

pub(crate) fn pairing_sessions(port: u16) -> Vec<PairingSession> {
    let Ok(response) = request(port, "GET", "/api/pairing/sessions") else {
        return Vec::new();
    };
    serde_json::from_str::<PairingSessionsResponse>(&response)
        .map(|payload| payload.sessions)
        .unwrap_or_default()
}

pub(crate) fn shutdown_mesh(port: u16) -> Result<()> {
    request(port, "POST", "/api/runtime/shutdown").map(|_| ())
}

fn request(port: u16, method: &str, path: &str) -> Result<String> {
    let management_addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
    let mut stream = TcpStream::connect_timeout(&management_addr, Duration::from_millis(600))
        .context("Mesh management API is not running")?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost:{port}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    let (head, body) = response
        .split_once("\r\n\r\n")
        .context("Mesh management API returned an incomplete response")?;
    let status = head.lines().next().unwrap_or_default();
    anyhow::ensure!(
        status.contains(" 200 ") || status.contains(" 201 ") || status.contains(" 202 "),
        "Mesh management API rejected the request: {status}"
    );
    Ok(body.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launcher_pairing_payload_has_only_display_fields() {
        let payload = serde_json::from_str::<PairingSessionsResponse>(
            r#"{"sessions":[{"id":"request-1","direction":"incoming","peer_name":"Kitchen PC","peer_id":"secret-id","comparison_code":"123456","status":"awaiting_approval","created_at":1,"expires_at":2}]}"#,
        )
        .unwrap();
        assert_eq!(payload.sessions[0].peer_name, "Kitchen PC");
        assert_eq!(payload.sessions[0].status, "awaiting_approval");
    }
}
