use super::*;

const STUN_BINDING_REQUEST: u16 = 0x0001;
const STUN_BINDING_SUCCESS_RESPONSE: u16 = 0x0101;
const STUN_MAGIC_COOKIE: [u8; 4] = [0x21, 0x12, 0xA4, 0x42];
const STUN_HEADER_LEN: usize = 20;
const STUN_TRANSACTION_ID_LEN: usize = 12;
const STUN_RESPONSE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

struct StunBindingRequest {
    bytes: [u8; STUN_HEADER_LEN],
}

impl StunBindingRequest {
    fn new() -> Self {
        let mut bytes = [0u8; STUN_HEADER_LEN];
        bytes[..2].copy_from_slice(&STUN_BINDING_REQUEST.to_be_bytes());
        bytes[4..8].copy_from_slice(&STUN_MAGIC_COOKIE);
        rand::fill(&mut bytes[8..STUN_HEADER_LEN]);
        Self { bytes }
    }

    fn bytes(&self) -> &[u8; STUN_HEADER_LEN] {
        &self.bytes
    }

    fn transaction_id(&self) -> &[u8] {
        &self.bytes[8..STUN_HEADER_LEN]
    }
}

pub(crate) async fn stun_public_addr(advertised_port: u16) -> Option<std::net::SocketAddr> {
    let stun_servers = [
        "stun.l.google.com:19302",
        "stun.cloudflare.com:3478",
        "stun.stunprotocol.org:3478",
    ];

    let sock = tokio::net::UdpSocket::bind("0.0.0.0:0").await.ok()?;

    for server in &stun_servers {
        if let Some(addr) = probe_stun_server(&sock, server, advertised_port).await {
            tracing::info!("STUN discovered public address: {addr}");
            return Some(addr);
        }
    }

    tracing::warn!("STUN: could not discover public address");
    None
}

pub(crate) async fn probe_stun_server(
    sock: &tokio::net::UdpSocket,
    server: &str,
    advertised_port: u16,
) -> Option<std::net::SocketAddr> {
    let request = StunBindingRequest::new();
    let dest = resolve_stun_server(server).await?;
    sock.send_to(request.bytes(), dest).await.ok()?;

    receive_matching_stun_response(sock, dest, request.transaction_id(), advertised_port).await
}

async fn receive_matching_stun_response(
    sock: &tokio::net::UdpSocket,
    dest: std::net::SocketAddr,
    transaction_id: &[u8],
    advertised_port: u16,
) -> Option<std::net::SocketAddr> {
    let deadline = tokio::time::Instant::now() + STUN_RESPONSE_TIMEOUT;
    let mut buf = [0u8; 256];

    loop {
        let remaining = deadline.checked_duration_since(tokio::time::Instant::now())?;
        let (len, source) = tokio::time::timeout(remaining, sock.recv_from(&mut buf))
            .await
            .ok()?
            .ok()?;
        if let Some(addr) =
            parse_stun_public_addr_from(&buf, len, source, dest, transaction_id, advertised_port)
        {
            return Some(addr);
        }
    }
}

async fn resolve_stun_server(server: &str) -> Option<std::net::SocketAddr> {
    let mut addrs = tokio::net::lookup_host(server).await.ok()?;
    addrs.next()
}

fn parse_stun_public_addr_from(
    response: &[u8],
    len: usize,
    source: std::net::SocketAddr,
    expected_source: std::net::SocketAddr,
    transaction_id: &[u8],
    advertised_port: u16,
) -> Option<std::net::SocketAddr> {
    if source != expected_source || !valid_stun_success_header(response, len, transaction_id) {
        return None;
    }
    parse_stun_public_addr(response, len, advertised_port)
}

fn valid_stun_success_header(response: &[u8], len: usize, transaction_id: &[u8]) -> bool {
    if len < STUN_HEADER_LEN
        || response.len() < len
        || transaction_id.len() != STUN_TRANSACTION_ID_LEN
    {
        return false;
    }
    let message_type = u16::from_be_bytes([response[0], response[1]]);
    let message_len = u16::from_be_bytes([response[2], response[3]]) as usize;
    message_type == STUN_BINDING_SUCCESS_RESPONSE
        && response[4..8] == STUN_MAGIC_COOKIE
        && response[8..STUN_HEADER_LEN] == *transaction_id
        && STUN_HEADER_LEN + message_len == len
}

fn parse_stun_mapped_ipv4(
    attr_type: u16,
    value: &[u8],
    advertised_port: u16,
) -> Option<std::net::SocketAddr> {
    use std::net::SocketAddrV4;

    if value.len() < 8 || value[1] != 0x01 {
        return None;
    }
    let ip = match attr_type {
        0x0020 => Ipv4Addr::new(
            value[4] ^ STUN_MAGIC_COOKIE[0],
            value[5] ^ STUN_MAGIC_COOKIE[1],
            value[6] ^ STUN_MAGIC_COOKIE[2],
            value[7] ^ STUN_MAGIC_COOKIE[3],
        ),
        0x0001 => Ipv4Addr::new(value[4], value[5], value[6], value[7]),
        _ => return None,
    };
    Some(std::net::SocketAddr::V4(SocketAddrV4::new(
        ip,
        advertised_port,
    )))
}

fn parse_stun_public_addr(
    response: &[u8],
    len: usize,
    advertised_port: u16,
) -> Option<std::net::SocketAddr> {
    let mut i = STUN_HEADER_LEN;
    while i + 4 <= len {
        let attr_type = u16::from_be_bytes([response[i], response[i + 1]]);
        let attr_len = u16::from_be_bytes([response[i + 2], response[i + 3]]) as usize;
        if i + 4 + attr_len > len {
            return None;
        }
        let value = &response[i + 4..i + 4 + attr_len];
        if let Some(addr) = parse_stun_mapped_ipv4(attr_type, value, advertised_port) {
            return Some(addr);
        }
        i += (4 + (attr_len + 3)) & !3;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response_bytes(transaction_id: &[u8], message_type: u16) -> Vec<u8> {
        let mapped_ip = [203, 0, 113, 7];
        let mut response = vec![0u8; 32];
        response[..2].copy_from_slice(&message_type.to_be_bytes());
        response[2..4].copy_from_slice(&12u16.to_be_bytes());
        response[4..8].copy_from_slice(&STUN_MAGIC_COOKIE);
        response[8..20].copy_from_slice(transaction_id);
        response[20..22].copy_from_slice(&0x0020u16.to_be_bytes());
        response[22..24].copy_from_slice(&8u16.to_be_bytes());
        response[25] = 0x01;
        response[28] = mapped_ip[0] ^ STUN_MAGIC_COOKIE[0];
        response[29] = mapped_ip[1] ^ STUN_MAGIC_COOKIE[1];
        response[30] = mapped_ip[2] ^ STUN_MAGIC_COOKIE[2];
        response[31] = mapped_ip[3] ^ STUN_MAGIC_COOKIE[3];
        response
    }

    #[test]
    fn accepts_success_response_when_source_and_transaction_match() {
        let request = StunBindingRequest::new();
        let source = std::net::SocketAddr::from(([127, 0, 0, 1], 3478));
        let response = response_bytes(request.transaction_id(), STUN_BINDING_SUCCESS_RESPONSE);

        let addr = parse_stun_public_addr_from(
            &response,
            response.len(),
            source,
            source,
            request.transaction_id(),
            7842,
        );

        assert_eq!(
            addr,
            Some(std::net::SocketAddr::from(([203, 0, 113, 7], 7842)))
        );
    }

    #[test]
    fn rejects_response_from_unqueried_source() {
        let request = StunBindingRequest::new();
        let expected = std::net::SocketAddr::from(([127, 0, 0, 1], 3478));
        let wrong_source = std::net::SocketAddr::from(([127, 0, 0, 1], 3479));
        let response = response_bytes(request.transaction_id(), STUN_BINDING_SUCCESS_RESPONSE);

        let addr = parse_stun_public_addr_from(
            &response,
            response.len(),
            wrong_source,
            expected,
            request.transaction_id(),
            7842,
        );

        assert_eq!(addr, None);
    }

    #[test]
    fn rejects_non_success_response_type() {
        let request = StunBindingRequest::new();
        let source = std::net::SocketAddr::from(([127, 0, 0, 1], 3478));
        let response = response_bytes(request.transaction_id(), STUN_BINDING_REQUEST);

        let addr = parse_stun_public_addr_from(
            &response,
            response.len(),
            source,
            source,
            request.transaction_id(),
            7842,
        );

        assert_eq!(addr, None);
    }

    #[test]
    fn rejects_response_with_wrong_magic_cookie() {
        let request = StunBindingRequest::new();
        let source = std::net::SocketAddr::from(([127, 0, 0, 1], 3478));
        let mut response = response_bytes(request.transaction_id(), STUN_BINDING_SUCCESS_RESPONSE);
        response[4] = 0;

        let addr = parse_stun_public_addr_from(
            &response,
            response.len(),
            source,
            source,
            request.transaction_id(),
            7842,
        );

        assert_eq!(addr, None);
    }

    #[test]
    fn rejects_response_with_wrong_transaction_id() {
        let request = StunBindingRequest::new();
        let source = std::net::SocketAddr::from(([127, 0, 0, 1], 3478));
        let other_request = StunBindingRequest::new();
        let response = response_bytes(
            other_request.transaction_id(),
            STUN_BINDING_SUCCESS_RESPONSE,
        );

        let addr = parse_stun_public_addr_from(
            &response,
            response.len(),
            source,
            source,
            request.transaction_id(),
            7842,
        );

        assert_eq!(addr, None);
    }
}
