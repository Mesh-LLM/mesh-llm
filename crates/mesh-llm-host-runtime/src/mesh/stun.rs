use super::*;
use iroh::Watcher;

fn public_ipv4_addr(endpoint_addr: &iroh::EndpointAddr) -> Option<std::net::SocketAddr> {
    endpoint_addr
        .ip_addrs()
        .copied()
        .find(is_public_ipv4_candidate)
}

pub(crate) async fn stun_public_addr(endpoint: &iroh::Endpoint) -> Option<std::net::SocketAddr> {
    let mut addresses = endpoint.watch_addr();
    let deadline =
        tokio::time::Instant::now() + std::time::Duration::from_secs(iroh::NET_REPORT_TIMEOUT);

    loop {
        if let Some(addr) = public_ipv4_addr(&addresses.get()) {
            tracing::info!(%addr, "QUIC endpoint discovered public address");
            return Some(addr);
        }

        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .unwrap_or_default();
        match tokio::time::timeout(remaining, addresses.updated()).await {
            Ok(Ok(_)) => {}
            Ok(Err(_)) | Err(_) => {
                tracing::warn!("QUIC endpoint could not discover a public address");
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_port_discovered_by_quic_endpoint() {
        let endpoint_id = iroh::SecretKey::generate().public();
        let mapped = std::net::SocketAddr::from(([9, 9, 9, 9], 45_678));
        let endpoint_addr = iroh::EndpointAddr::new(endpoint_id).with_ip_addr(mapped);

        assert_eq!(public_ipv4_addr(&endpoint_addr), Some(mapped));
    }

    #[test]
    fn ignores_local_quic_endpoint_addresses() {
        let endpoint_id = iroh::SecretKey::generate().public();
        let endpoint_addr = iroh::EndpointAddr::new(endpoint_id)
            .with_ip_addr(std::net::SocketAddr::from(([192, 168, 1, 8], 45_678)));

        assert_eq!(public_ipv4_addr(&endpoint_addr), None);
    }
}
