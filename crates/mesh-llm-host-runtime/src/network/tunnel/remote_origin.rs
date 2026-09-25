//! Preserve remote provenance when legacy embedders bridge QUIC through TCP.
//! A process-owned socket registry cannot be forged through HTTP headers.
use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::{Mutex, OnceLock};

fn connections() -> &'static Mutex<HashSet<SocketAddr>> {
    static CONNECTIONS: OnceLock<Mutex<HashSet<SocketAddr>>> = OnceLock::new();
    CONNECTIONS.get_or_init(Mutex::default)
}

pub(super) struct RemoteBridge(SocketAddr);
impl RemoteBridge {
    pub(super) fn register(address: SocketAddr) -> anyhow::Result<Self> {
        connections()
            .lock()
            .map_err(|_| anyhow::anyhow!("remote origin registry unavailable"))?
            .insert(address);
        Ok(Self(address))
    }
}
impl Drop for RemoteBridge {
    fn drop(&mut self) {
        if let Ok(mut connections) = connections().lock() {
            connections.remove(&self.0);
        }
    }
}

#[cfg(feature = "payments")]
pub(crate) fn is_remote_bridge(address: SocketAddr) -> bool {
    // Poisoned provenance fails closed for wallet spending.
    connections()
        .lock()
        .map_or(true, |connections| connections.contains(&address))
}

#[cfg(all(test, feature = "payments"))]
mod tests {
    use super::*;
    #[tokio::test]
    async fn payments_remote_provenance_follows_actual_bridge_socket_lifetime() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bridge = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let guard = RemoteBridge::register(bridge.local_addr().unwrap()).unwrap();
        let (_, caller) = listener.accept().await.unwrap();
        assert!(is_remote_bridge(caller));
        drop(guard);
        assert!(!is_remote_bridge(caller));
    }
}
