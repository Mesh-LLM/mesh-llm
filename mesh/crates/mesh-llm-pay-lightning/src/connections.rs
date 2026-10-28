//! Connection epochs keep detached recovery from reviving a disconnected peer.
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Default)]
pub(crate) struct Connections {
    peers: Mutex<BTreeMap<String, (u64, bool)>>,
    pub(crate) handshake: tokio::sync::Mutex<()>,
}
impl Connections {
    pub(crate) fn begin(&self, peer: &str) -> Option<u64> {
        let mut peers = self.peers.lock().unwrap();
        let (epoch, connected) = peers.entry(peer.into()).or_default();
        (!*connected).then_some(*epoch)
    }
    pub(crate) fn complete(&self, peer: &str, epoch: u64) -> bool {
        let mut peers = self.peers.lock().unwrap();
        let state = peers.entry(peer.into()).or_default();
        if state.0 != epoch {
            return false;
        }
        state.1 = true;
        true
    }
    pub(crate) fn disconnect(&self, peer: &str) {
        let mut peers = self.peers.lock().unwrap();
        let state = peers.entry(peer.into()).or_default();
        state.0 = state.0.wrapping_add(1);
        state.1 = false;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disconnect_invalidates_pending_recovery_without_waiting_for_wallet() {
        let connections = Connections::default();
        let epoch = connections.begin("seller").unwrap();
        connections.disconnect("seller");
        assert!(!connections.complete("seller", epoch));
        let epoch = connections.begin("seller").unwrap();
        assert!(connections.complete("seller", epoch));
        assert!(connections.begin("seller").is_none());
    }
}
