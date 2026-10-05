//! Bounded duplicate suppression for Mesh plugin messages.

use std::collections::{HashMap, VecDeque};

#[derive(Default)]
pub(crate) struct PluginMessageHistory {
    seen_plugin_messages: HashMap<String, std::time::Instant>,
    seen_plugin_message_order: VecDeque<(std::time::Instant, String)>,
}

impl PluginMessageHistory {
    pub(super) fn remember(&mut self, message_id: String, now: std::time::Instant) -> bool {
        /// How long to remember a message ID. Any duplicate arriving within
        /// this window is suppressed. This must be longer than the worst-case
        /// propagation delay across alternate mesh paths — 120s is generous.
        const DEDUP_TTL: std::time::Duration = std::time::Duration::from_secs(120);
        /// Hard cap to bound memory even if message volume is extreme.
        const DEDUP_HARD_CAP: usize = 100_000;

        // Evict entries older than the TTL
        while let Some((ts, _)) = self.seen_plugin_message_order.front() {
            if now.duration_since(*ts) >= DEDUP_TTL {
                if let Some((_, id)) = self.seen_plugin_message_order.pop_front() {
                    self.seen_plugin_messages.remove(&id);
                }
            } else {
                break;
            }
        }

        // Already seen?
        if self.seen_plugin_messages.contains_key(&message_id) {
            return false;
        }

        // Hard cap: if under extreme load we still accumulate too many,
        // evict the oldest regardless of TTL.
        while self.seen_plugin_message_order.len() >= DEDUP_HARD_CAP {
            if let Some((_, id)) = self.seen_plugin_message_order.pop_front() {
                self.seen_plugin_messages.remove(&id);
            }
        }

        self.seen_plugin_messages.insert(message_id.clone(), now);
        self.seen_plugin_message_order.push_back((now, message_id));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn duplicates_remain_suppressed_until_the_original_entry_expires() {
        let mut history = PluginMessageHistory::default();
        let now = Instant::now();
        assert!(history.remember("message".into(), now));
        assert!(!history.remember("message".into(), now + Duration::from_secs(119)));
        assert!(history.remember("message".into(), now + Duration::from_secs(120)));
    }

    #[test]
    fn capacity_evicts_the_oldest_id_and_keeps_recent_duplicates_suppressed() {
        let mut history = PluginMessageHistory::default();
        let now = Instant::now();
        for index in 0..=100_000 {
            assert!(history.remember(index.to_string(), now));
        }
        assert_eq!(history.seen_plugin_messages.len(), 100_000);
        assert!(!history.remember("100000".into(), now));
        assert!(history.remember("0".into(), now));
        assert_eq!(history.seen_plugin_messages.len(), 100_000);
    }

    #[tokio::test]
    async fn node_clones_share_deduplication_without_locking_membership() {
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let clone = node.clone();
        let _membership = node.state.lock().await;
        assert!(
            tokio::time::timeout(
                Duration::from_secs(1),
                node.remember_plugin_message("id".into())
            )
            .await
            .expect("plugin deduplication must not acquire the membership lock")
        );
        assert!(!clone.remember_plugin_message("id".into()).await);
    }
}
