//! Plugin contributions to `GET /api/status`.
//!
//! The node's collector keeps each plugin's latest accepted snapshot, so a
//! status read never reaches a plugin. The host owns the envelope around the
//! plugin's data: receipt time, freshness and availability. Limits and the
//! wire contract are in `mesh_llm_plugin::plugin_status`.

use crate::api::status::{PluginStatusPayload, PluginStatusState};
use crate::plugin::proto;
use mesh_llm_plugin::plugin_status::{MAX_DATA_BYTES, MIN_PUBLISH_INTERVAL, effective_fresh_for};
use std::collections::BTreeMap;
use std::time::Duration;
use tokio::task::AbortHandle;
use tokio::time::Instant;

/// Why the host refused a snapshot.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PluginStatusRejection {
    TooLarge { bytes: usize },
    NotAnObject,
    TooSoon { retry_after: Duration },
}

impl PluginStatusRejection {
    pub(crate) fn message(&self, plugin_name: &str) -> String {
        match self {
            Self::TooLarge { bytes } => format!(
                "status data from plugin `{plugin_name}` is {bytes} bytes; the limit is {MAX_DATA_BYTES}"
            ),
            Self::NotAnObject => {
                format!("status data from plugin `{plugin_name}` must be a JSON object")
            }
            Self::TooSoon { retry_after } => format!(
                "plugin `{plugin_name}` published its status less than {MIN_PUBLISH_INTERVAL:?} ago; retry in {retry_after:?}"
            ),
        }
    }
}

struct PluginStatusEntry {
    /// The plugin connection that published it.
    generation: u64,
    schema_version: u32,
    data: serde_json::Value,
    received_at: Instant,
    received_at_unix_ms: u64,
    fresh_for: Duration,
    /// False once that connection is gone.
    connected: bool,
    /// Wakes `/api/events` when the snapshot turns stale.
    expiry: Option<AbortHandle>,
}

impl PluginStatusEntry {
    fn state(&self, now: Instant) -> PluginStatusState {
        if !self.connected {
            PluginStatusState::Unavailable
        } else if now.saturating_duration_since(self.received_at) >= self.fresh_for {
            PluginStatusState::Stale
        } else {
            PluginStatusState::Ready
        }
    }

    fn fresh_until_unix_ms(&self) -> u64 {
        self.received_at_unix_ms
            .saturating_add(u64::try_from(self.fresh_for.as_millis()).unwrap_or(u64::MAX))
    }

    fn cancel_expiry(&mut self) {
        if let Some(expiry) = self.expiry.take() {
            expiry.abort();
        }
    }
}

impl Drop for PluginStatusEntry {
    fn drop(&mut self) {
        self.cancel_expiry();
    }
}

/// A snapshot the host accepted.
pub(crate) struct AcceptedPluginStatus {
    pub(crate) response: proto::PublishPluginStatusResponse,
    /// When it turns stale, for the expiry wake-up. `None` after a withdrawal.
    pub(crate) fresh_for: Option<Duration>,
}

#[derive(Default)]
pub(crate) struct PluginStatusStore {
    entries: BTreeMap<String, PluginStatusEntry>,
}

impl PluginStatusStore {
    /// Accept `request` from connection `generation` of `plugin_name`, which
    /// the caller has checked is the plugin's current, initialized connection
    /// and declares `schema_version`. An empty `data_json` withdraws.
    pub(crate) fn publish(
        &mut self,
        plugin_name: &str,
        generation: u64,
        schema_version: u32,
        request: &proto::PublishPluginStatusRequest,
        now: Instant,
        now_unix_ms: u64,
    ) -> Result<AcceptedPluginStatus, PluginStatusRejection> {
        if request.data_json.is_empty() {
            self.entries.remove(plugin_name);
            return Ok(AcceptedPluginStatus {
                response: proto::PublishPluginStatusResponse::default(),
                fresh_for: None,
            });
        }
        if request.data_json.len() > MAX_DATA_BYTES {
            return Err(PluginStatusRejection::TooLarge {
                bytes: request.data_json.len(),
            });
        }
        if let Some(previous) = self.entries.get(plugin_name)
            && previous.generation == generation
        {
            let elapsed = now.saturating_duration_since(previous.received_at);
            if elapsed < MIN_PUBLISH_INTERVAL {
                return Err(PluginStatusRejection::TooSoon {
                    retry_after: MIN_PUBLISH_INTERVAL - elapsed,
                });
            }
        }
        let data = match serde_json::from_str::<serde_json::Value>(&request.data_json) {
            Ok(data @ serde_json::Value::Object(_)) => data,
            _ => return Err(PluginStatusRejection::NotAnObject),
        };
        let entry = PluginStatusEntry {
            generation,
            schema_version,
            data,
            received_at: now,
            received_at_unix_ms: now_unix_ms,
            fresh_for: effective_fresh_for(request.fresh_for_ms),
            connected: true,
            expiry: None,
        };
        let accepted = AcceptedPluginStatus {
            response: proto::PublishPluginStatusResponse {
                received_at_unix_ms: entry.received_at_unix_ms,
                fresh_until_unix_ms: entry.fresh_until_unix_ms(),
            },
            fresh_for: Some(entry.fresh_for),
        };
        self.entries.insert(plugin_name.to_string(), entry);
        Ok(accepted)
    }

    /// Attach the wake-up that fires when `plugin_name`'s snapshot turns stale.
    pub(crate) fn set_expiry(&mut self, plugin_name: &str, expiry: AbortHandle) {
        match self.entries.get_mut(plugin_name) {
            Some(entry) => {
                entry.cancel_expiry();
                entry.expiry = Some(expiry);
            }
            None => expiry.abort(),
        }
    }

    /// The connection `generation` of `plugin_name` is gone: keep what it
    /// published, shown as unavailable. A snapshot from a later connection is
    /// left alone. Returns whether anything changed.
    pub(crate) fn mark_unavailable(&mut self, plugin_name: &str, generation: u64) -> bool {
        match self.entries.get_mut(plugin_name) {
            Some(entry) if entry.generation == generation && entry.connected => {
                entry.connected = false;
                entry.cancel_expiry();
                true
            }
            _ => false,
        }
    }

    /// The plugin stopped for good. Returns whether it had a snapshot.
    pub(crate) fn remove(&mut self, plugin_name: &str) -> bool {
        self.entries.remove(plugin_name).is_some()
    }

    pub(crate) fn payloads(&self, now: Instant) -> BTreeMap<String, PluginStatusPayload> {
        self.entries
            .iter()
            .map(|(name, entry)| {
                (
                    name.clone(),
                    PluginStatusPayload {
                        schema_version: entry.schema_version,
                        state: entry.state(now),
                        received_at_unix_ms: entry.received_at_unix_ms,
                        fresh_until_unix_ms: entry.fresh_until_unix_ms(),
                        data: entry.data.clone(),
                    },
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_plugin::plugin_status::{DEFAULT_FRESH_FOR, MAX_FRESH_FOR};

    fn request(data_json: &str, fresh_for_ms: u64) -> proto::PublishPluginStatusRequest {
        proto::PublishPluginStatusRequest {
            data_json: data_json.into(),
            fresh_for_ms,
        }
    }

    #[test]
    fn a_snapshot_is_ready_then_stale_inside_a_host_owned_envelope() {
        let mut store = PluginStatusStore::default();
        let start = Instant::now();
        let accepted = store
            .publish(
                "powermetrics",
                1,
                3,
                &request(r#"{"idle_watts":8.2}"#, 5_000),
                start,
                1_000,
            )
            .unwrap();
        assert_eq!(accepted.response.received_at_unix_ms, 1_000);
        assert_eq!(accepted.response.fresh_until_unix_ms, 6_000);
        assert_eq!(accepted.fresh_for, Some(Duration::from_secs(5)));

        let ready = &store.payloads(start + Duration::from_secs(4))["powermetrics"];
        assert_eq!(ready.state, PluginStatusState::Ready);
        assert_eq!(ready.schema_version, 3);
        assert_eq!(ready.data, serde_json::json!({"idle_watts": 8.2}));
        assert_eq!(
            store.payloads(start + Duration::from_secs(5))["powermetrics"].state,
            PluginStatusState::Stale
        );
    }

    #[test]
    fn two_plugins_keep_their_own_snapshots() {
        let mut store = PluginStatusStore::default();
        let now = Instant::now();
        store
            .publish("a", 1, 1, &request(r#"{"from":"a"}"#, 0), now, 1)
            .unwrap();
        store
            .publish("b", 1, 1, &request(r#"{"from":"b"}"#, 0), now, 1)
            .unwrap();
        store
            .publish(
                "a",
                1,
                1,
                &request(r#"{"from":"a again"}"#, 0),
                now + MIN_PUBLISH_INTERVAL,
                2,
            )
            .unwrap();
        let payloads = store.payloads(now + MIN_PUBLISH_INTERVAL);
        assert_eq!(payloads["a"].data, serde_json::json!({"from": "a again"}));
        assert_eq!(payloads["b"].data, serde_json::json!({"from": "b"}));
        assert_eq!(
            payloads["b"].fresh_until_unix_ms,
            1 + DEFAULT_FRESH_FOR.as_millis() as u64
        );
    }

    #[test]
    fn malformed_oversized_and_too_frequent_snapshots_are_refused() {
        let mut store = PluginStatusStore::default();
        let now = Instant::now();
        for not_an_object in ["{", "[1]", "8.2", "\"text\"", "null"] {
            assert_eq!(
                store
                    .publish("p", 1, 1, &request(not_an_object, 0), now, 1)
                    .err(),
                Some(PluginStatusRejection::NotAnObject),
                "{not_an_object}"
            );
        }
        let oversized = format!(r#"{{"pad":"{}"}}"#, "x".repeat(MAX_DATA_BYTES));
        assert_eq!(
            store
                .publish("p", 1, 1, &request(&oversized, 0), now, 1)
                .err(),
            Some(PluginStatusRejection::TooLarge {
                bytes: oversized.len()
            })
        );
        assert!(store.payloads(now).is_empty(), "nothing was stored");

        store
            .publish("p", 1, 1, &request(r#"{"n":1}"#, 0), now, 1)
            .unwrap();
        let half = MIN_PUBLISH_INTERVAL / 2;
        assert_eq!(
            store
                .publish("p", 1, 1, &request(r#"{"n":2}"#, 0), now + half, 2)
                .err(),
            Some(PluginStatusRejection::TooSoon {
                retry_after: MIN_PUBLISH_INTERVAL - half
            })
        );
        assert_eq!(store.payloads(now)["p"].data, serde_json::json!({"n": 1}));
        // A new connection's first snapshot is not held to the old one's pace.
        store
            .publish("p", 2, 1, &request(r#"{"n":3}"#, 0), now + half, 3)
            .unwrap();
        assert_eq!(store.payloads(now)["p"].data, serde_json::json!({"n": 3}));
    }

    #[test]
    fn freshness_is_kept_within_the_host_bounds() {
        let mut store = PluginStatusStore::default();
        let accepted = store
            .publish("p", 1, 1, &request("{}", u64::MAX), Instant::now(), 0)
            .unwrap();
        assert_eq!(accepted.fresh_for, Some(MAX_FRESH_FOR));
        assert_eq!(
            accepted.response.fresh_until_unix_ms,
            MAX_FRESH_FOR.as_millis() as u64
        );
    }

    #[test]
    fn a_closed_connection_leaves_its_snapshot_unavailable_but_not_a_later_one() {
        let mut store = PluginStatusStore::default();
        let now = Instant::now();
        store
            .publish("p", 1, 1, &request(r#"{"n":1}"#, 0), now, 1)
            .unwrap();
        assert!(store.mark_unavailable("p", 1));
        assert!(!store.mark_unavailable("p", 1), "already unavailable");
        let unavailable = &store.payloads(now)["p"];
        assert_eq!(unavailable.state, PluginStatusState::Unavailable);
        assert_eq!(unavailable.data, serde_json::json!({"n": 1}));

        // The restarted plugin publishes again; a late notice about the old
        // connection does not touch it.
        store
            .publish("p", 2, 1, &request(r#"{"n":2}"#, 0), now, 2)
            .unwrap();
        assert!(!store.mark_unavailable("p", 1));
        assert_eq!(store.payloads(now)["p"].state, PluginStatusState::Ready);
        assert!(
            !store.mark_unavailable("q", 1),
            "no snapshot, nothing to mark"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_collector_wakes_status_readers_when_a_snapshot_turns_stale() {
        use crate::runtime_data::{RuntimeDataCollector, RuntimeDataDirty};

        let collector = RuntimeDataCollector::new();
        let mut subscription = collector.subscribe();
        collector
            .publish_plugin_status("p", 1, 1, &request("{}", 2_000))
            .unwrap();
        assert!(
            subscription.has_changed().unwrap(),
            "a publication wakes readers"
        );
        subscription.borrow_and_update();
        assert_eq!(
            collector.plugin_status_payloads()["p"].state,
            PluginStatusState::Ready
        );

        // A newer snapshot replaces the old one's wake-up with its own.
        tokio::time::sleep(Duration::from_secs(1)).await;
        collector
            .publish_plugin_status("p", 1, 1, &request("{}", 5_000))
            .unwrap();
        subscription.borrow_and_update();
        tokio::time::sleep(Duration::from_millis(4_999)).await;
        assert!(
            !subscription.has_changed().unwrap(),
            "the replaced snapshot's expiry does not fire"
        );
        assert_eq!(
            collector.plugin_status_payloads()["p"].state,
            PluginStatusState::Ready
        );

        tokio::time::timeout(Duration::from_secs(1), subscription.changed())
            .await
            .expect("readers are woken when the snapshot turns stale")
            .unwrap();
        assert!(
            subscription
                .borrow()
                .dirty
                .contains(RuntimeDataDirty::STATUS)
        );
        assert_eq!(
            collector.plugin_status_payloads()["p"].state,
            PluginStatusState::Stale
        );
    }

    #[test]
    fn a_withdrawal_or_a_stop_removes_the_snapshot() {
        let mut store = PluginStatusStore::default();
        let now = Instant::now();
        store.publish("p", 1, 1, &request("{}", 0), now, 1).unwrap();
        // Withdrawing is not held to the publication pace.
        let withdrawn = store.publish("p", 1, 1, &request("", 0), now, 1).unwrap();
        assert_eq!(
            withdrawn.response,
            proto::PublishPluginStatusResponse::default()
        );
        assert_eq!(withdrawn.fresh_for, None);
        assert!(store.payloads(now).is_empty());

        store.publish("p", 2, 1, &request("{}", 0), now, 1).unwrap();
        assert!(store.remove("p"));
        assert!(!store.remove("p"));
        assert!(store.payloads(now).is_empty());
    }
}
