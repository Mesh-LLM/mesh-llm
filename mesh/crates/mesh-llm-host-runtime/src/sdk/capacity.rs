//! Capacity observations for local embedded runtimes, without HTTP or inference.

use super::EmbeddedServeHandle;
use crate::runtime_data::{RuntimeDataCollector, RuntimeLlamaEndpointStatus};
use anyhow::Result;

/// A complete observation across the local instances of one exact serving ID.
/// Busy lanes count toward capacity; this is not a count of free reservations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneCapacity {
    pub execution_lanes: u64,
    /// The smallest lane context, safe for a request assigned to any lane.
    pub context_tokens_per_lane: u64,
    /// Sum of all lane contexts, not the per-request context limit.
    pub context_pool_tokens: u64,
    /// Oldest lane observation. Long inference can delay runtime publication;
    /// callers choose their own freshness policy using this timestamp.
    pub captured_at_unix_ms: u64,
}

impl EmbeddedServeHandle {
    /// Read the latest published capacity for a local model in process.
    ///
    /// Returns `None` for an unknown/unsupported model, or if any matching
    /// instance is unavailable, has no lanes, or has incomplete/overflowing
    /// context data. Every call reads current shared state; no capacity is
    /// inferred from configuration or peer advertisements.
    ///
    /// # Errors
    /// Returns an error after runtime shutdown or when the native node is absent.
    pub fn runtime_capacity(&self, model: &str) -> Result<Option<LaneCapacity>> {
        let collector = self.native_node()?.runtime_data_collector();
        Ok(snapshot(&collector, model))
    }
}

fn snapshot(collector: &RuntimeDataCollector, model: &str) -> Option<LaneCapacity> {
    collector.with_runtime_status(|status| capacity_from_status(status, model))
}

fn capacity_from_status(
    status: &crate::runtime_data::RuntimeStatusSnapshot,
    model: &str,
) -> Option<LaneCapacity> {
    let mut capacity = LaneCapacity {
        execution_lanes: 0,
        context_tokens_per_lane: u64::MAX,
        context_pool_tokens: 0,
        captured_at_unix_ms: u64::MAX,
    };
    // Require a fresh observation for every active instance.
    for (instance, _) in status
        .active_runtime_instances
        .iter()
        .filter(|(_, name)| *name == model)
    {
        let slots = status
            .llama_runtime_by_instance
            .get(instance)
            .map(|runtime| &runtime.slots)?;
        if slots.model.as_deref() != Some(model)
            || slots.instance_id.as_ref().is_some_and(|id| id != instance)
        {
            return None;
        }
        if slots.status != RuntimeLlamaEndpointStatus::Ready || slots.slots.is_empty() {
            return None;
        }
        capacity.captured_at_unix_ms = capacity
            .captured_at_unix_ms
            .min(slots.last_success_unix_ms?);
        capacity.execution_lanes = capacity
            .execution_lanes
            .checked_add(u64::try_from(slots.slots.len()).ok()?)?;
        for lane in &slots.slots {
            let context = lane.n_ctx.filter(|context| *context > 0)?;
            capacity.context_tokens_per_lane = capacity.context_tokens_per_lane.min(context);
            capacity.context_pool_tokens = capacity.context_pool_tokens.checked_add(context)?;
        }
    }
    (capacity.execution_lanes > 0).then_some(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_data::{RuntimeLlamaSlotSnapshot, RuntimeLlamaSlotsSnapshot};

    fn observation(
        model: &str,
        instance: &str,
        contexts: &[u64],
        captured: u64,
    ) -> RuntimeLlamaSlotsSnapshot {
        RuntimeLlamaSlotsSnapshot {
            model: Some(model.into()),
            instance_id: Some(instance.into()),
            status: RuntimeLlamaEndpointStatus::Ready,
            last_success_unix_ms: Some(captured),
            slots: contexts
                .iter()
                .map(|context| RuntimeLlamaSlotSnapshot {
                    n_ctx: Some(*context),
                    is_processing: Some(true),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn capacity_tracks_instance_changes_and_keeps_model_budgets_separate() {
        let collector = RuntimeDataCollector::new();
        collector.register_runtime_instance("chat", "a");
        collector.register_runtime_instance("chat", "b");
        collector.register_runtime_instance("other", "c");
        collector.replace_active_llama_slots_snapshot(observation(
            "chat",
            "a",
            &[8192, 16384],
            100,
        ));
        collector.replace_active_llama_slots_snapshot(observation("chat", "b", &[4096], 200));
        collector.replace_active_llama_slots_snapshot(observation("other", "c", &[32768], 300));
        assert_eq!(
            snapshot(&collector, "chat"),
            Some(LaneCapacity {
                execution_lanes: 3,
                context_tokens_per_lane: 4096,
                context_pool_tokens: 28672,
                captured_at_unix_ms: 100,
            })
        );
        collector.replace_active_llama_slots_snapshot(observation("chat", "a", &[8192], 400));
        assert_eq!(snapshot(&collector, "chat").unwrap().execution_lanes, 2);
        assert_eq!(
            snapshot(&collector, "chat").unwrap().context_pool_tokens,
            12288
        );
        assert_eq!(
            snapshot(&collector, "chat").unwrap().captured_at_unix_ms,
            200
        );
        let mut stopped = observation("chat", "b", &[], 500);
        stopped.status = RuntimeLlamaEndpointStatus::Unavailable;
        collector.replace_active_llama_slots_snapshot(stopped);
        assert_eq!(snapshot(&collector, "chat"), None);
        assert!(snapshot(&collector, "other").is_some());
    }

    #[test]
    fn unknown_incomplete_and_overflowing_observations_have_no_capacity() {
        assert_eq!(snapshot(&RuntimeDataCollector::new(), "unknown"), None);
        let valid = observation("chat", "a", &[8192], 100);
        let mut invalid = Vec::new();
        let mut missing_time = valid.clone();
        missing_time.last_success_unix_ms = None;
        invalid.push(missing_time);
        let mut missing_context = valid.clone();
        missing_context.slots[0].n_ctx = None;
        invalid.push(missing_context);
        invalid.push(observation("chat", "a", &[], 100));
        invalid.push(observation("chat", "a", &[0], 100));
        invalid.push(observation("chat", "a", &[u64::MAX, 1], 100));
        for slots in invalid {
            let collector = RuntimeDataCollector::new();
            collector.register_runtime_instance("chat", "a");
            collector.replace_active_llama_slots_snapshot(slots);
            assert_eq!(snapshot(&collector, "chat"), None);
        }
    }

    #[test]
    fn retired_instances_do_not_count_or_block_a_later_load() {
        let collector = RuntimeDataCollector::new();
        collector.register_runtime_instance("chat", "a");
        collector.register_runtime_instance("chat", "b");
        collector.replace_active_llama_slots_snapshot(observation("chat", "a", &[8192], 100));
        collector.replace_active_llama_slots_snapshot(observation("chat", "b", &[4096], 200));
        collector.unregister_runtime_instance("a");
        assert!(
            !collector
                .runtime_llama_snapshots_by_instance()
                .contains_key("a")
        );
        assert_eq!(
            collector
                .runtime_llama_snapshot()
                .slots
                .instance_id
                .as_deref(),
            Some("b")
        );
        assert_eq!(
            snapshot(&collector, "chat"),
            Some(LaneCapacity {
                execution_lanes: 1,
                context_tokens_per_lane: 4096,
                context_pool_tokens: 4096,
                captured_at_unix_ms: 200,
            })
        );
        collector.unregister_runtime_instance("b");
        assert_eq!(snapshot(&collector, "chat"), None);
        assert_eq!(collector.runtime_llama_snapshot().items.slots_total, 0);
        assert!(collector.runtime_llama_snapshots_by_instance().is_empty());
        collector.register_runtime_instance("chat", "new");
        assert_eq!(snapshot(&collector, "chat"), None);
        collector.replace_active_llama_slots_snapshot(observation("chat", "new", &[16384], 300));
        let mut late_unavailable = observation("chat", "b", &[], 400);
        late_unavailable.status = RuntimeLlamaEndpointStatus::Unavailable;
        assert!(!collector.replace_active_llama_slots_snapshot(late_unavailable));
        assert!(
            !collector
                .runtime_llama_snapshots_by_instance()
                .contains_key("b")
        );
        assert_eq!(
            snapshot(&collector, "chat"),
            Some(LaneCapacity {
                execution_lanes: 1,
                context_tokens_per_lane: 16384,
                context_pool_tokens: 16384,
                captured_at_unix_ms: 300,
            })
        );
        collector.register_runtime_instance("chat", "pending");
        assert_eq!(snapshot(&collector, "chat"), None);
        collector.unregister_runtime_instance("pending");
        collector.unregister_runtime_instance("new");
        collector.register_runtime_instance("chat", "new");
        assert_eq!(snapshot(&collector, "chat"), None);
    }

    #[test]
    fn absent_or_stopping_runtime_is_an_error_instead_of_cached_capacity() {
        let (shutdown, _) = tokio::sync::watch::channel(false);
        let handle = EmbeddedServeHandle {
            api_base_url: String::new(),
            console_url: String::new(),
            invite_token: None,
            control_tx: None,
            task: None,
            shutdown,
            native_node: None,
            exit: tokio::sync::watch::channel(None).1,
        };
        assert!(
            handle
                .runtime_capacity("chat")
                .unwrap_err()
                .to_string()
                .contains("unavailable")
        );
        handle.shutdown.send_replace(true);
        assert!(
            handle
                .runtime_capacity("chat")
                .unwrap_err()
                .to_string()
                .contains("shutting down")
        );
    }
}
