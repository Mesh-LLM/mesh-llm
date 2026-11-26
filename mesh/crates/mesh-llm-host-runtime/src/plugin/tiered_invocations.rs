//! The tier of the requests each plugin is handling.
//!
//! A plugin calls back into the host for its own inference on a different
//! task from the request it is handling, so the request's tier does not
//! follow it (see [`crate::mesh::request_tier`]). While a plugin handles a
//! request below `local` (a virtual-model turn, an OpenAI exchange), the host
//! serves every nested call from that plugin at the lowest such tier.
//! Like [`super::private_invocations`] this is per plugin, not per request: a
//! `local` request handled by the same plugin at the same moment is lowered
//! too, never the other way round, so no peer gains a tier through a plugin.
//!
//! A plugin keeps working on a call the host sends no cancel for, so the
//! binding lasts as the private one does: to the plugin's reply
//! ([`run_to_reply`]), and for [`ABANDONED_LINGER`] after a call that ended
//! without one.

use super::private_invocations::ABANDONED_LINGER;
use skippy_inference_api::InferenceRequestTier;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct State {
    next_id: u64,
    /// Per plugin: each running invocation's tier, by invocation id.
    active: BTreeMap<String, BTreeMap<u64, InferenceRequestTier>>,
    /// Per plugin: abandoned invocations' tiers, until they stop lingering.
    lingering: BTreeMap<String, Vec<(Instant, InferenceRequestTier)>>,
}

impl State {
    /// Forget `plugin`'s lingering tiers that have expired.
    fn reap(&mut self, plugin: &str, now: Instant) {
        if let Some(lingering) = self.lingering.get_mut(plugin) {
            lingering.retain(|(until, _)| *until > now);
            if lingering.is_empty() {
                self.lingering.remove(plugin);
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct TieredInvocations {
    state: Arc<Mutex<State>>,
    linger: Duration,
}

impl Default for TieredInvocations {
    fn default() -> Self {
        Self {
            state: Arc::default(),
            linger: ABANDONED_LINGER,
        }
    }
}

/// Holds a plugin's nested calls at or below its tier until dropped. Call
/// [`finish`](Self::finish) once the plugin has replied; dropping it
/// unfinished keeps the tier for the linger period.
pub(crate) struct TieredInvocationGuard {
    invocations: TieredInvocations,
    plugin: String,
    id: u64,
    finished: bool,
}

impl TieredInvocations {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Record that `plugin` handles a request at `tier`. `None` for a
    /// `local` request, which lowers nothing.
    pub(crate) fn begin(
        &self,
        plugin: &str,
        tier: InferenceRequestTier,
    ) -> Option<TieredInvocationGuard> {
        if tier == InferenceRequestTier::Local {
            return None;
        }
        let mut state = self.lock();
        state.reap(plugin, Instant::now());
        let id = state.next_id;
        state.next_id += 1;
        state
            .active
            .entry(plugin.to_string())
            .or_default()
            .insert(id, tier);
        Some(TieredInvocationGuard {
            invocations: self.clone(),
            plugin: plugin.to_string(),
            id,
            finished: false,
        })
    }

    /// The tier `plugin`'s nested inference runs at: the lowest of the
    /// requests it is handling or has abandoned, `local` when there are none.
    pub(crate) fn nested_tier(&self, plugin: &str) -> InferenceRequestTier {
        let mut state = self.lock();
        state.reap(plugin, Instant::now());
        let active = state
            .active
            .get(plugin)
            .into_iter()
            .flat_map(|active| active.values().copied());
        let lingering = state
            .lingering
            .get(plugin)
            .into_iter()
            .flat_map(|lingering| lingering.iter().map(|(_, tier)| *tier));
        active
            .chain(lingering)
            .min()
            .unwrap_or(InferenceRequestTier::Local)
    }
}

/// Run `call`, a plugin invocation, holding `guard` until the plugin replies
/// (`Ok`) or the call ends without a reply (`Err`, which lingers). With a
/// guard, the call runs on its own task to completion, so a caller that
/// stops waiting does not cut the binding short while the plugin works on.
pub(crate) async fn run_to_reply<T, F>(
    guard: Option<TieredInvocationGuard>,
    call: F,
) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: std::future::Future<Output = anyhow::Result<T>> + Send + 'static,
{
    let Some(guard) = guard else {
        return call.await;
    };
    tokio::spawn(async move {
        let reply = call.await;
        if reply.is_ok() {
            guard.finish();
        }
        reply
    })
    .await
    .unwrap_or_else(|error| Err(anyhow::anyhow!("tiered plugin call failed: {error}")))
}

impl TieredInvocationGuard {
    /// The plugin replied: this invocation no longer needs to linger.
    pub(crate) fn finish(mut self) {
        self.finished = true;
    }
}

impl Drop for TieredInvocationGuard {
    fn drop(&mut self) {
        let linger = self.invocations.linger;
        let mut state = self.invocations.lock();
        let tier = state
            .active
            .get_mut(&self.plugin)
            .and_then(|active| active.remove(&self.id));
        if state
            .active
            .get(&self.plugin)
            .is_some_and(BTreeMap::is_empty)
        {
            state.active.remove(&self.plugin);
        }
        if !self.finished
            && let Some(tier) = tier
        {
            state
                .lingering
                .entry(self.plugin.clone())
                .or_default()
                .push((Instant::now() + linger, tier));
        }
        // Expired entries go even if the plugin never calls back.
        state.reap(&self.plugin, Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_calls_run_at_the_lowest_running_tier() {
        let invocations = TieredInvocations::default();
        assert_eq!(invocations.nested_tier("p"), InferenceRequestTier::Local);
        assert!(
            invocations
                .begin("p", InferenceRequestTier::Local)
                .is_none()
        );

        let trusted = invocations.begin("p", InferenceRequestTier::Trusted);
        assert_eq!(invocations.nested_tier("p"), InferenceRequestTier::Trusted);
        let mesh = invocations.begin("p", InferenceRequestTier::Mesh);
        let second_mesh = invocations.begin("p", InferenceRequestTier::Mesh);
        assert_eq!(invocations.nested_tier("p"), InferenceRequestTier::Mesh);
        assert_eq!(
            invocations.nested_tier("other"),
            InferenceRequestTier::Local
        );

        mesh.unwrap().finish();
        assert_eq!(invocations.nested_tier("p"), InferenceRequestTier::Mesh);
        second_mesh.unwrap().finish();
        assert_eq!(invocations.nested_tier("p"), InferenceRequestTier::Trusted);
        trusted.unwrap().finish();
        assert_eq!(invocations.nested_tier("p"), InferenceRequestTier::Local);
        let state = invocations.lock();
        assert!(state.active.is_empty() && state.lingering.is_empty());
    }

    #[test]
    fn an_abandoned_invocation_keeps_its_tier_until_the_linger_ends() {
        let lingering = TieredInvocations::default();
        drop(lingering.begin("p", InferenceRequestTier::Mesh));
        assert_eq!(lingering.nested_tier("p"), InferenceRequestTier::Mesh);

        let expired = TieredInvocations {
            linger: Duration::ZERO,
            ..TieredInvocations::default()
        };
        drop(expired.begin("p", InferenceRequestTier::Mesh));
        assert_eq!(expired.nested_tier("p"), InferenceRequestTier::Local);
        assert!(expired.lock().lingering.is_empty());
    }

    #[tokio::test]
    async fn a_caller_that_stops_waiting_does_not_end_the_binding() {
        let invocations = TieredInvocations::default();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let guard = invocations.begin("p", InferenceRequestTier::Mesh);
        let call = tokio::spawn(run_to_reply(guard, async move {
            let _ = started_tx.send(());
            let _ = release_rx.await;
            Ok(())
        }));
        // The plugin is working; the caller then stops waiting.
        started_rx.await.unwrap();
        call.abort();
        let _ = call.await;
        assert_eq!(invocations.nested_tier("p"), InferenceRequestTier::Mesh);

        release_tx.send(()).unwrap();
        for _ in 0..200 {
            if invocations.nested_tier("p") == InferenceRequestTier::Local {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(invocations.nested_tier("p"), InferenceRequestTier::Local);
    }
}
