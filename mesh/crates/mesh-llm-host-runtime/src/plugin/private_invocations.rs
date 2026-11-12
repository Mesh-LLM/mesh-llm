//! Which plugins are handling a private request, and for which owners.
//!
//! A plugin calls back into the host for its own inference
//! (`inference/chat_completions`) on a different task from the request it is
//! handling, so the request's owner scope does not follow it. While any
//! private request is being handled by a plugin (a virtual-model turn, an
//! OpenAI exchange hook, a request transform), the host marks every nested
//! call from that plugin private, limited to the owners of every such
//! request at once. Fails closed: a public request handled by the same plugin
//! at the same moment is narrowed too, never the other way round.
//!
//! The limit keeps a nested call inside the circle of the request that
//! caused it. Without it the node running the plugin would scope the call
//! to its own trust circle, which for a request forwarded from another
//! owner's node is not the circle that request was given.
//!
//! A plugin keeps working on a call the host sends no cancel for. So a
//! private call runs to the plugin's reply on its own task
//! ([`run_to_reply`]): a client that goes away does not end the binding.
//! A call that ends without a reply keeps the plugin private, with the same
//! limit, for [`ABANDONED_LINGER`] after it. A virtual-model turn or request
//! transform that times out restarts an external plugin, which ends its
//! work. An exchange hook is not restarted at its deadline; its binding is
//! held until the plugin replies or disconnects instead, and the hook is
//! restarted if it has done neither [`ABANDONED_LINGER`] past the deadline.
// shortcut: per plugin, not per request. A per-request token would need the
// plugin protocol to echo it back on nested calls.

use crate::mesh::OwnerScope;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a plugin stays private after a private invocation ended without
/// its reply: the longest a virtual-model turn may run.
pub(crate) const ABANDONED_LINGER: Duration = Duration::from_secs(300);

/// How a plugin's nested inference must be scoped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NestedScope {
    /// Private, in this node's own trust circle: the request it handles
    /// entered here and has not been scoped yet (a request transform).
    NodeCircle,
    /// Private, and only these owners' nodes.
    Owners(OwnerScope),
}

impl NestedScope {
    fn narrowed_by(self, limit: Option<&OwnerScope>) -> Self {
        match (self, limit) {
            (scope, None) => scope,
            (Self::NodeCircle, Some(limit)) => Self::Owners(limit.clone()),
            (Self::Owners(owners), Some(limit)) => Self::Owners(owners.intersect(limit)),
        }
    }
}

#[derive(Default)]
struct State {
    next_id: u64,
    /// Per plugin: each running invocation's owner limit, by invocation id.
    active: BTreeMap<String, BTreeMap<u64, Option<OwnerScope>>>,
    /// Per plugin: abandoned invocations' limits, until they stop lingering.
    lingering: BTreeMap<String, Vec<(Instant, Option<OwnerScope>)>>,
}

impl State {
    /// Forget `plugin`'s lingering limits that have expired.
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
pub(crate) struct PrivateInvocations {
    state: Arc<Mutex<State>>,
    linger: Duration,
}

impl Default for PrivateInvocations {
    fn default() -> Self {
        Self {
            state: Arc::default(),
            linger: ABANDONED_LINGER,
        }
    }
}

/// Keeps a plugin's nested calls private until dropped. Call
/// [`finish`](Self::finish) once the plugin has replied; dropping it
/// unfinished keeps the plugin private for the linger period.
pub(crate) struct PrivateInvocationGuard {
    invocations: PrivateInvocations,
    plugin: String,
    id: u64,
    finished: bool,
}

impl PrivateInvocations {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `owners` limits the plugin's nested calls to those owners' nodes;
    /// `None` leaves them to this node's trust circle.
    pub(crate) fn begin(&self, plugin: &str, owners: Option<OwnerScope>) -> PrivateInvocationGuard {
        let mut state = self.lock();
        state.reap(plugin, Instant::now());
        let id = state.next_id;
        state.next_id += 1;
        state
            .active
            .entry(plugin.to_string())
            .or_default()
            .insert(id, owners);
        PrivateInvocationGuard {
            invocations: self.clone(),
            plugin: plugin.to_string(),
            id,
            finished: false,
        }
    }

    /// The scope `plugin`'s nested inference must keep to, if it must be
    /// private: the intersection of every limit in force for it.
    pub(crate) fn nested_scope(&self, plugin: &str) -> Option<NestedScope> {
        let mut state = self.lock();
        state.reap(plugin, Instant::now());
        let active = state
            .active
            .get(plugin)
            .into_iter()
            .flat_map(|a| a.values());
        let lingering = state
            .lingering
            .get(plugin)
            .into_iter()
            .flat_map(|l| l.iter().map(|(_, owners)| owners));
        let mut scope = None;
        for owners in active.chain(lingering) {
            scope = Some(
                scope
                    .unwrap_or(NestedScope::NodeCircle)
                    .narrowed_by(owners.as_ref()),
            );
        }
        scope
    }
}

/// Run `call`, a plugin invocation, holding `guard` until the plugin replies
/// (`Ok`) or the call ends without a reply (`Err`, which lingers). With a
/// guard, the call runs on its own task to completion, so a caller that
/// stops waiting does not cut the binding short while the plugin works on.
pub(crate) async fn run_to_reply<T, F>(
    guard: Option<PrivateInvocationGuard>,
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
    .unwrap_or_else(|error| Err(anyhow::anyhow!("private plugin call failed: {error}")))
}

impl PrivateInvocationGuard {
    /// The plugin replied: this invocation no longer needs to linger.
    pub(crate) fn finish(mut self) {
        self.finished = true;
    }
}

impl Drop for PrivateInvocationGuard {
    fn drop(&mut self) {
        let linger = self.invocations.linger;
        let mut state = self.invocations.lock();
        let owners = state
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
            && let Some(owners) = owners
        {
            state
                .lingering
                .entry(self.plugin.clone())
                .or_default()
                .push((Instant::now() + linger, owners));
        }
        // Expired entries go even if the plugin never calls back.
        state.reap(&self.plugin, Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plugin_stays_private_until_its_last_private_invocation_ends() {
        let invocations = PrivateInvocations::default();
        assert_eq!(invocations.nested_scope("moa"), None);
        let first = invocations.begin("moa", None);
        let second = invocations.begin("moa", None);
        assert_eq!(
            invocations.nested_scope("moa"),
            Some(NestedScope::NodeCircle)
        );
        assert_eq!(invocations.nested_scope("other"), None);
        first.finish();
        assert!(invocations.nested_scope("moa").is_some());
        second.finish();
        assert_eq!(invocations.nested_scope("moa"), None);
    }

    #[test]
    fn nested_calls_keep_to_the_owners_of_every_request_in_progress() {
        let invocations = PrivateInvocations::default();
        let ab = OwnerScope::of(["a".to_string(), "b".to_string()]);
        let bc = OwnerScope::of(["b".to_string(), "c".to_string()]);
        let first = invocations.begin("moa", Some(ab.clone()));
        assert_eq!(
            invocations.nested_scope("moa"),
            Some(NestedScope::Owners(ab.clone()))
        );
        let second = invocations.begin("moa", Some(bc));
        let unscoped = invocations.begin("moa", None);
        assert_eq!(
            invocations.nested_scope("moa"),
            Some(NestedScope::Owners(OwnerScope::new("b")))
        );
        second.finish();
        unscoped.finish();
        assert_eq!(
            invocations.nested_scope("moa"),
            Some(NestedScope::Owners(ab))
        );
        first.finish();
        assert_eq!(invocations.nested_scope("moa"), None);
    }

    #[test]
    fn an_abandoned_invocation_keeps_the_plugin_private_for_a_while() {
        let invocations = PrivateInvocations::default();
        drop(invocations.begin("moa", Some(OwnerScope::new("a"))));
        assert_eq!(
            invocations.nested_scope("moa"),
            Some(NestedScope::Owners(OwnerScope::new("a")))
        );
        assert_eq!(invocations.nested_scope("other"), None);

        let short = PrivateInvocations {
            linger: Duration::ZERO,
            ..PrivateInvocations::default()
        };
        drop(short.begin("moa", None));
        assert_eq!(short.nested_scope("moa"), None);
    }

    #[test]
    fn expired_lingers_are_reaped_without_a_nested_call() {
        let invocations = PrivateInvocations {
            linger: Duration::ZERO,
            ..PrivateInvocations::default()
        };
        for _ in 0..3 {
            drop(invocations.begin("moa", None));
        }
        assert!(invocations.lock().lingering.is_empty());
        assert!(invocations.lock().active.is_empty());
    }

    #[tokio::test]
    async fn a_caller_that_stops_waiting_does_not_end_the_binding() {
        // No linger: only the running call keeps the plugin private.
        let invocations = PrivateInvocations {
            linger: Duration::ZERO,
            ..PrivateInvocations::default()
        };
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<()>();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let call = async move {
            let _ = started_tx.send(());
            reply_rx.await.ok();
            let _ = done_tx.send(());
            Ok(())
        };
        let waiting = tokio::spawn(run_to_reply(Some(invocations.begin("moa", None)), call));
        // The plugin call is running before the host stops waiting.
        started_rx.await.unwrap();
        // The client went away: the host stops waiting.
        waiting.abort();
        let _ = waiting.await;
        assert_eq!(
            invocations.nested_scope("moa"),
            Some(NestedScope::NodeCircle)
        );
        // The plugin replies later; only then is it public again.
        reply_tx.send(()).unwrap();
        done_rx.await.unwrap();
        for _ in 0..100 {
            if invocations.nested_scope("moa").is_none() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("the plugin stayed private after it replied");
    }

    #[tokio::test]
    async fn a_call_past_its_deadline_stays_private_until_it_replies() {
        // As an exchange hook: the host gives up at the deadline, the plugin
        // is not restarted and replies later. No linger, so only the held
        // call keeps the plugin private.
        let invocations = PrivateInvocations {
            linger: Duration::ZERO,
            ..PrivateInvocations::default()
        };
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<()>();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let call = async move {
            reply_rx.await.ok();
            let _ = done_tx.send(());
            Ok(())
        };
        let held = run_to_reply(Some(invocations.begin("hook", None)), call);
        let deadline = tokio::time::Instant::now() + Duration::from_millis(20);
        assert!(tokio::time::timeout_at(deadline, held).await.is_err());
        assert_eq!(
            invocations.nested_scope("hook"),
            Some(NestedScope::NodeCircle)
        );
        reply_tx.send(()).unwrap();
        done_rx.await.unwrap();
        for _ in 0..100 {
            if invocations.nested_scope("hook").is_none() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("the plugin stayed private after it replied");
    }
}
