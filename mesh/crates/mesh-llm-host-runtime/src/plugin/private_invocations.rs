//! Which plugins are handling a private request.
//!
//! A plugin calls back into the host for its own inference
//! (`inference/chat_completions`) on a different task from the request it is
//! handling, so the request's owner scope does not follow it. While any
//! private request is being handled by a plugin (a virtual-model turn, an
//! OpenAI exchange hook, a request transform), the host marks every nested
//! call from that plugin private. Fails closed: a public request handled by
//! the same plugin at the same moment is narrowed to the trust circle too,
//! never the other way round.
//!
//! A plugin can keep working after the host stops waiting for it: the client
//! went away, or the host's deadline passed. The host sends no cancel, so an
//! invocation that ends without the plugin's reply keeps the plugin private
//! for [`ABANDONED_LINGER`] after it.
// shortcut: per plugin, not per request. A per-request token would need the
// plugin protocol to echo it back on nested calls.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a plugin stays private after a private invocation ended without
/// its reply: the longest a virtual-model turn may run.
pub(crate) const ABANDONED_LINGER: Duration = Duration::from_secs(300);

#[derive(Default)]
struct State {
    active: BTreeMap<String, usize>,
    lingering: BTreeMap<String, Instant>,
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
    finished: bool,
}

impl PrivateInvocations {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn begin(&self, plugin: &str) -> PrivateInvocationGuard {
        *self.lock().active.entry(plugin.to_string()).or_default() += 1;
        PrivateInvocationGuard {
            invocations: self.clone(),
            plugin: plugin.to_string(),
            finished: false,
        }
    }

    pub(crate) fn is_active(&self, plugin: &str) -> bool {
        let mut state = self.lock();
        if state.active.get(plugin).is_some_and(|count| *count > 0) {
            return true;
        }
        match state.lingering.get(plugin) {
            Some(until) if *until > Instant::now() => true,
            Some(_) => {
                state.lingering.remove(plugin);
                false
            }
            None => false,
        }
    }
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
        if let Some(count) = state.active.get_mut(&self.plugin) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                state.active.remove(&self.plugin);
            }
        }
        if !self.finished {
            let until = Instant::now() + linger;
            let entry = state.lingering.entry(self.plugin.clone()).or_insert(until);
            *entry = (*entry).max(until);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plugin_stays_private_until_its_last_private_invocation_ends() {
        let invocations = PrivateInvocations::default();
        assert!(!invocations.is_active("moa"));
        let first = invocations.begin("moa");
        let second = invocations.begin("moa");
        assert!(invocations.is_active("moa"));
        assert!(!invocations.is_active("other"));
        first.finish();
        assert!(invocations.is_active("moa"));
        second.finish();
        assert!(!invocations.is_active("moa"));
    }

    #[test]
    fn an_abandoned_invocation_keeps_the_plugin_private_for_a_while() {
        let invocations = PrivateInvocations::default();
        drop(invocations.begin("moa"));
        assert!(invocations.is_active("moa"));
        assert!(!invocations.is_active("other"));

        let short = PrivateInvocations {
            linger: Duration::ZERO,
            ..PrivateInvocations::default()
        };
        drop(short.begin("moa"));
        assert!(!short.is_active("moa"));
    }
}
