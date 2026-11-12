//! Which plugins are running a virtual-model turn for a private request.
//!
//! A virtual model's plugin calls back into the host for its own inference
//! (`inference/chat_completions`), on a different task from the request that
//! started the turn, so the request's owner scope does not follow it. While
//! any private turn of a plugin is running, the host marks every nested call
//! from that plugin private. Fails closed: a public turn of the same plugin
//! that overlaps a private one is narrowed to your own nodes too, never the
//! other way round.
// shortcut: per plugin, not per turn. A per-turn token would need the plugin
// protocol to echo it back on nested calls.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub(crate) struct PrivateInvocations {
    active: Arc<Mutex<BTreeMap<String, usize>>>,
}

/// Keeps a plugin's nested calls private until dropped.
pub(crate) struct PrivateInvocationGuard {
    active: Arc<Mutex<BTreeMap<String, usize>>>,
    plugin: String,
}

impl PrivateInvocations {
    pub(crate) fn begin(&self, plugin: &str) -> PrivateInvocationGuard {
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        *active.entry(plugin.to_string()).or_default() += 1;
        PrivateInvocationGuard {
            active: self.active.clone(),
            plugin: plugin.to_string(),
        }
    }

    pub(crate) fn is_active(&self, plugin: &str) -> bool {
        let active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        active.get(plugin).is_some_and(|count| *count > 0)
    }
}

impl Drop for PrivateInvocationGuard {
    fn drop(&mut self) {
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(count) = active.get_mut(&self.plugin) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                active.remove(&self.plugin);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plugin_stays_private_until_its_last_private_turn_ends() {
        let invocations = PrivateInvocations::default();
        assert!(!invocations.is_active("moa"));
        let first = invocations.begin("moa");
        let second = invocations.begin("moa");
        assert!(invocations.is_active("moa"));
        assert!(!invocations.is_active("other"));
        drop(first);
        assert!(invocations.is_active("moa"));
        drop(second);
        assert!(!invocations.is_active("moa"));
    }
}
