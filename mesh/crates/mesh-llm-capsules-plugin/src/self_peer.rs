//! This node's own mesh peer id: the sender identity record-push declares.
//!
//! The host reports it on every mesh event (`MeshEvent.local_peer_id`, the
//! full hex endpoint id -- the same form as `served_by_node_id`), starting
//! with the snapshot it sends right after the plugin loads, so the id is
//! known before any exchange can complete. A client node gets a new identity
//! on every start, so the id is learned from the host each run rather than
//! configured once.
//!
//! `CAPSULES_SELF_PEER_ID` still overrides the learned id when set.
//! The learned id is written to `<data dir>/self-peer-id` so an operator (or
//! a harness wiring a peer's `CAPSULES_PEER_KEYS`) can read exactly
//! what this node will declare.

use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock};

pub const ENV_OVERRIDE: &str = "CAPSULES_SELF_PEER_ID";
pub const FILE_NAME: &str = "self-peer-id";

#[derive(Clone)]
pub struct SelfPeer {
    learned: Arc<RwLock<Option<String>>>,
    data_dir: PathBuf,
}

impl SelfPeer {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            learned: Arc::new(RwLock::new(None)),
            data_dir: data_dir.into(),
        }
    }

    /// Records the id the host reported. Returns true when it is new or
    /// changed; an empty id is ignored.
    pub fn learn(&self, local_peer_id: &str) -> bool {
        let id = local_peer_id.trim();
        if id.is_empty() {
            return false;
        }
        {
            let mut learned = self.learned.write().unwrap_or_else(PoisonError::into_inner);
            if learned.as_deref() == Some(id) {
                return false;
            }
            *learned = Some(id.to_string());
        }
        if let Err(error) = write_id(&self.data_dir, id) {
            tracing::warn!(%error, "could not write {FILE_NAME}; the push path still uses the learned id");
        }
        true
    }

    pub fn learned(&self) -> Option<String> {
        self.learned
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The id to declare: the operator override if set, else what the host
    /// reported, else `None` (no push can name its sender).
    pub fn current(&self) -> Option<String> {
        resolve(
            crate::settings::var(ENV_OVERRIDE).ok().as_deref(),
            self.learned(),
        )
    }
}

pub fn resolve(override_id: Option<&str>, learned: Option<String>) -> Option<String> {
    match override_id.map(str::trim) {
        Some(id) if !id.is_empty() => Some(id.to_string()),
        _ => learned,
    }
}

/// Write-then-rename, so a reader never sees a half-written id.
fn write_id(data_dir: &Path, id: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(data_dir)?;
    let tmp = data_dir.join(format!("{FILE_NAME}.tmp"));
    std::fs::write(&tmp, format!("{id}\n"))?;
    std::fs::rename(tmp, data_dir.join(FILE_NAME))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_ID: &str = "5df4d6a4620f3b7c9e1a2d4f6b8c0e1f3a5b7d9c1e3f5a7b9d1c3e5f7a9b1d3c";

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("self-peer-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn learns_the_host_reported_id_and_writes_it_for_operators() {
        let dir = scratch_dir("learn");
        let peer = SelfPeer::new(&dir);
        assert_eq!(peer.learned(), None);

        // MUTANT: make `learn` return false without storing and this goes red.
        assert!(peer.learn(FULL_ID));
        assert_eq!(peer.learned().as_deref(), Some(FULL_ID));
        assert_eq!(
            std::fs::read_to_string(dir.join(FILE_NAME)).expect("self-peer-id written"),
            format!("{FULL_ID}\n")
        );
    }

    #[test]
    fn repeated_empty_and_changed_ids() {
        let dir = scratch_dir("repeat");
        let peer = SelfPeer::new(&dir);
        assert!(!peer.learn("   "), "an empty id is never learned");
        assert!(peer.learn(FULL_ID));
        assert!(!peer.learn(FULL_ID), "the same id is not news");
        assert!(peer.learn("abc123"), "a new identity replaces the old one");
        assert_eq!(peer.learned().as_deref(), Some("abc123"));
        assert_eq!(
            std::fs::read_to_string(dir.join(FILE_NAME)).expect("rewritten"),
            "abc123\n"
        );
    }

    #[test]
    fn the_operator_override_wins_and_a_blank_one_is_ignored() {
        let learned = Some(FULL_ID.to_string());
        assert_eq!(
            resolve(Some("ovr"), learned.clone()).as_deref(),
            Some("ovr")
        );
        assert_eq!(
            resolve(Some("  "), learned.clone()).as_deref(),
            Some(FULL_ID)
        );
        assert_eq!(resolve(None, learned).as_deref(), Some(FULL_ID));
        assert_eq!(resolve(None, None), None);
    }
}
