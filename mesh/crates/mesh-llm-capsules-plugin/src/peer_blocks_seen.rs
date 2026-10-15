//! The peers this node's host has stopped routing to, as the host tells its
//! plugins.
//!
//! The host publishes every block and unblock on `routing.choice.v1` to the
//! local plugins that declare that channel: the operator's own Stop routing
//! and any a plugin requested (`PeerBlockRequest`). This plugin keeps the
//! latest choice per peer in `<ledger>/peer-blocks-seen.json`, and reads it to
//! leave blocked peers out (a referee is never chosen among them, and the
//! stop-routing rule does not ask to block a peer that is already blocked).
//! It never calls the host's operator routes for this.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// The host's channel for routing choices.
pub const ROUTING_CHOICE_CHANNEL: &str = "routing.choice.v1";
const SEEN_FILE: &str = "peer-blocks-seen.json";

/// One routing choice, as the host publishes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Choice {
    /// `block` or `unblock`.
    pub change: String,
    /// The peer's endpoint id, lowercase hex.
    pub peer: String,
    pub at_ms: u64,
    /// `None`: until it is undone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub until_ms: Option<u64>,
    /// `operator`, or `plugin:<id>`.
    pub requested_by: String,
}

fn load(ledger_dir: &Path) -> BTreeMap<String, Choice> {
    std::fs::read_to_string(ledger_dir.join(SEEN_FILE))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Keep `choice_json` (one `routing.choice.v1` body) as the latest choice for
/// its peer. A later choice replaces an earlier one; an older one is ignored.
pub fn record(ledger_dir: &Path, choice_json: &[u8]) -> anyhow::Result<Choice> {
    let choice: Choice = serde_json::from_slice(choice_json)?;
    anyhow::ensure!(
        matches!(choice.change.as_str(), "block" | "unblock"),
        "not a routing choice: change is {:?}",
        choice.change
    );
    let mut seen = load(ledger_dir);
    let peer = choice.peer.to_ascii_lowercase();
    if seen
        .get(&peer)
        .is_some_and(|known| known.at_ms > choice.at_ms)
    {
        return Ok(choice);
    }
    seen.insert(peer, choice.clone());
    let tmp = ledger_dir.join(format!("{SEEN_FILE}.tmp"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&seen)?)?;
    std::fs::rename(&tmp, ledger_dir.join(SEEN_FILE))?;
    Ok(choice)
}

/// The peers blocked at `now_ms`: the latest choice is a block that has not
/// lapsed.
pub fn blocked_at(ledger_dir: &Path, now_ms: u64) -> Vec<String> {
    load(ledger_dir)
        .into_iter()
        .filter(|(_, c)| c.change == "block" && c.until_ms.is_none_or(|until| until > now_ms))
        .map(|(peer, _)| peer)
        .collect()
}

/// [`blocked_at`] now.
pub fn blocked_now(ledger_dir: &Path) -> Vec<String> {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    blocked_at(ledger_dir, now_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn choice(change: &str, peer: &str, at_ms: u64, until_ms: Option<u64>) -> Vec<u8> {
        let mut c =
            json!({"change": change, "peer": peer, "at_ms": at_ms, "requested_by": "operator"});
        if let Some(until) = until_ms {
            c["until_ms"] = json!(until);
        }
        serde_json::to_vec(&c).unwrap()
    }

    #[test]
    fn the_latest_choice_per_peer_decides_and_a_lapsed_block_is_not_a_block() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b, c) = ("a".repeat(64), "b".repeat(64), "c".repeat(64));
        record(dir.path(), &choice("block", &a, 10, None)).unwrap();
        record(dir.path(), &choice("block", &b, 10, Some(500))).unwrap();
        record(dir.path(), &choice("block", &c, 10, None)).unwrap();
        record(dir.path(), &choice("unblock", &c, 20, None)).unwrap();
        // An older choice arriving late does not undo a newer one.
        record(dir.path(), &choice("unblock", &a, 5, None)).unwrap();
        assert_eq!(blocked_at(dir.path(), 100), vec![a.clone(), b.clone()]);
        assert_eq!(blocked_at(dir.path(), 600), vec![a]);
    }

    #[test]
    fn anything_but_a_routing_choice_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        assert!(record(dir.path(), b"{}").is_err());
        assert!(record(dir.path(), &choice("route", &"a".repeat(64), 1, None)).is_err());
        assert!(blocked_now(dir.path()).is_empty());
    }
}
