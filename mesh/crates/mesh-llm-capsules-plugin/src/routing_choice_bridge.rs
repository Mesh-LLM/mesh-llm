//! `mesh_local_routing_choice`: a host's block store may ask this plugin to
//! seal a record of a local routing choice (block or unblock a peer) that its
//! operator made. A block the opt-in operator rule asked for is sealed by the
//! rule itself (`routing_rule`), so a record sealed here names no rule.
//!
//! The host owns the choice and enforces it in its router; this plugin only
//! seals the record onto the node's one chain. The record names the peer by a
//! salted commitment; the host chooses the salt, keeps it in its local store,
//! and checks the returned commitment against its own
//! (see `crate::producer::capsule::seal_local_routing_choice`).

use std::sync::Arc;

use crate::producer::capsule::{RoutingChoiceChange, RoutingRuleCitation};
use mesh_llm_plugin::{PluginError, PluginResult};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::capsule_emit::CapsuleState;

pub const LOCAL_ROUTING_CHOICE_OPERATION: &str = "mesh_local_routing_choice";

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ChangeArg {
    Block,
    Unblock,
}

impl From<ChangeArg> for RoutingChoiceChange {
    fn from(change: ChangeArg) -> Self {
        match change {
            ChangeArg::Block => Self::Block,
            ChangeArg::Unblock => Self::Unblock,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LocalRoutingChoiceArgs {
    pub change: ChangeArg,
    /// Hex-encoded mesh endpoint id of the peer.
    pub peer_id: String,
    /// RFC 3339 time a block lapses; absent for "until I undo" and for unblock.
    #[serde(default)]
    pub until: Option<String>,
    /// Hex of the caller's 32-byte salt. The plugin never picks its own: the
    /// caller must already hold it if the answer never arrives.
    pub salt: String,
}

/// Returns `{capsule_id, peer_commitment}` for the host's block store.
pub fn handle_local_routing_choice(
    capsules: &Arc<CapsuleState>,
    args: LocalRoutingChoiceArgs,
) -> PluginResult<Value> {
    let peer_id = args.peer_id.trim();
    if peer_id.is_empty() {
        return Err(PluginError::invalid_params("peer_id must not be empty"));
    }
    if matches!(args.change, ChangeArg::Unblock) && args.until.is_some() {
        return Err(PluginError::invalid_params("an unblock has no `until`"));
    }
    let salt: [u8; 32] = hex::decode(args.salt.trim())
        .ok()
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| PluginError::invalid_params("salt must be 32 bytes of hex"))?;
    // The operator's choice: it names no rule (`routing_rule` seals its own).
    let citation: Option<RoutingRuleCitation> = None;
    let emitted = capsules
        .emit_local_routing_choice(
            args.change.into(),
            peer_id,
            args.until.as_deref(),
            &salt,
            citation.as_ref(),
        )
        .map_err(|error| PluginError::internal(format!("could not seal the record: {error}")))?;
    Ok(json!({
        "capsule_id": emitted.capsule_id,
        "peer_commitment": emitted.peer_commitment,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(tag: &str) -> Arc<CapsuleState> {
        let dir = std::env::temp_dir().join(format!("route-bridge-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Arc::new(CapsuleState::open(&dir, "node-under-test").expect("open state"))
    }

    fn args(change: ChangeArg, until: Option<&str>, salt: &str) -> LocalRoutingChoiceArgs {
        LocalRoutingChoiceArgs {
            change,
            peer_id: "a".repeat(64),
            until: until.map(str::to_string),
            salt: salt.to_string(),
        }
    }

    #[test]
    fn block_returns_the_sealed_id_and_the_commitment_the_callers_salt_gives() {
        let capsules = state("ok");
        let out =
            handle_local_routing_choice(&capsules, args(ChangeArg::Block, None, &"07".repeat(32)))
                .expect("sealed");
        assert_eq!(
            out["capsule_id"].as_str(),
            capsules.chain_head().as_deref(),
            "the returned id is the record now at the chain head"
        );
        assert_eq!(
            out["peer_commitment"].as_str(),
            Some(crate::producer::capsule::peer_commitment(&"a".repeat(64), &[7u8; 32]).as_str())
        );
        assert!(
            out.get("salt").is_none(),
            "the salt never leaves in the answer"
        );
    }

    #[test]
    fn bad_calls_are_refused_without_sealing() {
        let capsules = state("bad");
        let salt = "07".repeat(32);
        let mut empty = args(ChangeArg::Block, None, &salt);
        empty.peer_id = "  ".into();
        assert!(handle_local_routing_choice(&capsules, empty).is_err());
        let unblock_until = args(ChangeArg::Unblock, Some("2026-10-04T00:00:00Z"), &salt);
        assert!(handle_local_routing_choice(&capsules, unblock_until).is_err());
        let short_salt = args(ChangeArg::Block, None, "07");
        assert!(handle_local_routing_choice(&capsules, short_salt).is_err());
        assert_eq!(capsules.chain_head(), None, "a refused call seals nothing");
    }
}
