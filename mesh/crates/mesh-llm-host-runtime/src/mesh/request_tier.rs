//! Request tiers: whom a request this node serves is for.
//!
//! The node a request enters decides its tier once and runs the rest of that
//! request inside [`with_request_tier`]:
//!
//! - `local`: this machine, a node on its `[private] nodes` list, or a node
//!   with a verified certificate from this node's own owner;
//! - `trusted`: a node with a verified certificate from an owner in this
//!   node's trust store;
//! - `mesh`: everyone else.
//!
//! `[request_tiers] owners` in the config file replaces the tier of one
//! owner's verified nodes. A node that forwards a request names its client's
//! tier in [`REQUEST_TIER_HEADER`]; the receiving node keeps the lower of
//! that claim and its own classification of the sender, so no peer can raise
//! its own requests. Skippy orders admission and eviction by the tier and
//! keeps a separate cache for each one.

use super::config_reload::{ReloadedSetting, SettingSpec, WhenUnloadable};
use crate::crypto::{OwnershipStatus, OwnershipSummary};
use iroh::EndpointId;
use std::collections::BTreeMap;
use std::path::Path;

mod class;
pub(crate) use class::{
    CACHE_PARTITION_HEADER, InferenceRequestTier, PRIVATE_CACHE_PARTITION,
    SKIPPY_REQUEST_TIER_HEADER,
};

/// The header a forwarding node names its client's tier in.
pub(crate) const REQUEST_TIER_HEADER: &str = "x-mesh-request-tier";

tokio::task_local! {
    static CURRENT_REQUEST_TIER: InferenceRequestTier;
}

/// Run `future` with `tier` as the tier of the request this task serves.
pub(crate) async fn with_request_tier<F: std::future::Future>(
    tier: InferenceRequestTier,
    future: F,
) -> F::Output {
    CURRENT_REQUEST_TIER.scope(tier, future).await
}

/// The tier of the request this task is serving. Work that no ingress
/// classified started on this machine, so it is `local`.
pub(crate) fn current_request_tier() -> InferenceRequestTier {
    classified_request_tier().unwrap_or(InferenceRequestTier::Local)
}

/// The tier of the request this task is serving, or `None` on a task no
/// ingress classified.
pub(crate) fn classified_request_tier() -> Option<InferenceRequestTier> {
    CURRENT_REQUEST_TIER.try_with(|tier| *tier).ok()
}

/// The tier the raw HTTP request `raw` claims in [`REQUEST_TIER_HEADER`]:
/// the lowest of its values, `local` (no limit) when it has none. A value
/// that is not a tier, or a head that cannot be read, claims `mesh`.
pub(crate) fn claimed_request_tier(raw: &[u8]) -> InferenceRequestTier {
    match crate::network::openai::request_parse::header_values_from_parsed_raw(
        raw,
        REQUEST_TIER_HEADER,
    ) {
        Ok(values) => values
            .iter()
            .flat_map(|value| value.split(','))
            .map(|value| InferenceRequestTier::parse(value).unwrap_or(InferenceRequestTier::Mesh))
            .fold(InferenceRequestTier::Local, InferenceRequestTier::min),
        Err(()) => InferenceRequestTier::Mesh,
    }
}

/// Insert `x-mesh-request-tier: <tier>` as the first header of the raw HTTP
/// request `raw`.
pub(crate) fn insert_request_tier_header(raw: &mut Vec<u8>, tier: InferenceRequestTier) {
    insert_header_lines(raw, &format!("{REQUEST_TIER_HEADER}: {tier}\r\n"));
}

/// The raw request `raw` with the current request's class named for this
/// node's own Skippy server, which is reached over loopback outside this
/// task. `None` for a `local` request outside the private partition: Skippy
/// serves a request without the headers as that.
pub(crate) fn mark_class_for_local_server(raw: &[u8], private: bool) -> Option<Vec<u8>> {
    let tier = current_request_tier();
    if tier == InferenceRequestTier::Local && !private {
        return None;
    }
    let mut lines = format!("{SKIPPY_REQUEST_TIER_HEADER}: {tier}\r\n");
    if private {
        lines.push_str(&format!(
            "{}: {}\r\n",
            CACHE_PARTITION_HEADER, PRIVATE_CACHE_PARTITION
        ));
    }
    let mut raw = raw.to_vec();
    insert_header_lines(&mut raw, &lines);
    Some(raw)
}

/// Insert header `lines` right after the request line of `raw`.
fn insert_header_lines(raw: &mut Vec<u8>, lines: &str) {
    let request_line_end = raw
        .windows(2)
        .position(|pair| pair == b"\r\n")
        .map_or(0, |at| at + 2);
    raw.splice(request_line_end..request_line_end, lines.bytes());
}

/// Per-owner tier overrides: `[request_tiers] owners` in the config file,
/// kept current while the node runs.
#[derive(Clone, Debug)]
pub(crate) struct RequestTierOwners(ReloadedSetting<BTreeMap<String, InferenceRequestTier>>);

impl Default for RequestTierOwners {
    fn default() -> Self {
        Self::of(BTreeMap::new())
    }
}

// An override can lower an owner, so a broken file keeps the last overrides
// that loaded rather than lifting them.
const SETTING: SettingSpec<BTreeMap<String, InferenceRequestTier>> = SettingSpec {
    name: "request_tiers.owners",
    read: owner_tiers_in,
    when_unloadable: WhenUnloadable::KeepLast,
};

impl RequestTierOwners {
    pub(crate) fn of(owners: BTreeMap<String, InferenceRequestTier>) -> Self {
        Self(ReloadedSetting::fixed(SETTING, owners))
    }

    /// The overrides in `config`, kept current from the config file it came
    /// from when `reload`.
    pub(crate) fn kept_current(
        config: &mesh_llm_config::MeshConfig,
        config_path: Option<&Path>,
        reload: bool,
    ) -> Self {
        Self(ReloadedSetting::kept_current(
            SETTING,
            config,
            config_path,
            reload,
        ))
    }

    /// Read the overrides from the config file now.
    pub(crate) fn reload_now(&self) {
        self.0.reload_now();
    }

    fn tier_of(&self, owner_id: &str) -> Option<InferenceRequestTier> {
        self.0
            .current()
            .get(&owner_id.to_ascii_lowercase())
            .copied()
    }
}

/// The overrides in `config`, by lowercase owner id.
fn owner_tiers_in(config: &mesh_llm_config::MeshConfig) -> BTreeMap<String, InferenceRequestTier> {
    config
        .request_tiers
        .owners
        .iter()
        .map(|(owner, tier)| (owner.trim().to_ascii_lowercase(), request_tier(*tier)))
        .collect()
}

fn request_tier(tier: mesh_llm_config::RequestTierName) -> InferenceRequestTier {
    match tier {
        mesh_llm_config::RequestTierName::Local => InferenceRequestTier::Local,
        mesh_llm_config::RequestTierName::Trusted => InferenceRequestTier::Trusted,
        mesh_llm_config::RequestTierName::Mesh => InferenceRequestTier::Mesh,
    }
}

/// The owner a peer's certificate proves, while the certificate is valid.
fn verified_owner(summary: &OwnershipSummary, now_unix_ms: u64) -> Option<&str> {
    let unexpired = summary
        .expires_at_unix_ms
        .is_none_or(|expires| expires > now_unix_ms);
    (summary.status == OwnershipStatus::Verified && unexpired)
        .then_some(summary.owner_id.as_deref())
        .flatten()
}

/// What this node knows about one peer's owner, for [`owner_tier`].
struct OwnerFacts<'a> {
    peer_owner: Option<&'a str>,
    local_owner: Option<&'a str>,
    trusted: bool,
    overridden: Option<InferenceRequestTier>,
}

/// The tier of a peer that is neither this node nor a private node.
fn owner_tier(facts: OwnerFacts<'_>) -> InferenceRequestTier {
    let Some(owner) = facts.peer_owner else {
        return InferenceRequestTier::Mesh;
    };
    if let Some(tier) = facts.overridden {
        return tier;
    }
    if facts
        .local_owner
        .is_some_and(|local| local.eq_ignore_ascii_case(owner))
    {
        return InferenceRequestTier::Local;
    }
    if facts.trusted {
        InferenceRequestTier::Trusted
    } else {
        InferenceRequestTier::Mesh
    }
}

impl super::Node {
    /// The tier this node serves requests from `peer` at.
    pub(crate) async fn peer_request_tier(&self, peer: EndpointId) -> InferenceRequestTier {
        if peer == self.endpoint.id() || self.is_private_node(peer) {
            return InferenceRequestTier::Local;
        }
        let summary = {
            let state = self.state.lock().await;
            state
                .peers
                .get(&peer)
                .map(|info| info.owner_summary.clone())
        };
        let Some(summary) = summary else {
            return InferenceRequestTier::Mesh;
        };
        let Some(owner) = verified_owner(&summary, super::current_time_unix_ms()) else {
            return InferenceRequestTier::Mesh;
        };
        let local_owner = self.local_verified_owner_id().await;
        let trusted = self
            .trust_store
            .lock()
            .await
            .trusted_owners
            .iter()
            .any(|entry| entry.owner_id.eq_ignore_ascii_case(owner));
        owner_tier(OwnerFacts {
            peer_owner: Some(owner),
            local_owner: local_owner.as_deref(),
            trusted,
            overridden: self.request_tier_owners.tier_of(owner),
        })
    }

    /// The tier of a request entering at this node. A peer's request is the
    /// lower of its claim and this node's view of the peer; any other request
    /// started on this machine and is `local`, lowered by its own claim.
    pub(crate) async fn ingress_request_tier(
        &self,
        raw: &[u8],
        ingress_type: crate::runtime::IngressType,
        requested_by: Option<EndpointId>,
    ) -> InferenceRequestTier {
        let claimed = claimed_request_tier(raw);
        match (ingress_type, requested_by) {
            (crate::runtime::IngressType::RemoteQuicHttp, Some(peer)) => {
                self.forwarded_request_tier(peer, claimed).await
            }
            (crate::runtime::IngressType::RemoteQuicHttp, None) => InferenceRequestTier::Mesh,
            _ => claimed,
        }
    }

    /// The tier of a request `peer` forwarded here claiming `claimed`: the
    /// lower of the claim and this node's own view of the peer.
    pub(crate) async fn forwarded_request_tier(
        &self,
        peer: EndpointId,
        claimed: InferenceRequestTier,
    ) -> InferenceRequestTier {
        claimed.min(self.peer_request_tier(peer).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const OTHER: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    fn facts(peer_owner: Option<&'static str>) -> OwnerFacts<'static> {
        OwnerFacts {
            peer_owner,
            local_owner: Some(OWNER),
            trusted: false,
            overridden: None,
        }
    }

    #[test]
    fn an_unverified_peer_is_mesh_even_when_overridden() {
        let tier = owner_tier(OwnerFacts {
            overridden: Some(InferenceRequestTier::Local),
            ..facts(None)
        });
        assert_eq!(tier, InferenceRequestTier::Mesh);
    }

    #[test]
    fn the_own_owner_is_local_and_a_trusted_owner_is_trusted() {
        assert_eq!(owner_tier(facts(Some(OWNER))), InferenceRequestTier::Local);
        assert_eq!(owner_tier(facts(Some(OTHER))), InferenceRequestTier::Mesh);
        let trusted = owner_tier(OwnerFacts {
            trusted: true,
            ..facts(Some(OTHER))
        });
        assert_eq!(trusted, InferenceRequestTier::Trusted);
    }

    #[test]
    fn an_override_replaces_the_derived_tier() {
        let demoted = owner_tier(OwnerFacts {
            overridden: Some(InferenceRequestTier::Mesh),
            ..facts(Some(OWNER))
        });
        assert_eq!(demoted, InferenceRequestTier::Mesh);
        let promoted = owner_tier(OwnerFacts {
            overridden: Some(InferenceRequestTier::Local),
            ..facts(Some(OTHER))
        });
        assert_eq!(promoted, InferenceRequestTier::Local);
    }

    #[test]
    fn only_a_valid_certificate_proves_an_owner() {
        let summary = OwnershipSummary {
            owner_id: Some(OWNER.into()),
            status: OwnershipStatus::Verified,
            expires_at_unix_ms: Some(1_000),
            verified: true,
            ..OwnershipSummary::default()
        };
        assert_eq!(verified_owner(&summary, 999), Some(OWNER));
        assert_eq!(verified_owner(&summary, 1_000), None);
        let untrusted = OwnershipSummary {
            status: OwnershipStatus::UntrustedOwner,
            ..summary
        };
        assert_eq!(verified_owner(&untrusted, 0), None);
    }

    #[test]
    fn owner_overrides_come_from_the_config_file() {
        let config = mesh_llm_config::parse_config_toml(&format!(
            "[request_tiers.owners]\n\"{}\" = \"trusted\"\n",
            OTHER.to_ascii_uppercase()
        ))
        .unwrap();
        let owners = RequestTierOwners::kept_current(&config, None, false);
        assert_eq!(owners.tier_of(OTHER), Some(InferenceRequestTier::Trusted));
        assert_eq!(owners.tier_of(OWNER), None);
    }

    #[tokio::test]
    async fn the_tier_is_local_outside_a_classified_request() {
        assert_eq!(current_request_tier(), InferenceRequestTier::Local);
        let inside =
            with_request_tier(InferenceRequestTier::Mesh, async { current_request_tier() }).await;
        assert_eq!(inside, InferenceRequestTier::Mesh);
    }

    #[test]
    fn a_claim_reads_the_lowest_value_and_fails_low() {
        let head = |headers: &str| {
            format!("POST /v1/chat/completions HTTP/1.1\r\n{headers}\r\n").into_bytes()
        };
        assert_eq!(claimed_request_tier(&head("")), InferenceRequestTier::Local);
        let lowered = head("x-mesh-request-tier: local, trusted\r\nX-Mesh-Request-Tier: local\r\n");
        assert_eq!(
            claimed_request_tier(&lowered),
            InferenceRequestTier::Trusted
        );
        let unknown = head("x-mesh-request-tier: owner\r\n");
        assert_eq!(claimed_request_tier(&unknown), InferenceRequestTier::Mesh);
    }
}
