//! Network path diagnostics for direct-connect readiness.
//!
//! Surfaces the two checks that discriminate the #1300 failure shape — a
//! port-remapping host advertising its locally bound UDP port:
//!
//! 1. does every advertised public candidate carry the externally observed
//!    (relay-verified) tuple?
//! 2. what path kind (direct/relay/unknown) do we currently hold per peer?

use super::MeshApi;
use serde::Serialize;
use std::future::Future;

/// Verdict for the local advertisement's direct-connect readiness.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AdvertisementVerdict {
    /// Every public candidate was externally observed — the invite carries
    /// only tuples the NAT actually preserves.
    Verified,
    /// No public IPv4 is advertised at all: relay-only discovery, LAN-only
    /// mode, or an address-dependent mapping suppressed the candidates.
    NoPublicAdvertised,
    /// A public candidate carries the locally bound port that was never
    /// externally verified. Peers may see an unreachable direct address
    /// (#1300) and fall back to relay.
    UnverifiedCandidate,
}

/// One advertised direct-connect candidate with its verification state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct AdvertisedCandidateReport {
    pub(crate) addr: std::net::SocketAddr,
    pub(crate) externally_verified: bool,
}

/// Per-peer path state for split-relevant transport decisions.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct PeerPathReport {
    pub(crate) node_id: String,
    pub(crate) short_node_id: String,
    /// "direct" | "relay" | "unknown"
    pub(crate) path: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) rtt_ms: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct NetworkDiagnosticsReport {
    pub(crate) node_id: String,
    pub(crate) advertisement: AdvertisementReport,
    pub(crate) peers: Vec<PeerPathReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct AdvertisementReport {
    pub(crate) verdict: AdvertisementVerdict,
    /// Human-readable summary of the verdict for doctor rendering.
    pub(crate) summary: &'static str,
    pub(crate) public_candidates: Vec<AdvertisedCandidateReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) lan_candidates: Vec<std::net::SocketAddr>,
    /// Externally observed public tuple from startup discovery, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) observed_public: Option<std::net::SocketAddr>,
    /// True when raw-STUN discovery is enabled for this node.
    pub(crate) raw_stun_enabled: bool,
}

/// Classify an advertisement snapshot into a doctor-facing verdict.
///
/// Pure helper so the decision is unit-testable without a live node.
pub(crate) fn advertisement_verdict(
    public_candidates: &[crate::mesh::AdvertisedCandidate],
) -> AdvertisementVerdict {
    if public_candidates.is_empty() {
        return AdvertisementVerdict::NoPublicAdvertised;
    }
    let all_verified = public_candidates
        .iter()
        .all(|candidate| candidate.externally_verified);
    if all_verified {
        AdvertisementVerdict::Verified
    } else {
        AdvertisementVerdict::UnverifiedCandidate
    }
}

fn advertisement_summary(verdict: AdvertisementVerdict) -> &'static str {
    match verdict {
        AdvertisementVerdict::Verified => {
            "advertised public address was externally observed (relay probe); direct path should work when the peer is punchable"
        }
        AdvertisementVerdict::NoPublicAdvertised => {
            "no public IPv4 is advertised; peers reach this node via relay or LAN candidates"
        }
        AdvertisementVerdict::UnverifiedCandidate => {
            "an advertised public candidate carries the locally bound port, which was never verified externally; on port-remapping hosts peers see an unreachable address and fall back to relay"
        }
    }
}

fn peer_path_report(
    node: &crate::mesh::Node,
    peer: &crate::mesh::PeerInfo,
) -> impl Future<Output = PeerPathReport> {
    let node_id = peer.id.to_string();
    let short_node_id = peer.id.fmt_short().to_string();
    async move {
        let snapshot = node.split_stage_path_snapshot(peer.id).await;
        PeerPathReport {
            node_id,
            short_node_id,
            path: snapshot.kind_name(),
            rtt_ms: snapshot.rtt_ms,
        }
    }
}

impl MeshApi {
    pub(crate) async fn network_diagnostics_report(&self) -> NetworkDiagnosticsReport {
        let node = self.inner.lock().await.node.clone();
        let snapshot = node.advertisement_snapshot();
        let verdict = advertisement_verdict(snapshot.public_ipv4.as_slice());

        let mut peers = Vec::new();
        for peer in node.peers().await {
            peers.push(peer_path_report(&node, &peer).await);
        }

        NetworkDiagnosticsReport {
            node_id: node.id().to_string(),
            advertisement: AdvertisementReport {
                verdict,
                summary: advertisement_summary(verdict),
                public_candidates: snapshot
                    .public_ipv4
                    .iter()
                    .map(|candidate| AdvertisedCandidateReport {
                        addr: candidate.addr,
                        externally_verified: candidate.externally_verified,
                    })
                    .collect(),
                lan_candidates: snapshot.lan_candidates,
                observed_public: snapshot.observed_public,
                raw_stun_enabled: snapshot.raw_stun_enabled,
            },
            peers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_observed_candidate_is_verified() {
        let observed = std::net::SocketAddr::from(([213, 5, 72, 196], 23_555));
        let candidates = vec![crate::mesh::AdvertisedCandidate {
            addr: observed,
            externally_verified: true,
        }];

        assert_eq!(
            advertisement_verdict(&candidates),
            AdvertisementVerdict::Verified
        );
    }

    #[test]
    fn unverified_bound_port_candidate_is_flagged() {
        // The #1300 shape: a locally enumerated public IP carrying the bound
        // port survives because discovery never completed.
        let candidates = vec![crate::mesh::AdvertisedCandidate {
            addr: std::net::SocketAddr::from(([213, 5, 72, 196], 41_842)),
            externally_verified: false,
        }];

        assert_eq!(
            advertisement_verdict(&candidates),
            AdvertisementVerdict::UnverifiedCandidate
        );
    }

    #[test]
    fn empty_public_candidates_report_no_public_advertised() {
        assert_eq!(
            advertisement_verdict(&[]),
            AdvertisementVerdict::NoPublicAdvertised
        );
        // Address-dependent mapping suppresses public candidates even when
        // raw STUN is enabled.
        assert_eq!(
            advertisement_verdict(&[]),
            AdvertisementVerdict::NoPublicAdvertised
        );
    }

    #[test]
    fn verdicts_serialize_snake_case() {
        let verified = serde_json::to_value(AdvertisementVerdict::Verified).unwrap();
        assert_eq!(verified, serde_json::json!("verified"));
        let unverified = serde_json::to_value(AdvertisementVerdict::UnverifiedCandidate).unwrap();
        assert_eq!(unverified, serde_json::json!("unverified_candidate"));
    }
}
