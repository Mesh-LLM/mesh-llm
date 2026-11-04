//! The request trust policy (`mesh/docs/design/TRUST_TIERS.md`): the tiers a
//! candidate must qualify for to serve a request. This node knows two:
//! `any`, today's routing, and `receipts`, a runtime that issues inference
//! receipts for the model (`mesh/docs/design/RECEIPTS.md`).
//!
//! The node default is `[receipts] require`; a request narrows it with
//! `x-mesh-trust: receipts`. The policy of a request is decided once, where
//! local ingress reads it off its socket ([`decide`]), and kept with the
//! request as its [`RequestTrust`]. A request nothing decided for is served
//! by nothing.
//!
//! Under `receipts` a candidate is on an allow-list checked where it is
//! about to be used ([`permits`]): this node's own runtime when all of it
//! runs here and issues receipts, or a peer whose advertisement says, right
//! then, that it serves the model on this endpoint with receipts. A peer
//! counts only while this node audits its turns, since an unchecked receipt
//! proves nothing: with `[receipts] offline` no peer qualifies. Receipts
//! cover chat completions, so the policy refuses every other inference
//! endpoint ([`covers`]). A request no candidate may serve fails closed
//! with `trust_policy_unsatisfied`.

use std::collections::HashMap;

use iroh::EndpointId;
use mesh_llm_config::ReceiptsConfig;

use crate::inference::election::InferenceTarget;
use crate::mesh;
use crate::network::openai::request_parse::header_values_from_raw;
use crate::network::openai::workload_routing::descriptor_serves_request;

/// The request header that narrows the node's trust policy.
pub(crate) const MESH_TRUST_HEADER: &str = "x-mesh-trust";
/// The error code of a request no candidate can serve under its policy.
pub(crate) const UNSATISFIED_CODE: &str = "trust_policy_unsatisfied";
/// The endpoint receipts cover.
const CHAT_PATH: &str = "/v1/chat/completions";
/// Where a client opens the receipt of a turn it audits. They reach only a
/// runtime that issues receipts, so the policy lets them through.
const RECEIPT_PATHS: [&str; 2] = ["/v1/receipts/open", "/v1/receipts/rows"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    Any,
    Receipts,
}

impl Tier {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "any" => Ok(Self::Any),
            "receipts" => Ok(Self::Receipts),
            "trusted" | "measured" | "tee" => {
                Err(format!("trust tier '{name}' is not available on this node"))
            }
            _ => Err(format!("unknown trust tier '{name}'")),
        }
    }
}

/// The tiers `x-mesh-trust` names, `None` without the header. Each value is
/// a comma-separated list; an empty or unknown entry rejects the request.
fn parse_tiers(values: &[String]) -> Result<Option<Vec<Tier>>, String> {
    if values.is_empty() {
        return Ok(None);
    }
    let mut tiers = Vec::new();
    for part in values.iter().flat_map(|value| value.split(',')) {
        let name = part.trim().to_ascii_lowercase();
        if name.is_empty() {
            return Err(format!("{MESH_TRUST_HEADER} contains an empty entry"));
        }
        tiers.push(Tier::parse(&name)?);
    }
    Ok(Some(tiers))
}

/// Whether a request must be served with receipts: the node default unless
/// the header narrows it. A header that names no tier the node allows is an
/// error, never a weaker policy.
fn receipts_required(header: Option<&[Tier]>, node_requires: bool) -> Result<bool, String> {
    let Some(tiers) = header else {
        return Ok(node_requires);
    };
    let allowed = |tier: &&Tier| !node_requires || **tier == Tier::Receipts;
    let allowed: Vec<_> = tiers.iter().filter(allowed).collect();
    if allowed.is_empty() {
        return Err(format!(
            "{MESH_TRUST_HEADER} names no tier this node allows: it requires receipts"
        ));
    }
    Ok(!allowed.contains(&&Tier::Any))
}

/// The trust policy one request runs under.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RequestTrust {
    /// No ingress has decided the policy. Nothing serves such a request: a
    /// path that reads requests without deciding fails closed.
    #[default]
    Undecided,
    /// Any candidate may serve the request. `audit`: this node audits the
    /// receipted turns its peers serve for it.
    Any { audit: bool },
    /// Only a runtime that issues receipts may serve the request. `audit`
    /// is off when the node never audits (`[receipts] offline`), and then
    /// no peer qualifies.
    Receipts { audit: bool },
}

impl RequestTrust {
    /// No policy and no audit: a request a peer assigned to this node,
    /// which is executed here and never routed, so its policy was the
    /// sender's to apply.
    pub(crate) const UNRESTRICTED: Self = Self::Any { audit: false };

    /// Whether only a runtime that issues receipts may serve the request.
    pub(crate) fn requires_receipts(self) -> bool {
        !matches!(self, Self::Any { .. })
    }

    /// Whether this node audits the receipted turns its peers serve for
    /// the request.
    pub(crate) fn audits_peers(self) -> bool {
        matches!(
            self,
            Self::Any { audit: true } | Self::Receipts { audit: true }
        )
    }

    /// Why nothing may serve a request for `model` under this policy.
    pub(crate) fn unsatisfied(self, model: &str) -> Unsatisfied {
        Unsatisfied::NoCandidate(match self {
            Self::Undecided => {
                "the request reached routing without a trust policy decision, so nothing may serve it"
                    .to_owned()
            }
            Self::Receipts { audit: false } => format!(
                "this node does not serve model '{model}' with receipts, and no peer can satisfy the request's trust policy: receipt audits are turned off here ([receipts] offline = true), so a peer's receipts cannot be checked"
            ),
            Self::Any { .. } | Self::Receipts { audit: true } => format!(
                "no candidate serves model '{model}' with receipts, which the request's trust policy requires"
            ),
        })
    }

    /// Why a route that cannot prove the `receipts` tier (`what`: a plugin
    /// endpoint, a virtual model, a pipeline) refuses the request.
    pub(crate) fn unprovable(what: &str) -> Unsatisfied {
        Unsatisfied::NoCandidate(format!(
            "the request's trust policy requires receipts, which {what} cannot provide"
        ))
    }
}

/// Why a request cannot run under its policy.
#[derive(Debug)]
pub(crate) enum Unsatisfied {
    /// The header is malformed or names nothing the node allows: 400.
    Invalid(String),
    /// No candidate qualifies: 409 with [`UNSATISFIED_CODE`].
    NoCandidate(String),
}

/// The policy of the request `raw` under the node's `[receipts]` settings.
/// A head that cannot be read is an error, never "no header".
fn decide_with(receipts: &ReceiptsConfig, raw: &[u8]) -> Result<RequestTrust, String> {
    let values = header_values_from_raw(raw, MESH_TRUST_HEADER)
        .map_err(|()| format!("{MESH_TRUST_HEADER} header contains invalid UTF-8"))?;
    let required = receipts_required(parse_tiers(&values)?.as_deref(), receipts.require)?;
    Ok(if required {
        RequestTrust::Receipts {
            audit: !receipts.offline,
        }
    } else {
        RequestTrust::Any {
            audit: receipts.audit && !receipts.offline,
        }
    })
}

/// Decides the policy of the request `raw` a local client sent: one read of
/// the node's settings and one parse of the head, for the whole request.
pub(crate) async fn decide(node: &mesh::Node, raw: &[u8]) -> Result<RequestTrust, Unsatisfied> {
    let receipts = node.config_state.lock().await.config().receipts.clone();
    decide_with(&receipts, raw).map_err(Unsatisfied::Invalid)
}

/// Whether this node audits the turn the forwarded request `raw` asks a
/// peer for: `[receipts] audit` or a policy that requires receipts, and
/// never with `[receipts] offline`. For the audit itself
/// (`network::receipt_audit`), which sees only the forwarded bytes.
pub(crate) async fn audits_turn(node: &mesh::Node, raw: &[u8]) -> bool {
    let receipts = node.config_state.lock().await.config().receipts.clone();
    // a malformed policy never reaches a peer; were it to, it is audited
    decide_with(&receipts, raw).map_or(!receipts.offline, RequestTrust::audits_peers)
}

/// Whether the `receipts` tier covers the endpoint at `path`.
pub(crate) fn covers(path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    path == CHAT_PATH || RECEIPT_PATHS.contains(&path)
}

/// Why a request under the `receipts` tier is refused on an endpoint the
/// tier does not cover.
pub(crate) fn uncovered() -> Unsatisfied {
    Unsatisfied::NoCandidate(format!("the receipts trust policy covers {CHAT_PATH} only"))
}

/// Whether `peer` advertises, as its gossip stands now, that it serves
/// `model` on the endpoint at `path` with receipts. One descriptor has to
/// say all of it: a peer that issues receipts for an embedding model does
/// not qualify for chat under the same name.
fn peer_serves_with_receipts(peer: &mesh::PeerInfo, model: &str, path: &str) -> bool {
    peer.served_model_descriptors
        .iter()
        .any(|descriptor| descriptor.receipts && descriptor_serves_request(descriptor, model, path))
}

/// Whether `host` advertises, as its gossip stands now, that it serves
/// `model` for chat completions with receipts. The audit reads it before a
/// request is sent, so a peer cannot take its word back with the answer.
pub(crate) async fn peer_advertises_receipts(
    node: &mesh::Node,
    host: &EndpointId,
    model: &str,
) -> bool {
    let state = node.state.lock().await;
    state
        .peers
        .get(host)
        .is_some_and(|peer| peer_serves_with_receipts(peer, model, CHAT_PATH))
}

fn permits_now(
    trust: RequestTrust,
    local_issues_receipts: bool,
    peers: &HashMap<EndpointId, mesh::PeerInfo>,
    model: &str,
    path: &str,
    target: &InferenceTarget,
) -> bool {
    match (trust, target) {
        (RequestTrust::Any { .. }, _) => true,
        (RequestTrust::Undecided, _) | (_, InferenceTarget::None) => false,
        (RequestTrust::Receipts { .. }, InferenceTarget::Local(_)) => local_issues_receipts,
        (RequestTrust::Receipts { audit }, InferenceTarget::Remote(peer)) => {
            audit
                && peers
                    .get(peer)
                    .is_some_and(|peer| peer_serves_with_receipts(peer, model, path))
        }
    }
}

/// The candidates of `candidates` that may serve a request for `model` on
/// `path` under `trust`, as their advertisements stand now. Every pool a
/// request is routed from goes through this, the fallbacks included.
pub(crate) async fn permitted(
    node: &mesh::Node,
    trust: RequestTrust,
    model: &str,
    path: &str,
    candidates: Vec<InferenceTarget>,
) -> Vec<InferenceTarget> {
    if !trust.requires_receipts() {
        return candidates;
    }
    let local = node.issues_receipts(model);
    let state = node.state.lock().await;
    candidates
        .into_iter()
        .filter(|target| permits_now(trust, local, &state.peers, model, path, target))
        .collect()
}

/// Whether `target` may serve a request for `model` on `path` under
/// `trust`, as its advertisement stands now. Asked again before every
/// attempt, so a retry or a failover never reaches a candidate that stopped
/// qualifying, or one a pool picked up after the policy was first applied.
pub(crate) async fn permits(
    node: &mesh::Node,
    trust: RequestTrust,
    model: &str,
    path: &str,
    target: &InferenceTarget,
) -> bool {
    if !trust.requires_receipts() {
        return true;
    }
    let local = node.issues_receipts(model);
    let state = node.state.lock().await;
    permits_now(trust, local, &state.peers, model, path, target)
}

/// Whether this node's own runtime for `model` may serve a request under
/// `trust`, whichever port it listens on: asked of a model that is still
/// starting here and has no candidate to ask about yet.
pub(crate) fn permits_local_runtime(node: &mesh::Node, trust: RequestTrust, model: &str) -> bool {
    !trust.requires_receipts() || node.issues_receipts(model)
}

/// The peers of `hosts` that may serve a request for `model` on `path`
/// under `trust`.
pub(crate) async fn permitted_hosts(
    node: &mesh::Node,
    trust: RequestTrust,
    model: &str,
    path: &str,
    hosts: Vec<EndpointId>,
) -> Vec<EndpointId> {
    let candidates = hosts.into_iter().map(InferenceTarget::Remote).collect();
    permitted(node, trust, model, path, candidates)
        .await
        .into_iter()
        .filter_map(|target| match target {
            InferenceTarget::Remote(peer) => Some(peer),
            InferenceTarget::Local(_) | InferenceTarget::None => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(raw: &[&str]) -> Vec<String> {
        raw.iter().map(|value| value.to_string()).collect()
    }

    fn policy(header: &[&str], node_requires: bool) -> Result<bool, String> {
        receipts_required(parse_tiers(&values(header))?.as_deref(), node_requires)
    }

    fn chat(header: &str) -> Vec<u8> {
        format!("POST /v1/chat/completions HTTP/1.1\r\n{header}Content-Length: 2\r\n\r\n{{}}")
            .into_bytes()
    }

    #[test]
    fn the_header_narrows_the_node_default() {
        assert_eq!(policy(&[], false), Ok(false));
        assert_eq!(policy(&[], true), Ok(true));
        assert_eq!(policy(&["receipts"], false), Ok(true));
        assert_eq!(policy(&["any"], false), Ok(false));
        assert_eq!(policy(&["Receipts, any"], false), Ok(false));
        assert_eq!(policy(&["receipts"], true), Ok(true));
        assert_eq!(policy(&["any", "receipts"], true), Ok(true));
    }

    #[test]
    fn a_header_cannot_weaken_the_node_default() {
        assert!(policy(&["any"], true).is_err());
    }

    #[test]
    fn unknown_empty_and_unavailable_tiers_are_errors() {
        assert!(policy(&["fast"], false).is_err());
        assert!(policy(&["receipts,"], false).is_err());
        let unavailable = policy(&["trusted"], false).unwrap_err();
        assert!(unavailable.contains("not available"), "{unavailable}");
    }

    #[test]
    fn the_policy_is_read_from_the_request_and_the_node_settings() {
        let off = ReceiptsConfig::default();
        let named = chat("X-Mesh-Trust: receipts\r\n");
        assert_eq!(
            decide_with(&off, &named),
            Ok(RequestTrust::Receipts { audit: true })
        );
        assert_eq!(
            decide_with(&off, &chat("")),
            Ok(RequestTrust::Any { audit: false })
        );
        let auditing = ReceiptsConfig {
            audit: true,
            ..Default::default()
        };
        assert_eq!(
            decide_with(&auditing, &chat("")),
            Ok(RequestTrust::Any { audit: true })
        );
        let requiring = ReceiptsConfig {
            require: true,
            ..Default::default()
        };
        assert_eq!(
            decide_with(&requiring, &chat("")),
            Ok(RequestTrust::Receipts { audit: true })
        );
        assert!(decide_with(&requiring, &chat("x-mesh-trust: any\r\n")).is_err());
    }

    #[test]
    fn an_offline_node_never_audits_whatever_asks_it_to() {
        let offline = ReceiptsConfig {
            audit: true,
            offline: true,
            ..Default::default()
        };
        assert_eq!(
            decide_with(&offline, &chat("")),
            Ok(RequestTrust::Any { audit: false })
        );
        let named = decide_with(&offline, &chat("x-mesh-trust: receipts\r\n")).unwrap();
        assert_eq!(named, RequestTrust::Receipts { audit: false });
        assert!(!named.audits_peers());
        let Unsatisfied::NoCandidate(why) = named.unsatisfied("m") else {
            panic!("an unsatisfied policy is a 409");
        };
        assert!(why.contains("offline"), "{why}");
    }

    #[test]
    fn an_undecided_request_requires_what_nothing_provides() {
        let undecided = RequestTrust::default();
        assert!(undecided.requires_receipts());
        assert!(!undecided.audits_peers());
        let peers = HashMap::new();
        for target in [InferenceTarget::Local(1), InferenceTarget::None] {
            assert!(!permits_now(
                undecided, true, &peers, "m", CHAT_PATH, &target
            ));
        }
        assert!(!RequestTrust::UNRESTRICTED.requires_receipts());
    }

    #[test]
    fn the_policy_covers_chat_and_the_receipt_routes() {
        assert!(covers("/v1/chat/completions"));
        assert!(covers("/v1/chat/completions?x=1"));
        assert!(covers("/v1/receipts/open"));
        assert!(covers("/v1/receipts/rows"));
        for path in ["/v1/embeddings", "/v1/realtime", "/v1/completions"] {
            assert!(!covers(path), "{path}");
        }
    }
}
