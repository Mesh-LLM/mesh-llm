//! The request trust policy (`mesh/docs/design/TRUST_TIERS.md`): the tiers a
//! candidate must qualify for to serve a request. This node knows two:
//! `any`, today's routing, and `receipts`, a runtime that issues inference
//! receipts for the model (`mesh/docs/design/RECEIPTS.md`).
//!
//! The node default is `[receipts] require`; a request narrows it with
//! `x-mesh-trust: receipts`. A request under `receipts` goes only to this
//! node's own runtime or to a peer that advertises receipts for the model,
//! and fails closed with `trust_policy_unsatisfied` when there is none.
//! Receipts cover chat completions only, so the policy refuses every other
//! inference endpoint. Its turns are audited even with `[receipts] audit`
//! off: an unchecked receipt proves nothing.

use iroh::EndpointId;

use crate::mesh;
use crate::network::openai::request_parse::header_values_from_raw;

/// The request header that narrows the node's trust policy.
pub(crate) const MESH_TRUST_HEADER: &str = "x-mesh-trust";
/// The error code of a request no candidate can serve under its policy.
pub(crate) const UNSATISFIED_CODE: &str = "trust_policy_unsatisfied";
/// The one endpoint receipts cover.
const CHAT_PATH: &str = "/v1/chat/completions";

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

/// The policy of the request `raw`, against the node default.
pub(crate) fn requires_receipts(raw: &[u8], node_requires: bool) -> Result<bool, String> {
    let values = header_values_from_raw(raw, MESH_TRUST_HEADER)
        .map_err(|()| format!("{MESH_TRUST_HEADER} header contains invalid UTF-8"))?;
    receipts_required(parse_tiers(&values)?.as_deref(), node_requires)
}

/// Whether this node requires receipts on every request by default.
pub(crate) async fn node_requires_receipts(node: &mesh::Node) -> bool {
    node.config_state.lock().await.config().receipts.require
}

/// Why a request cannot run under its policy.
pub(crate) enum Unsatisfied {
    /// The header is malformed or names nothing the node allows: 400.
    Invalid(String),
    /// No candidate qualifies: 409 with [`UNSATISFIED_CODE`].
    NoCandidate(String),
}

/// The candidates the policy rules out for a request for a model.
pub(crate) struct Disqualified {
    /// This node's own runtime does not qualify.
    pub(crate) this_node: bool,
    /// The peers serving the model that do not qualify.
    pub(crate) peers: Vec<EndpointId>,
}

/// Applies the policy of `raw`, a request on `path` for `model`, to the
/// candidates that serve it. Blocked and excluded peers are the caller's
/// business; this only rules candidates out.
pub(crate) async fn disqualified(
    node: &mesh::Node,
    raw: &[u8],
    path: &str,
    model: &str,
) -> Result<Option<Disqualified>, Unsatisfied> {
    let required =
        requires_receipts(raw, node_requires_receipts(node).await).map_err(Unsatisfied::Invalid)?;
    if !required {
        return Ok(None);
    }
    if path.split('?').next() != Some(CHAT_PATH) {
        return Err(Unsatisfied::NoCandidate(format!(
            "the receipts trust policy covers {CHAT_PATH} only"
        )));
    }
    let hosts = node.hosts_for_model(model).await;
    let receipted = peers_with_receipts(node, model).await;
    let this_node = !node.issues_receipts(model);
    let peers: Vec<_> = hosts
        .iter()
        .copied()
        .filter(|id| !receipted.contains(id))
        .collect();
    if this_node && peers.len() == hosts.len() {
        return Err(Unsatisfied::NoCandidate(format!(
            "no candidate serves model '{model}' with receipts, which the request's trust policy requires"
        )));
    }
    Ok(Some(Disqualified { this_node, peers }))
}

/// The peers that advertise receipts for `model`, under its served name or
/// its public identity.
async fn peers_with_receipts(node: &mesh::Node, model: &str) -> Vec<EndpointId> {
    let state = node.state.lock().await;
    state
        .peers
        .values()
        .filter(|peer| {
            peer.served_model_descriptors.iter().any(|descriptor| {
                let name = &descriptor.identity.model_name;
                descriptor.receipts
                    && (name == model
                        || mesh::public_model_id_for_routable_model(peer, name) == model)
            })
        })
        .map(|peer| peer.id)
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
    fn the_policy_is_read_from_the_request() {
        let raw = b"POST /v1/chat/completions HTTP/1.1\r\nX-Mesh-Trust: receipts\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(requires_receipts(raw, false), Ok(true));
        let plain = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(requires_receipts(plain, false), Ok(false));
    }
}
