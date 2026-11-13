//! The mesh routing headers a client sends to narrow where its request
//! runs: `x-mesh-target`, `x-mesh-exclude` and `x-mesh-twin-bracket`.
use super::*;

/// Decode a hex-encoded 32-byte `EndpointId` from a header value string.
/// Leading/trailing whitespace is trimmed before decoding. Returns `None` on
/// any decode or length error — the caller is responsible for turning `None`
/// into an appropriate rejection (400 or 409).
pub(super) fn parse_endpoint_id_hex(value: &str) -> Option<iroh::EndpointId> {
    let bytes = hex::decode(value.trim()).ok()?;
    let bytes: [u8; 32] = bytes.as_slice().try_into().ok()?;
    iroh::EndpointId::from_bytes(&bytes).ok()
}

/// Parse the (possibly repeated) `x-mesh-target` header values. Zero values
/// is a no-op; exactly one must decode as an `EndpointId`; more than one is
/// ambiguous and rejected rather than silently picking one.
pub(super) fn parse_mesh_target_header(
    values: &[String],
) -> Result<Option<iroh::EndpointId>, String> {
    match values {
        [] => Ok(None),
        [only] => parse_endpoint_id_hex(only)
            .map(Some)
            .ok_or_else(|| format!("invalid x-mesh-target value '{only}'")),
        _ => Err("multiple x-mesh-target headers are ambiguous".to_string()),
    }
}

/// Parse the `x-mesh-exclude` header value(s), each a comma-separated list of
/// `EndpointId`s. Any unparseable OR empty entry rejects the whole request
/// rather than silently dropping an exclusion the client asked for.
pub(super) fn parse_mesh_exclude_header(
    values: &[String],
) -> Result<Vec<iroh::EndpointId>, String> {
    let mut excluded = Vec::new();
    for value in values {
        for part in value.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return Err("x-mesh-exclude contains an empty entry".to_string());
            }
            let id = parse_endpoint_id_hex(part)
                .ok_or_else(|| format!("invalid x-mesh-exclude value '{part}'"))?;
            excluded.push(id);
        }
    }
    Ok(excluded)
}

/// The longest `x-mesh-twin-bracket` value the host copies.
pub(super) const MAX_TWIN_BRACKET_LEN: usize = 128;

/// Parse the (possibly repeated) `x-mesh-twin-bracket` header values. Zero
/// values is a no-op; exactly one must be 1..=128 characters of
/// `[A-Za-z0-9._:-]` once HTTP whitespace (SP/HTAB) is trimmed from its ends;
/// any other whitespace, at the ends or inside, makes it malformed. More than
/// one value is ambiguous and rejected. The value is otherwise opaque: the
/// host copies it and never reads it.
pub(super) fn parse_twin_bracket_header(values: &[String]) -> Result<Option<String>, String> {
    match values {
        [] => Ok(None),
        [only] => {
            let value = only.trim_matches([' ', '\t']);
            let valid = !value.is_empty()
                && value.len() <= MAX_TWIN_BRACKET_LEN
                && value
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | ':' | '-'));
            valid
                .then(|| value.to_string())
                .map(Some)
                .ok_or_else(|| format!("invalid x-mesh-twin-bracket value '{value}'"))
        }
        _ => Err("multiple x-mesh-twin-bracket headers are ambiguous".to_string()),
    }
}

pub(super) fn twin_bracket_id(
    request: &proxy::BufferedHttpRequest,
) -> Result<Option<String>, String> {
    parse_twin_bracket_header(&request.twin_bracket_header_values()?)
}

/// Parse and validate the raw `x-mesh-target` / `x-mesh-exclude` header
/// values off `request` in one place, so every caller enforces them the same
/// way before making a routing decision.
pub(in crate::network::openai) fn parse_mesh_routing_headers(
    request: &proxy::BufferedHttpRequest,
) -> Result<(Option<iroh::EndpointId>, Vec<iroh::EndpointId>), String> {
    let (target_values, exclude_values) = request.mesh_routing_header_values()?;
    let target = parse_mesh_target_header(&target_values)?;
    let excluded = parse_mesh_exclude_header(&exclude_values)?;
    Ok((target, excluded))
}

/// Whether `request` carries a non-trivial `x-mesh-target`/`x-mesh-exclude`
/// ask. Malformed headers are already rejected with 400 wherever this
/// request is validated before reaching MoA, so a parse failure here (which
/// should not happen at this point) is treated as "no ask" rather than
/// re-rejecting. Shared by virtual-model and ordinary routing call sites
/// and the passive mesh-request path in `transport.rs`) so "does this
/// request want routing headers honored" is answered the same way in both
/// places. The request's trust policy is a separate question, answered by
/// `trust_policy::unprovable` with its own error code.
pub(crate) fn mesh_routing_headers_requested(request: &proxy::BufferedHttpRequest) -> bool {
    matches!(
        parse_mesh_routing_headers(request),
        Ok((target, excluded)) if target.is_some() || !excluded.is_empty()
    )
}

/// Whether `x-mesh-target`/`x-mesh-exclude` must force this request away from
/// local candidates: an exclude naming this node, or a target naming some
/// other peer. A target naming this node is not forcing -- it is allowed to
/// serve locally (checked by the ordinary local-candidate path) and only
/// changes whether `x-mesh-served-by` is echoed.
pub(super) fn mesh_headers_force_remote(
    self_id: iroh::EndpointId,
    target: Option<iroh::EndpointId>,
    excluded: &[iroh::EndpointId],
) -> bool {
    excluded.contains(&self_id) || target.is_some_and(|id| id != self_id)
}

#[cfg(test)]
#[path = "../ingress_tests/routing_headers.rs"]
mod tests;
