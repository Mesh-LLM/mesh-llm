//! Client selectors for private routing: `x-mesh-scope: private` or a
//! `@private` model suffix. `incognito` is an alias for `private` in both.
//! They work on the automatic directive too: `auto@private`, `mesh@private`.
//!
//! Both mean the same thing: serve this request only on nodes signed by the
//! owner of the node the client is talking to. The selector is consumed here,
//! at the entry node: the suffix is removed from the model name before any
//! model lookup, so a peer advertising a model literally named `x@private`
//! can never be selected by it, and the header is stripped before a request
//! is forwarded. Whether a peer is "yours" is decided by its verified owner
//! certificate (`mesh::owner_scope`), never by a name.
//!
//! A trailing `@private` or `@incognito` is reserved: it always means scope.
//! A Hugging Face revision with either name must be referenced by commit
//! instead.

use super::client_stream::ClientStream;
use super::request_parse::{BufferedHttpRequest, header_values_from_raw, rewrite_model_field};
use crate::mesh::{self, OwnerScope};
use crate::runtime::IngressType;

pub(crate) const MESH_SCOPE_HEADER: &str = "x-mesh-scope";
pub(crate) const PRIVATE_SCOPE: &str = "private";
/// Header values that select private scope; the first is canonical.
const PRIVATE_SCOPE_ALIASES: [&str; 2] = [PRIVATE_SCOPE, "incognito"];
/// Reserved model-name suffixes that select private scope.
pub(crate) const PRIVATE_MODEL_SUFFIXES: [&str; 2] = ["@private", "@incognito"];

fn is_private_scope_value(value: &str) -> bool {
    let value = value.trim();
    PRIVATE_SCOPE_ALIASES
        .iter()
        .any(|alias| value.eq_ignore_ascii_case(alias))
}

/// Whether `name` ends in a reserved private suffix, so it can never be a
/// model a peer advertises.
pub(crate) fn has_private_model_suffix(name: &str) -> bool {
    PRIVATE_MODEL_SUFFIXES
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

/// Base URL that selects private scope for every request under it, so an
/// OpenAI client only needs `base_url = http://node:9337/private/v1`.
pub(crate) const PRIVATE_BASE_PREFIX: &str = "/private";

/// Split a `/private/v1/...` request target into its ordinary `/v1/...` path
/// and whether the private base was used. Other paths are returned as is.
pub(crate) fn split_private_base(target: &str) -> (String, bool) {
    match target.strip_prefix(PRIVATE_BASE_PREFIX) {
        Some(rest) if rest == "/v1" || rest.starts_with("/v1/") || rest.starts_with("/v1?") => {
            (rest.to_string(), true)
        }
        _ => (target.to_string(), false),
    }
}

/// The scope header a `/private/v1` request is rebuilt with.
pub(crate) fn private_scope_header_line() -> String {
    format!("{MESH_SCOPE_HEADER}: {PRIVATE_SCOPE}\r\n")
}

/// Why a private request cannot be served at this node.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ScopeRejection {
    pub(crate) status: u16,
    pub(crate) message: String,
}

/// Whether the client asked for private scope. A malformed header is an
/// error rather than silently routing the request publicly.
fn private_scope_requested(request: &BufferedHttpRequest) -> Result<bool, ScopeRejection> {
    let bad = |message: String| ScopeRejection {
        status: 400,
        message,
    };
    let header = match header_values_from_raw(&request.raw, MESH_SCOPE_HEADER)
        .map_err(|()| bad(format!("{MESH_SCOPE_HEADER} header contains invalid UTF-8")))?
        .as_slice()
    {
        [] => false,
        [value] if is_private_scope_value(value) => true,
        [value] => {
            return Err(bad(format!(
                "unsupported {MESH_SCOPE_HEADER} value '{}': the only scope is '{PRIVATE_SCOPE}'",
                value.trim()
            )));
        }
        _ => return Err(bad(format!("multiple {MESH_SCOPE_HEADER} headers"))),
    };
    Ok(header || private_model_base(request.model_name.as_deref()).is_some())
}

/// `model` without its reserved private suffix, when it has one.
pub(crate) fn private_model_base(model: Option<&str>) -> Option<&str> {
    let model = model?;
    PRIVATE_MODEL_SUFFIXES
        .iter()
        .find_map(|suffix| model.strip_suffix(suffix))
        .filter(|base| !base.is_empty())
}

/// Consume the private selector on a request entering at this node.
///
/// Returns the owner scope to route it in, `None` for an ordinary request,
/// or a rejection: private was asked for but this node has no verified owner
/// to scope to. Never falls back to public routing.
pub(crate) async fn resolve(
    node: &mesh::Node,
    request: &mut BufferedHttpRequest,
) -> Result<Option<OwnerScope>, ScopeRejection> {
    if !private_scope_requested(request)? {
        return Ok(None);
    }
    let Some(scope) = node.private_owner_scope().await else {
        return Err(ScopeRejection {
            status: 403,
            message: "private routing needs this node to be signed by your owner key \
                      (start it with `--owner-key`, after `mesh-llm auth init`)"
                .to_string(),
        });
    };
    if let Some(base) = private_model_base(request.model_name.as_deref()).map(str::to_string) {
        rewrite_model_field(request, &base);
    }
    Ok(Some(scope))
}

/// [`resolve`] for a request entering this node's API ingress. Returns
/// `None` once a rejection has been written.
///
/// A request from a peer is already assigned here and runs locally. It is
/// private only when the peer that forwarded it marked it so and is signed by
/// this node's own owner: then this node serves it only from what stays in
/// that owner's scope. A stranger's marker is ignored; it cannot widen
/// anything, and a peer request never routes onward.
pub(crate) async fn resolve_at_ingress(
    node: &mesh::Node,
    stream: ClientStream,
    mut request: BufferedHttpRequest,
    ingress_type: IngressType,
    requested_by: Option<iroh::EndpointId>,
) -> Option<(ClientStream, BufferedHttpRequest, Option<OwnerScope>)> {
    if ingress_type == IngressType::RemoteQuicHttp {
        let scope = peer_private_scope(node, &request, requested_by).await;
        return Some((stream, request, scope));
    }
    match resolve(node, &mut request).await {
        Ok(scope) => Some((stream, request, scope)),
        Err(rejection) => {
            // shortcut: written before the request lifecycle is claimed, so
            // this rejection is not in the request log.
            let _ = super::response::send_error(stream, rejection.status, &rejection.message).await;
            None
        }
    }
}

async fn peer_private_scope(
    node: &mesh::Node,
    request: &BufferedHttpRequest,
    requested_by: Option<iroh::EndpointId>,
) -> Option<OwnerScope> {
    let marked = matches!(
        header_values_from_raw(&request.raw, MESH_SCOPE_HEADER).as_deref(),
        Ok([value]) if is_private_scope_value(value)
    );
    if !marked || !node.peer_shares_my_owner(requested_by?).await {
        return None;
    }
    node.private_owner_scope().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_private_base_url_maps_onto_v1() {
        for (target, path) in [
            ("/private/v1/chat/completions", "/v1/chat/completions"),
            ("/private/v1/models", "/v1/models"),
            ("/private/v1", "/v1"),
            ("/private/v1?x=1", "/v1?x=1"),
        ] {
            assert_eq!(split_private_base(target), (path.to_string(), true));
        }
        for target in [
            "/v1/models",
            "/private",
            "/private/v10/models",
            "/privatev1/models",
        ] {
            assert_eq!(split_private_base(target), (target.to_string(), false));
        }
    }

    #[test]
    fn the_suffix_is_reserved_and_stripped_once() {
        assert_eq!(
            private_model_base(Some("qwen3-32b@private")),
            Some("qwen3-32b")
        );
        assert_eq!(
            private_model_base(Some("org/repo:Q4_K_M@private")),
            Some("org/repo:Q4_K_M")
        );
        assert_eq!(
            private_model_base(Some("x@private@private")),
            Some("x@private")
        );
        assert_eq!(private_model_base(Some("@private")), None);
        assert_eq!(private_model_base(Some("@incognito")), None);
        assert_eq!(
            private_model_base(Some("qwen3-32b@incognito")),
            Some("qwen3-32b")
        );
        assert_eq!(private_model_base(Some("auto@private")), Some("auto"));
        assert_eq!(private_model_base(Some("mesh@incognito")), Some("mesh"));
        assert_eq!(private_model_base(Some("qwen3-32b")), None);
        assert_eq!(private_model_base(Some("org/repo@main")), None);
        assert_eq!(private_model_base(None), None);
    }

    #[test]
    fn incognito_is_an_alias_and_names_with_either_suffix_are_reserved() {
        for value in ["private", "incognito", " Incognito "] {
            assert!(is_private_scope_value(value));
        }
        for value in ["public", "", "incog"] {
            assert!(!is_private_scope_value(value));
        }
        assert!(has_private_model_suffix("x@private"));
        assert!(has_private_model_suffix("x@incognito"));
        assert!(!has_private_model_suffix("org/repo@main"));
    }
}
