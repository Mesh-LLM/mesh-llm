//! Client selectors for private routing: `x-mesh-scope: private` or a
//! `@private` model suffix.
//!
//! Both mean the same thing: serve this request only on nodes signed by the
//! owner of the node the client is talking to. The selector is consumed here,
//! at the entry node: the suffix is removed from the model name before any
//! model lookup, so a peer advertising a model literally named `x@private`
//! can never be selected by it, and the header is stripped before a request
//! is forwarded. Whether a peer is "yours" is decided by its verified owner
//! certificate (`mesh::owner_scope`), never by a name.
//!
//! A trailing `@private` is reserved: it always means scope. A Hugging Face
//! revision literally named `private` must be referenced by commit instead.

use super::client_stream::ClientStream;
use super::request_parse::{BufferedHttpRequest, header_values_from_raw, rewrite_model_field};
use crate::mesh::{self, OwnerScope};
use crate::runtime::IngressType;

pub(crate) const MESH_SCOPE_HEADER: &str = "x-mesh-scope";
pub(crate) const PRIVATE_SCOPE: &str = "private";
pub(crate) const PRIVATE_MODEL_SUFFIX: &str = "@private";

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
        [value] if value.trim().eq_ignore_ascii_case(PRIVATE_SCOPE) => true,
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

/// `model` without its reserved `@private` suffix, when it has one.
pub(crate) fn private_model_base(model: Option<&str>) -> Option<&str> {
    model?
        .strip_suffix(PRIVATE_MODEL_SUFFIX)
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

/// [`resolve`] for a request entering this node's API ingress. A request
/// from a peer is already assigned here and runs locally, so it is never
/// scoped (its selector was stripped before it was forwarded). Returns `None`
/// once a rejection has been written.
pub(crate) async fn resolve_at_ingress(
    node: &mesh::Node,
    stream: ClientStream,
    mut request: BufferedHttpRequest,
    ingress_type: IngressType,
) -> Option<(ClientStream, BufferedHttpRequest, Option<OwnerScope>)> {
    if ingress_type == IngressType::RemoteQuicHttp {
        return Some((stream, request, None));
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(private_model_base(Some("qwen3-32b")), None);
        assert_eq!(private_model_base(Some("org/repo@main")), None);
        assert_eq!(private_model_base(None), None);
    }
}
