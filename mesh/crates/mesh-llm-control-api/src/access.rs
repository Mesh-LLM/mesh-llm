use std::net::{IpAddr, SocketAddr};

/// A security header cannot be read from an incomplete or malformed request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeaderParseError;
impl std::fmt::Display for HeaderParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("invalid management request header")
    }
}
impl std::error::Error for HeaderParseError {}

pub fn requires_trusted_local_access(method: &str, path: &str) -> bool {
    // Log history is local operator data. Keep this broad path classification
    // at the server boundary so all current and future `/api/logs/**` routes
    // are rejected before dispatch can touch the query facade or store.
    if path == "/api/logs" || path.starts_with("/api/logs/") {
        return true;
    }
    // The operator's local routing blocks: reading them shows which peers
    // this node avoids, and writing them changes routing.
    if path == "/api/peer-blocks" || path.starts_with("/api/peer-blocks/") {
        return true;
    }
    if path == "/mcp"
        || path.starts_with("/api/plugins")
        || (method == "POST"
            && (path == "/mesh/hook"
                || path == "/api/objects"
                || path.starts_with("/api/objects/")))
    {
        return true;
    }

    matches!(
        (method, path),
        ("GET", "/api/runtime/events/v1")
            | ("GET", "/api/runtime/endpoints")
            | ("GET", "/api/diagnostics/network")
            | ("GET", "/api/runtime/config-schema")
            | ("GET", "/api/runtime/config-control-state")
            | ("GET", "/api/runtime/control-bootstrap")
            | ("GET", "/api/runtime/intents")
            | ("GET", "/api/runtime/activity")
            | ("POST", "/api/runtime/control/get-config")
            | ("POST", "/api/runtime/control/scan-refresh")
            | ("POST", "/api/runtime/control/refresh-inventory")
            | ("POST", "/api/runtime/control/apply-config")
            | ("POST", "/api/runtime/control/load-model")
            | ("POST", "/api/runtime/control/unload-model")
            | ("POST", "/api/runtime/control/ensure-model")
            | ("POST", "/api/runtime/control/drain-model")
            | ("POST", "/api/runtime/control/kv-cache")
            | ("POST", "/api/runtime/kv-cache/prune")
            | ("GET", "/api/runtime/control/owner-control-token")
            | ("POST", "/api/runtime/config/validate")
            | ("POST", "/api/runtime/pick-directory")
            | ("POST", "/api/runtime/mesh-guardrails")
            | ("POST", "/api/runtime/models")
            | ("POST", "/api/model-interests")
            | ("PUT", "/api/runtime/activity/override")
    ) || (method == "DELETE"
        && (path == "/api/runtime/activity/override"
            || path == "/api/runtime/kv-cache"
            || path.starts_with("/api/runtime/models/")
            || path.starts_with("/api/runtime/instances/")
            || path.starts_with("/api/model-interests/")))
}

/// Cache mutations whose body must arrive as exactly the JSON media type.
///
/// The trusted-local gate accepts any loopback origin, so a page served by
/// another local port is already a trusted caller. The media type is what stops
/// it: a browser sends a cross-origin request without a CORS preflight only for
/// `application/x-www-form-urlencoded`, `multipart/form-data`, and
/// `text/plain`, so requiring `application/json` forces the preflight the
/// management listener never answers. Classified at the server boundary like
/// `requires_trusted_local_access`, so a future cache route is covered by this
/// table rather than by the handler that happens to exist today.
pub fn requires_json_content_type(method: &str, path: &str) -> bool {
    matches!(
        (method, path),
        ("POST", "/api/runtime/control/kv-cache")
            | ("POST", "/api/runtime/kv-cache/prune")
            | ("DELETE", "/api/runtime/kv-cache")
    )
}

/// Whether a management request declares the JSON media type. A missing or
/// unparseable `Content-Type` fails closed; an optional parameter such as
/// `charset` is still the same media type, while a lookalike token is not.
pub fn request_content_type_is_json(raw_request: &[u8]) -> Result<bool, HeaderParseError> {
    Ok(request_header(raw_request, "content-type")?.is_some_and(is_json_media_type))
}

fn is_json_media_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("application/json")
}

pub fn is_trusted_local_request(
    peer_addr: Option<SocketAddr>,
    origin: Option<&str>,
    host: Option<&str>,
) -> bool {
    let Some(peer_addr) = peer_addr else {
        return false;
    };
    if !is_loopback_ip(peer_addr.ip()) {
        return false;
    }

    host.is_none_or(is_trusted_local_authority) && origin.is_none_or(is_trusted_local_origin)
}

pub fn request_origin(raw_request: &[u8]) -> Result<Option<&str>, HeaderParseError> {
    request_header(raw_request, "origin")
}

pub fn request_host(raw_request: &[u8]) -> Result<Option<&str>, HeaderParseError> {
    request_header(raw_request, "host")
}

/// Returns the bearer token of the `Authorization` header when present.
///
/// Only the `Bearer <token>` scheme is recognized; any other scheme (or a
/// malformed value) yields `None` so callers treat it as a missing credential.
pub fn request_authorization_bearer(raw_request: &[u8]) -> Result<Option<&str>, HeaderParseError> {
    let Some(value) = request_header(raw_request, "authorization")? else {
        return Ok(None);
    };
    let Some(token) = value
        .get(7..)
        .filter(|_| value[..7].eq_ignore_ascii_case("Bearer "))
    else {
        return Ok(None);
    };
    let token = token.trim();
    Ok((!token.is_empty()).then_some(token))
}

fn request_header<'a>(
    raw_request: &'a [u8],
    name: &str,
) -> Result<Option<&'a str>, HeaderParseError> {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut headers);
    match request.parse(raw_request).map_err(|_| HeaderParseError)? {
        httparse::Status::Complete(_) => {}
        httparse::Status::Partial => return Err(HeaderParseError),
    }
    request
        .headers
        .iter()
        .find(|header| header.name.eq_ignore_ascii_case(name))
        .map(|header| std::str::from_utf8(header.value).map_err(|_| HeaderParseError))
        .transpose()
}

fn is_loopback_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.is_loopback(),
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip
                    .to_ipv4_mapped()
                    .is_some_and(|mapped| mapped.is_loopback())
        }
    }
}

fn is_trusted_local_origin(origin: &str) -> bool {
    let Ok(origin) = url::Url::parse(origin.trim()) else {
        return false;
    };
    if !matches!(origin.scheme(), "http" | "https") {
        return false;
    }
    origin
        .host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("localhost") || is_loopback_host(host))
}

fn is_trusted_local_authority(authority: &str) -> bool {
    let Ok(authority) = authority.trim().parse::<http::uri::Authority>() else {
        return false;
    };
    let host = authority.host();
    host.eq_ignore_ascii_case("localhost") || is_loopback_host(host)
}

fn is_loopback_host(host: &str) -> bool {
    host.trim_matches(['[', ']'])
        .parse::<IpAddr>()
        .is_ok_and(is_loopback_ip)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitive_management_routes_require_trusted_local_access() {
        for (method, path) in [
            ("GET", "/api/logs/requests"),
            ("POST", "/api/logs/requests/export"),
            ("POST", "/mcp"),
            ("GET", "/api/plugins"),
            ("POST", "/api/plugins/agents/tools/run"),
            ("POST", "/api/objects"),
            ("POST", "/api/objects/complete"),
            ("POST", "/mesh/hook"),
            ("GET", "/api/runtime/events/v1"),
            ("GET", "/api/runtime/endpoints"),
            ("GET", "/api/diagnostics/network"),
            ("POST", "/api/runtime/models"),
            ("DELETE", "/api/runtime/models/qwen"),
            ("DELETE", "/api/runtime/instances/instance-1"),
            ("POST", "/api/runtime/mesh-guardrails"),
            ("POST", "/api/model-interests"),
            ("DELETE", "/api/model-interests/qwen"),
            ("GET", "/api/peer-blocks"),
            ("POST", "/api/peer-blocks"),
            ("POST", "/api/peer-blocks/unblock"),
            ("POST", "/api/runtime/control/kv-cache"),
            ("POST", "/api/runtime/kv-cache/prune"),
            ("DELETE", "/api/runtime/kv-cache"),
            ("GET", "/api/runtime/control/owner-control-token"),
        ] {
            assert!(
                requires_trusted_local_access(method, path),
                "{method} {path} must be local-only"
            );
        }
    }

    #[test]
    fn read_only_status_routes_remain_network_accessible() {
        for path in [
            "/api/status",
            "/api/models",
            "/api/runtime",
            "/api/events",
            "/api/discover",
        ] {
            assert!(
                !requires_trusted_local_access("GET", path),
                "GET {path} should remain readable"
            );
        }
    }

    #[test]
    fn kv_cache_mutations_require_the_json_media_type() {
        for (method, path) in [
            ("POST", "/api/runtime/control/kv-cache"),
            ("POST", "/api/runtime/kv-cache/prune"),
            ("DELETE", "/api/runtime/kv-cache"),
        ] {
            assert!(
                requires_json_content_type(method, path),
                "{method} {path} must require application/json"
            );
        }
        assert!(!requires_json_content_type("GET", "/api/runtime/kv-cache"));
        assert!(!requires_json_content_type("POST", "/api/runtime/models"));
    }

    #[test]
    fn only_the_json_media_type_counts_as_json() {
        let request = |content_type: &str| {
            request_content_type_is_json(
                format!(
                    "POST /api/runtime/kv-cache/prune HTTP/1.1\r\nHost: localhost\r\nContent-Type: {content_type}\r\nContent-Length: 0\r\n\r\n"
                )
                .as_bytes(),
            )
        };
        assert_eq!(request("application/json"), Ok(true));
        assert_eq!(request("Application/JSON; charset=utf-8"), Ok(true));
        assert_eq!(request("application/json;charset=UTF-8"), Ok(true));
        assert_eq!(request("text/plain"), Ok(false));
        assert_eq!(request("application/x-www-form-urlencoded"), Ok(false));
        assert_eq!(request("multipart/form-data"), Ok(false));
        assert_eq!(request("application/jsonp"), Ok(false));
        assert_eq!(
            request_content_type_is_json(
                b"DELETE /api/runtime/kv-cache HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n"
            ),
            Ok(false)
        );
    }

    #[test]
    fn non_loopback_callers_are_rejected() {
        assert!(!is_trusted_local_request(
            Some(SocketAddr::from(([192, 0, 2, 10], 40123))),
            None,
            Some("localhost:3131")
        ));
    }

    #[test]
    fn browser_requests_from_untrusted_origins_are_rejected() {
        assert!(!is_trusted_local_request(
            Some(SocketAddr::from(([127, 0, 0, 1], 40123))),
            Some("https://attacker.example"),
            Some("localhost:3131")
        ));
        assert!(!is_trusted_local_request(
            Some(SocketAddr::from(([127, 0, 0, 1], 40123))),
            Some("null"),
            Some("localhost:3131")
        ));
        assert!(!is_trusted_local_request(
            Some(SocketAddr::from(([127, 0, 0, 1], 40123))),
            None,
            Some("attacker.example")
        ));
    }

    #[test]
    fn authorization_bearer_token_is_extracted() {
        let request = b"POST /api/runtime/control/apply-config HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer tok-123\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(request_authorization_bearer(request), Ok(Some("tok-123")));
        let lowercase = b"POST /x HTTP/1.1\r\nHost: localhost\r\nauthorization: bearer tok-456\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(request_authorization_bearer(lowercase), Ok(Some("tok-456")));
        let other_scheme = b"POST /x HTTP/1.1\r\nHost: localhost\r\nAuthorization: Basic abc\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(request_authorization_bearer(other_scheme), Ok(None));
        let absent = b"POST /x HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(request_authorization_bearer(absent), Ok(None));
        let empty = b"POST /x HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer \r\nContent-Length: 0\r\n\r\n";
        assert_eq!(request_authorization_bearer(empty), Ok(None));
    }

    #[test]
    fn origin_header_is_extracted_case_insensitively() {
        let request = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\noRiGiN: https://attacker.example\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(
            request_origin(request),
            Ok(Some("https://attacker.example"))
        );
    }

    #[test]
    fn malformed_security_header_is_rejected() {
        let request = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nOrigin: https://local\xff\r\n\r\n";
        assert_eq!(request_origin(request), Err(HeaderParseError));
    }

    #[test]
    fn trusted_local_callers_are_allowed() {
        let peer = Some(SocketAddr::from(([127, 0, 0, 1], 40123)));
        assert!(is_trusted_local_request(peer, None, Some("localhost:3131")));
        assert!(is_trusted_local_request(
            peer,
            Some("http://localhost:3131"),
            Some("localhost:3131")
        ));
        assert!(is_trusted_local_request(
            peer,
            Some("http://127.0.0.1:3131"),
            Some("127.0.0.1:3131")
        ));
        assert!(is_trusted_local_request(
            Some(SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], 40123))),
            Some("http://[::1]:3131"),
            Some("[::1]:3131")
        ));
    }
}
