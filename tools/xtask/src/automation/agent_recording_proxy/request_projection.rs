//! URL and capture contracts for the OpenCode recording proxy.
use crate::command::DynResult;
use serde::Serialize;
use serde_json::Value;
use url::Url;

pub(super) const BODY_LIMIT: usize = 32 * 1024 * 1024;

pub(super) struct Endpoint {
    base: String,
    prefix: String,
}

impl Endpoint {
    pub(super) fn parse(text: &str) -> DynResult<Self> {
        let url = Url::parse(text)?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err("recording proxy upstream must be an HTTP(S) API base URL without credentials, query or fragment".into());
        }
        Ok(Self {
            base: url.as_str().trim_end_matches('/').to_owned(),
            prefix: url.path().trim_end_matches('/').to_owned(),
        })
    }

    pub(super) fn target(&self, path_and_query: &str) -> DynResult<String> {
        if !path_and_query.starts_with('/')
            || path_and_query.starts_with("//")
            || path_and_query.contains('#')
            || path_and_query.chars().any(char::is_control)
        {
            return Err("recording proxy requires an origin-form request path".into());
        }
        let path = path_and_query.split('?').next().unwrap();
        let matches = |prefix: &str| {
            !prefix.is_empty() && (path == prefix || path.starts_with(&format!("{prefix}/")))
        };
        // The listener advertises /v1 even when the upstream API base is nested.
        // Remove either the full upstream prefix or the advertised version prefix once.
        let suffix = if matches(&self.prefix) {
            &path_and_query[self.prefix.len()..]
        } else if !self.prefix.is_empty() && matches("/v1") {
            &path_and_query[3..]
        } else {
            path_and_query
        };
        Ok(format!("{}{suffix}", self.base))
    }
}

#[derive(Serialize)]
struct CapturedHeaders<'a> {
    #[serde(rename = "content-type")]
    content_type: Option<&'a str>,
    accept: Option<&'a str>,
}

#[derive(Serialize)]
struct Capture<'a> {
    method: &'a str,
    path: &'a str,
    body: Option<Value>,
    headers: CapturedHeaders<'a>,
}

pub(super) fn capture(
    method: &str,
    path: &str,
    body: &[u8],
    content_type: Option<&str>,
    accept: Option<&str>,
) -> DynResult<Vec<u8>> {
    if body.len() > BODY_LIMIT {
        return Err("recording proxy request exceeds 32 MiB".into());
    }
    let record = Capture {
        method,
        path,
        body: serde_json::from_slice(body).ok(),
        headers: CapturedHeaders {
            content_type,
            accept,
        },
    };
    let mut bytes = serde_json::to_vec(&record)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(super) fn forward_request_header(name: &str) -> bool {
    ![
        "host",
        "content-length",
        "accept-encoding",
        "connection",
        "transfer-encoding",
        "proxy-connection",
        "keep-alive",
        "te",
        "trailer",
        "upgrade",
    ]
    .iter()
    .any(|excluded| name.eq_ignore_ascii_case(excluded))
}

pub(super) fn forward_response_header(name: &str) -> bool {
    ![
        "content-length",
        "connection",
        "transfer-encoding",
        "content-encoding",
        "proxy-connection",
        "keep-alive",
        "te",
        "trailer",
        "upgrade",
    ]
    .iter()
    .any(|excluded| name.eq_ignore_ascii_case(excluded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn api_prefix_is_removed_once_without_losing_path_or_query() {
        let endpoint = Endpoint::parse("http://127.0.0.1:9337/v1/").unwrap();
        assert_eq!(
            endpoint.target("/v1/models").unwrap(),
            "http://127.0.0.1:9337/v1/models"
        );
        assert_eq!(
            endpoint.target("/v1/chat/completions?trace=1").unwrap(),
            "http://127.0.0.1:9337/v1/chat/completions?trace=1"
        );
        assert_eq!(
            endpoint.target("/models").unwrap(),
            "http://127.0.0.1:9337/v1/models"
        );
        assert_eq!(
            endpoint.target("/v1/v1/models").unwrap(),
            "http://127.0.0.1:9337/v1/v1/models"
        );
        assert_eq!(
            endpoint.target("/v1?trace=1").unwrap(),
            "http://127.0.0.1:9337/v1?trace=1"
        );
        let nested = Endpoint::parse("https://example.invalid/tenant/v1").unwrap();
        for path in ["/v1/models", "/tenant/v1/models"] {
            assert_eq!(
                nested.target(path).unwrap(),
                "https://example.invalid/tenant/v1/models"
            );
        }
        let root = Endpoint::parse("https://[::1]:8443/").unwrap();
        assert_eq!(
            root.target("/v1/models").unwrap(),
            "https://[::1]:8443/v1/models"
        );
    }

    #[test]
    fn upstream_and_request_path_shapes_are_admitted_explicitly() {
        for base in [
            "file:///tmp/input",
            "http://user:password@localhost/v1",
            "http://localhost/v1?secret=1",
            "http://localhost/v1#fragment",
        ] {
            assert!(Endpoint::parse(base).is_err());
        }
        let endpoint = Endpoint::parse("https://example.invalid/v1").unwrap();
        for path in [
            "https://elsewhere.invalid/v1",
            "//elsewhere.invalid/v1",
            "models",
            "/models#fragment",
            "/models\n",
        ] {
            assert!(endpoint.target(path).is_err());
        }
    }

    #[test]
    fn capture_retains_consumer_schema_and_only_selected_headers() {
        let record = capture(
            "POST",
            "/v1/chat/completions",
            br#"{"stream":true,"messages":[{"role":"user","content":"hello"}]}"#,
            Some("application/json"),
            Some("text/event-stream"),
        )
        .unwrap();
        assert_eq!(record.last(), Some(&b'\n'));
        let value: Value = serde_json::from_slice(&record).unwrap();
        assert_eq!(
            value,
            json!({"method":"POST","path":"/v1/chat/completions","body":{"stream":true,"messages":[{"role":"user","content":"hello"}]},"headers":{"content-type":"application/json","accept":"text/event-stream"}})
        );
        assert!(!String::from_utf8(record).unwrap().contains("authorization"));
    }

    #[test]
    fn absent_or_invalid_json_body_remains_null_capture_evidence() {
        for body in [&b""[..], &b"not JSON"[..], &[0xff][..]] {
            let bytes = capture("GET", "/v1/models", body, None, None).unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["body"], Value::Null);
            assert_eq!(value["headers"], json!({"content-type":null,"accept":null}));
        }
    }

    #[test]
    fn forwarding_excludes_transport_headers_and_preserves_api_headers() {
        for name in [
            "Content-Length",
            "Connection",
            "Transfer-Encoding",
            "Keep-Alive",
            "Upgrade",
        ] {
            assert!(!forward_request_header(name));
            assert!(!forward_response_header(name));
        }
        assert!(!forward_request_header("Host"));
        assert!(!forward_response_header("Content-Encoding"));
        for name in ["Content-Type", "Accept", "Authorization", "X-Request-Id"] {
            assert!(forward_request_header(name));
        }
        for name in ["Content-Type", "X-Request-Id", "Retry-After"] {
            assert!(forward_response_header(name));
        }
    }
}
