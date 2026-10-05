use std::time::Duration;

use reqwest::header::HeaderMap;
use thiserror::Error;

/// Context captured from a failed HTTP response, carried by every HTTP-derived
/// variant of [`HFError`]. `request_id`, `error_code`, and `server_message` come
/// from the Hub's `X-Request-Id`, `X-Error-Code`, and `X-Error-Message` headers;
/// `server_message` falls back to the JSON body's `"error"` field.
///
/// `#[non_exhaustive]`: destructure with `..` and use the public constructors —
/// new fields may be added without a SemVer break.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HttpErrorContext {
    pub status: reqwest::StatusCode,
    pub url: String,
    pub request_id: Option<String>,
    pub error_code: Option<String>,
    /// Best-effort human-readable message from headers or JSON body.
    pub server_message: Option<String>,
    pub body: String,
}

impl HttpErrorContext {
    /// Build context from a non-success response. Drains the response body.
    pub(crate) async fn from_response(response: reqwest::Response) -> Self {
        let status = response.status();
        let url = response.url().to_string();
        let request_id = header_string(response.headers(), "x-request-id");
        let error_code = header_string(response.headers(), "x-error-code");
        let header_message = header_string(response.headers(), "x-error-message");
        let body = response.text().await.unwrap_or_default();
        let server_message = header_message.or_else(|| extract_json_error(&body));
        Self {
            status,
            url,
            request_id,
            error_code,
            server_message,
            body,
        }
    }
}

fn header_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(name)?.to_str().ok().map(|s| s.to_string())
}

fn extract_json_error(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value.get("error")?.as_str().map(|s| s.to_string())
}

/// Parse an integer-seconds `Retry-After` header. HTTP-date form is not supported.
pub(crate) fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    raw.trim().parse::<u64>().ok().map(Duration::from_secs)
}

fn format_http_detail(ctx: &HttpErrorContext) -> String {
    let mut detail = String::new();

    if let Some(message) = ctx.server_message.as_deref().filter(|m| !m.is_empty()) {
        detail.push_str(": ");
        detail.push_str(message);
    }

    let mut fields = vec![format!("url={}", ctx.url)];
    if let Some(rid) = &ctx.request_id {
        fields.push(format!("request_id={rid}"));
    }
    if let Some(code) = &ctx.error_code {
        fields.push(format!("error_code={code}"));
    }
    detail.push_str(&format!(" ({})", fields.join(", ")));

    detail
}

/// Error type returned by public `hf-hub` APIs.
///
/// Match on the specific variants first for common Hub cases such as
/// authentication, missing repos/files/revisions, forbidden access, and rate
/// limiting. Lower-level transport failures use [`HFError::Request`], while
/// unmapped HTTP responses use [`HFError::Http`].
///
/// This enum is `#[non_exhaustive]`: external matches must include a wildcard
/// arm so new variants can be added without a SemVer break.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum HFError {
    /// Non-success HTTP response that was not mapped to a more specific
    /// variant.
    #[error("HTTP error: {}{}", .context.status, format_http_detail(.context))]
    Http {
        /// Response metadata captured from the server.
        context: Box<HttpErrorContext>,
    },

    /// Authentication is required or the configured token was rejected.
    #[error("Authentication required{}", format_http_detail(.context))]
    AuthRequired {
        /// Response metadata captured from the server.
        context: Box<HttpErrorContext>,
    },

    /// Repository was not found.
    #[error("Repository not found: {repo_id}")]
    RepoNotFound {
        /// Repository id, usually in `owner/name` form.
        repo_id: String,
        /// Original HTTP response context, when available.
        context: Option<Box<HttpErrorContext>>,
    },

    /// Revision was not found within a repository.
    #[error("Revision not found: {revision} in {repo_id}")]
    RevisionNotFound {
        /// Repository id, usually in `owner/name` form.
        repo_id: String,
        /// Missing revision name or commit SHA.
        revision: String,
        /// Original HTTP response context, when available.
        context: Option<Box<HttpErrorContext>>,
    },

    /// File or path was not found within a repository or bucket.
    #[error("Entry not found: {path} in {repo_id}")]
    EntryNotFound {
        /// Missing repository-relative or bucket-relative path.
        path: String,
        /// Repository or bucket id associated with the lookup.
        repo_id: String,
        /// Original HTTP response context, when available.
        context: Option<Box<HttpErrorContext>>,
    },

    /// Bucket was not found.
    #[error("Bucket not found: {bucket_id}")]
    BucketNotFound {
        /// Bucket id in `owner/name` form.
        bucket_id: String,
        /// Original HTTP response context, when available.
        context: Option<Box<HttpErrorContext>>,
    },

    /// Credentials are valid, but the caller is not allowed to perform the
    /// operation.
    #[error("Forbidden{}", format_http_detail(.context))]
    Forbidden {
        /// Response metadata captured from the server.
        context: Box<HttpErrorContext>,
    },

    /// Server reported a conflict, such as an already-existing resource or a
    /// branch head mismatch.
    #[error("Conflict{}", format_http_detail(.context))]
    Conflict {
        /// Response metadata captured from the server.
        context: Box<HttpErrorContext>,
    },

    /// Request was rate limited by the Hub.
    #[error("Rate limited{}", format_http_detail(.context))]
    RateLimited {
        /// Parsed `Retry-After` delay, when the server returned one.
        retry_after: Option<Duration>,
        /// Response metadata captured from the server.
        context: Box<HttpErrorContext>,
    },

    /// File was not found in the local cache during a cache-only lookup.
    #[error("File not found in local cache: {path}")]
    LocalEntryNotFound {
        /// Missing cached path.
        path: String,
    },

    /// The operation requires the local cache, but caching is disabled.
    #[error(
        "Cache is not enabled — set cache_enabled(true) on HFClientBuilder, or set .local_dir(...) on the download builder"
    )]
    CacheNotEnabled,

    /// Timed out waiting for an on-disk cache lock.
    #[error("Cache lock timed out: {}", path.display())]
    CacheLockTimeout {
        /// Path of the lock file or locked entry.
        path: std::path::PathBuf,
    },

    /// Transport-level HTTP client error before a usable Hub response was
    /// produced.
    #[error("HTTP request error: {source}{}", .url.as_deref().map(|u| format!(" ({u})")).unwrap_or_default())]
    Request {
        /// Underlying `reqwest` transport error.
        #[source]
        source: reqwest::Error,
        /// Request URL, when known.
        url: Option<String>,
    },

    /// Filesystem I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// URL parsing error.
    #[error(transparent)]
    Url(#[from] url::ParseError),

    /// Caller provided an invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Raw diff parsing error.
    #[error(transparent)]
    DiffParse(#[from] crate::repository::HFDiffParseError),

    /// A xet (high-performance content-addressable) operation failed.
    ///
    /// The [`operation`](Self::Xet::operation) field identifies which step
    /// failed; [`source`](Self::Xet::source) carries the underlying error
    /// (typically a `xet_core` error or a `tokio` join failure).
    #[error("Xet {operation} failed: {source}")]
    Xet {
        /// Which xet operation produced the error.
        operation: XetOperation,
        /// Underlying error, type-erased so the `xet_core` types stay out of the public API.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Hub responded with success but the response is missing data the client
    /// needs to proceed (e.g., an `ETag` or `X-Repo-Commit` header that should
    /// always be present, or a `304 Not Modified` without a corresponding
    /// cached state).
    ///
    /// Distinct from [`Http`](Self::Http): the status code was a success, the
    /// shape of the response was wrong.
    #[error(
        "Hub response missing required data: {what}{}",
        url.as_deref().map(|u| format!(" ({u})")).unwrap_or_default()
    )]
    MalformedResponse {
        /// Short description of what was missing or unexpected.
        what: String,
        /// URL of the request that produced the bad response, when known.
        url: Option<String>,
    },

    /// Catch-all error for cases that do not fit another variant.
    #[error("{0}")]
    Other(String),
}

/// Identifies the xet operation associated with an [`HFError::Xet`] error.
///
/// New variants may be added in the future; matches must include a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum XetOperation {
    /// Building or refreshing the cached `XetSession` used for transfers.
    Session,
    /// Single-file or multi-file upload via xet.
    Upload,
    /// Single-file download via xet (writing to the cache or a local dir).
    Download,
    /// Multi-file batch download via xet.
    BatchDownload,
    /// Streaming download via xet (returns bytes to the caller as they arrive).
    StreamDownload,
    /// Bucket-level batch download via xet.
    BucketBatchDownload,
}

impl std::fmt::Display for XetOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            XetOperation::Session => "session",
            XetOperation::Upload => "upload",
            XetOperation::Download => "download",
            XetOperation::BatchDownload => "batch download",
            XetOperation::StreamDownload => "stream download",
            XetOperation::BucketBatchDownload => "bucket batch download",
        };
        f.write_str(s)
    }
}

impl From<reqwest::Error> for HFError {
    fn from(source: reqwest::Error) -> Self {
        let url = source.url().map(|u| u.to_string());
        HFError::Request { source, url }
    }
}

impl HFError {
    /// Construct an [`HFError::Xet`] from a typed source error.
    ///
    /// The source is type-erased into `Box<dyn Error + Send + Sync>` so that
    /// upstream `xet_core` types stay out of the public API; downstream
    /// matchers can use `source.downcast_ref::<T>()` if they need the concrete
    /// type.
    pub fn xet<E>(operation: XetOperation, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        HFError::Xet {
            operation,
            source: Box::new(source),
        }
    }

    /// Construct a [`HFError::MalformedResponse`] without a known URL.
    pub fn malformed_response(what: impl Into<String>) -> Self {
        HFError::MalformedResponse {
            what: what.into(),
            url: None,
        }
    }

    /// Construct a [`HFError::MalformedResponse`] with the originating URL attached.
    pub fn malformed_response_at(what: impl Into<String>, url: impl Into<String>) -> Self {
        HFError::MalformedResponse {
            what: what.into(),
            url: Some(url.into()),
        }
    }

    /// Returns true for errors that indicate transient network/server issues
    /// where falling back to a cached version is appropriate.
    pub(crate) fn is_transient(&self) -> bool {
        match self {
            // `reqwest::Error::is_connect` is not available on the wasm32 backend.
            HFError::Request { source, .. } => {
                let timeout = source.is_timeout();
                let connect = {
                    #[cfg(not(target_family = "wasm"))]
                    {
                        source.is_connect()
                    }
                    #[cfg(target_family = "wasm")]
                    {
                        false
                    }
                };
                timeout || connect
            }
            HFError::Http { context } => {
                matches!(context.status.as_u16(), 500 | 502 | 503 | 504)
            }
            _ => false,
        }
    }
}

/// Convenience alias used by public `hf-hub` APIs.
///
/// Equivalent to `Result<T, HFError>`.
pub type HFResult<T> = Result<T, HFError>;

/// Context for mapping HTTP 404 errors to specific HFError variants.
pub(crate) enum NotFoundContext {
    /// 404 means the repository does not exist
    Repo,
    /// 404 means the bucket does not exist
    Bucket,
    /// 404 means a file/path does not exist within the repo
    Entry { path: String },
    /// 404 means the revision does not exist
    Revision { revision: String },
    /// No special mapping — use generic Http error
    Generic,
}

#[cfg(test)]
mod tests {
    use reqwest::StatusCode;
    use reqwest::header::{HeaderMap, HeaderValue};

    use super::*;

    #[test]
    fn retry_after_parses_integer_seconds() {
        let mut h = HeaderMap::new();
        h.insert(reqwest::header::RETRY_AFTER, HeaderValue::from_static("42"));
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(42)));
    }

    #[test]
    fn retry_after_rejects_http_date() {
        let mut h = HeaderMap::new();
        h.insert(
            reqwest::header::RETRY_AFTER,
            HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn retry_after_absent() {
        let h = HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn header_string_case_insensitive() {
        let mut h = HeaderMap::new();
        h.insert("X-Request-Id", HeaderValue::from_static("abc123"));
        assert_eq!(
            header_string(&h, "x-request-id"),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn json_error_extraction() {
        let body = r#"{"error": "gated repository"}"#;
        assert_eq!(
            extract_json_error(body),
            Some("gated repository".to_string())
        );

        assert_eq!(extract_json_error("not json"), None);
        assert_eq!(extract_json_error(r#"{"other": "x"}"#), None);
    }

    fn ctx(status: u16) -> HttpErrorContext {
        HttpErrorContext {
            status: StatusCode::from_u16(status).unwrap(),
            url: "https://example".to_string(),
            request_id: None,
            error_code: None,
            server_message: None,
            body: String::new(),
        }
    }

    #[test]
    fn is_transient_classifies_http_statuses() {
        for s in [500u16, 502, 503, 504] {
            assert!(
                HFError::Http {
                    context: Box::new(ctx(s))
                }
                .is_transient(),
                "{s} should be transient"
            );
        }
        for s in [400u16, 401, 403, 404, 409, 429] {
            assert!(
                !HFError::Http {
                    context: Box::new(ctx(s))
                }
                .is_transient(),
                "{s} should not be transient"
            );
        }
    }

    #[test]
    fn display_includes_request_id_when_present() {
        let mut c = ctx(500);
        c.request_id = Some("req-xyz".to_string());
        let msg = HFError::Http {
            context: Box::new(c),
        }
        .to_string();
        assert!(msg.contains("req-xyz"), "display missing request_id: {msg}");
    }

    #[test]
    fn display_includes_server_message_when_present() {
        let mut c = ctx(400);
        c.server_message = Some("You can't create a commit with more than 1000 files".to_string());
        c.request_id = Some("req-xyz".to_string());
        let msg = HFError::Http {
            context: Box::new(c),
        }
        .to_string();
        assert!(
            msg.contains("You can't create a commit with more than 1000 files"),
            "display missing server_message: {msg}"
        );
        assert!(msg.contains("req-xyz"), "display missing request_id: {msg}");
    }

    #[test]
    fn display_surfaces_server_message_across_variants() {
        let make = |status| {
            let mut c = ctx(status);
            c.server_message = Some("server explanation".to_string());
            Box::new(c)
        };
        for err in [
            HFError::Http { context: make(400) },
            HFError::AuthRequired { context: make(401) },
            HFError::Forbidden { context: make(403) },
            HFError::Conflict { context: make(409) },
            HFError::RateLimited {
                retry_after: None,
                context: make(429),
            },
        ] {
            let msg = err.to_string();
            assert!(
                msg.contains("server explanation"),
                "variant dropped server_message: {msg}"
            );
        }
    }

    #[test]
    fn display_omits_message_section_when_absent() {
        // server_message is None on the bare ctx(); the rendered string should carry the URL in
        // the parenthetical without a dangling ": " where the message would go.
        let msg = HFError::Http {
            context: Box::new(ctx(500)),
        }
        .to_string();
        assert_eq!(
            msg,
            "HTTP error: 500 Internal Server Error (url=https://example)"
        );
    }

    #[tokio::test]
    async fn from_response_extracts_headers_and_body() {
        // Spin up a tiny server that returns 500 with custom headers + body.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await;
            let body = r#"{"error":"server said no"}"#;
            let resp = format!(
                "HTTP/1.1 500 Internal Server Error\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\
                 X-Request-Id: req-123\r\n\
                 X-Error-Code: GatedRepo\r\n\
                 X-Error-Message: acceptance required\r\n\
                 \r\n{}",
                body.len(),
                body
            );
            tokio::io::AsyncWriteExt::write_all(&mut sock, resp.as_bytes())
                .await
                .unwrap();
        });

        let response = reqwest::get(format!("http://{addr}/test")).await.unwrap();
        let ctx = HttpErrorContext::from_response(response).await;

        assert_eq!(ctx.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(ctx.url.ends_with("/test"));
        assert_eq!(ctx.request_id.as_deref(), Some("req-123"));
        assert_eq!(ctx.error_code.as_deref(), Some("GatedRepo"));
        assert_eq!(ctx.server_message.as_deref(), Some("acceptance required"));
        assert!(ctx.body.contains("server said no"));
    }

    #[tokio::test]
    async fn from_response_falls_back_to_json_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await;
            let body = r#"{"error":"repo is gated"}"#;
            let resp = format!(
                "HTTP/1.1 403 Forbidden\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\
                 \r\n{}",
                body.len(),
                body
            );
            tokio::io::AsyncWriteExt::write_all(&mut sock, resp.as_bytes())
                .await
                .unwrap();
        });

        let response = reqwest::get(format!("http://{addr}/")).await.unwrap();
        let ctx = HttpErrorContext::from_response(response).await;

        assert_eq!(ctx.request_id, None);
        assert_eq!(ctx.server_message.as_deref(), Some("repo is gated"));
    }
}
