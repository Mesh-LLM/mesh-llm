use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue, USER_AGENT};
use tracing::debug;
use url::Url;

use crate::constants;
use crate::error::{HFError, HFResult, HttpErrorContext, NotFoundContext};
use crate::retry::{self, RetryConfig};

/// Append `path` to `url` as percent-encoded URL path segments.
///
/// Splits `path` on `/` and pushes each non-empty piece via
/// [`Url::path_segments_mut`], which percent-encodes characters that aren't
/// valid in a path segment (e.g. `#`, `?`, ` `, `%`).
///
/// Returns an error only if `url` is a cannot-be-a-base URL (e.g. `mailto:`,
/// `data:`); all URLs hf-hub constructs are http(s) and so this never fails
/// in practice.
pub(crate) fn append_path_segments(url: &mut Url, path: &str) -> HFResult<()> {
    let url_str = url.as_str().to_string();
    let mut segments = url
        .path_segments_mut()
        .map_err(|()| HFError::InvalidParameter(format!("unexpected non-base URL: {url_str}")))?;
    for seg in path.split('/').filter(|s| !s.is_empty()) {
        segments.push(seg);
    }
    Ok(())
}

/// Everything except RFC 3986 unreserved characters (`A-Za-z0-9-._~`), mirroring
/// Python's `urllib.parse.quote(s, safe="")` as used by `huggingface_hub`.
const REF_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

/// Percent-encode a git revision (branch, tag, or commit SHA) for use as a single
/// URL path segment.
///
/// Unlike [`append_path_segments`], `/` is encoded too (`feature/x` → `feature%2Fx`)
/// so the revision occupies exactly one segment — the Hub API expects revisions
/// fully encoded. Also safe for `<base>..<head>` compare specs: `.` is unreserved,
/// so the `..` separator stays literal while each side is encoded.
pub(crate) fn encode_ref(revision: &str) -> Cow<'_, str> {
    utf8_percent_encode(revision, REF_ENCODE_SET).into()
}

/// Split a repo or bucket id into its `(owner, name)` parts.
///
/// A fully-qualified id (`"owner/name"`) splits on the first `/`. A short-form
/// id with no owner (`"gpt2"`) yields an empty owner, matching the `owner, name`
/// argument pair taken by [`HFClient::model`], [`HFClient::dataset`],
/// [`HFClient::space`], and [`HFClient::repository`]. Only the first `/` is a
/// separator, so nested names (`"owner/sub/name"`) keep their tail intact
/// (`("owner", "sub/name")`).
///
/// ```
/// use hf_hub::split_id;
///
/// assert_eq!(split_id("openai-community/gpt2"), ("openai-community", "gpt2"));
/// assert_eq!(split_id("gpt2"), ("", "gpt2"));
/// ```
pub fn split_id(id: &str) -> (&str, &str) {
    match id.split_once('/') {
        Some((owner, name)) => (owner, name),
        None => ("", id),
    }
}

/// Whether a `Location` header value is a relative reference (no scheme, no
/// host), like `/Snowflake/repo/resolve/main/config.json`. Protocol-relative
/// URLs (`//host/path`) carry a host and are treated as absolute.
#[cfg(test)]
fn is_relative_location(location: &str) -> bool {
    Url::parse(location) == Err(url::ParseError::RelativeUrlWithoutBase)
        && !location.starts_with("//")
}

/// Async client for the Hugging Face Hub API.
///
/// `HFClient` wraps an `Arc<HFClientInner>` so it is cheap to clone — all clones
/// share the same underlying HTTP connection pool, token, and configuration.
///
/// # Creating a client
///
/// ```rust,no_run
/// use hf_hub::HFClient;
///
/// // Reads token and settings from the environment (HF_TOKEN, HF_ENDPOINT, …).
/// let client = HFClient::new()?;
///
/// // Or configure explicitly:
/// let client = HFClient::builder().token("hf_…").endpoint("https://huggingface.co").build()?;
/// # Ok::<(), hf_hub::HFError>(())
/// ```
pub struct HFClient {
    pub(crate) inner: Arc<HFClientInner>,
}

impl Clone for HFClient {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for HFClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("HFClient");
        s.field("endpoint", &self.inner.endpoint);
        s.field("token_set", &self.inner.token.is_some());
        s.field("cache_enabled", &self.inner.cache_enabled);
        if self.inner.cache_enabled {
            s.field("cache_dir", &self.inner.cache_dir);
        }
        s.finish()
    }
}

pub(crate) struct HFClientInner {
    pub(crate) client: reqwest::Client,
    // The wasm32 reqwest backend ignores `redirect()` configuration — the browser
    // (or fetch in Node) handles redirects opaquely — so we only carry a separate
    // no-redirect client on native targets.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) no_redirect_client: reqwest::Client,
    pub(crate) retry_config: RetryConfig,
    pub(crate) endpoint: String,
    pub(crate) token: Option<String>,
    pub(crate) cache_dir: std::path::PathBuf,
    pub(crate) cache_enabled: bool,
    pub(crate) xet_state: std::sync::Mutex<crate::xet::XetState>,
}

/// Builder for [`HFClient`].
///
/// Get one via [`HFClient::builder()`] or [`HFClientBuilder::new()`].
/// Call [`build()`](HFClientBuilder::build) when all options are set.
pub struct HFClientBuilder {
    endpoint: Option<String>,
    token: Option<String>,
    user_agent: Option<String>,
    headers: Option<HeaderMap>,
    client: Option<reqwest::Client>,
    cache_dir: Option<std::path::PathBuf>,
    cache_enabled: Option<bool>,
    retry_max_attempts: Option<usize>,
    retry_base_delay: Option<Duration>,
}

impl HFClientBuilder {
    /// Creates a builder with all options unset; defaults are applied at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            endpoint: None,
            token: None,
            user_agent: None,
            headers: None,
            client: None,
            cache_dir: None,
            cache_enabled: None,
            retry_max_attempts: None,
            retry_base_delay: None,
        }
    }

    /// Overrides the Hub base URL (default: `https://huggingface.co`, or `HF_ENDPOINT` env var).
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Sets the authentication token. Without this, the client falls back to the `HF_TOKEN` env
    /// var and the cached token file written by `huggingface-cli login`.
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Overrides the `User-Agent` header sent with every request.
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Merges additional default headers into every request. Ignored when a custom
    /// [`client`](Self::client) is supplied (configure headers on that client directly).
    pub fn headers(mut self, headers: HeaderMap) -> Self {
        self.headers = Some(headers);
        self
    }

    /// Supplies a pre-configured `reqwest::Client`. Retry middleware is still applied on top.
    /// The caller is responsible for any default headers (including `User-Agent`) on this client;
    /// the [`headers`](Self::headers) and [`user_agent`](Self::user_agent) options are ignored.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Sets the local cache directory. Defaults to `HF_HUB_CACHE` → `$HF_HOME/hub` →
    /// `~/.cache/huggingface/hub`.
    pub fn cache_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Enables or disables the local file cache. Caching is on by default.
    pub fn cache_enabled(mut self, enabled: bool) -> Self {
        self.cache_enabled = Some(enabled);
        self
    }

    /// Overrides the maximum number of retry attempts after an initial failure (default: 3).
    pub fn retry_max_attempts(mut self, n: usize) -> Self {
        self.retry_max_attempts = Some(n);
        self
    }

    /// Overrides the base delay for exponential backoff between retries (default: 100ms).
    pub fn retry_base_delay(mut self, delay: Duration) -> Self {
        self.retry_base_delay = Some(delay);
        self
    }

    /// Builds the [`HFClient`].
    ///
    /// # Errors
    ///
    /// Returns an error if the endpoint URL is not a valid URL or if the `reqwest` client
    /// cannot be constructed (e.g., an invalid `User-Agent` string was provided).
    pub fn build(self) -> HFResult<HFClient> {
        let endpoint = self
            .endpoint
            .or_else(|| std::env::var(constants::HF_ENDPOINT).ok())
            .unwrap_or_else(|| constants::DEFAULT_HF_ENDPOINT.to_string());

        let _ = url::Url::parse(&endpoint)?;

        let token = self.token.or_else(resolve_token);

        let cache_dir = self.cache_dir.unwrap_or_else(constants::resolve_cache_dir);

        let mut default_headers = self.headers.unwrap_or_default();

        let user_agent = self.user_agent.unwrap_or_else(|| {
            let ua_origin = std::env::var(constants::HF_HUB_USER_AGENT_ORIGIN).ok();
            match ua_origin {
                Some(origin) => format!("hf-hub/{}; {origin}", env!("CARGO_PKG_VERSION")),
                None => format!("hf-hub/{}", env!("CARGO_PKG_VERSION")),
            }
        });
        default_headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&user_agent).map_err(|e| {
                HFError::InvalidParameter(format!("invalid user agent {user_agent:?}: {e}"))
            })?,
        );

        let client = match self.client {
            Some(c) => c,
            None => reqwest::Client::builder()
                .default_headers(default_headers.clone())
                .build()?,
        };

        #[cfg(not(target_family = "wasm"))]
        let no_redirect_client = reqwest::Client::builder()
            .default_headers(default_headers)
            .redirect(reqwest::redirect::Policy::none())
            .build()?;

        let retry_config = RetryConfig {
            max_attempts: self
                .retry_max_attempts
                .unwrap_or(retry::DEFAULT_MAX_ATTEMPTS),
            base_delay: self.retry_base_delay.unwrap_or(retry::DEFAULT_BASE_DELAY),
        };

        Ok(HFClient {
            inner: Arc::new(HFClientInner {
                client,
                #[cfg(not(target_family = "wasm"))]
                no_redirect_client,
                retry_config,
                endpoint: endpoint.trim_end_matches('/').to_string(),
                token,
                cache_dir,
                cache_enabled: self.cache_enabled.unwrap_or(true),
                xet_state: std::sync::Mutex::new(crate::xet::XetState::default()),
            }),
        })
    }

    /// Builds the [`crate::blocking::HFClientSync`].
    ///
    /// Requires the `blocking` feature and is equivalent to calling
    /// [`build`](Self::build) and then [`HFClientSync::from_inner`](crate::HFClientSync::from_inner).
    ///
    /// # Errors
    ///
    /// Returns an error if the endpoint URL is not a valid URL, if the `reqwest` client
    /// cannot be constructed (e.g., an invalid `User-Agent` string was provided), or if the
    /// tokio runtime handle could not be correctly created for the blocking client.
    #[cfg(feature = "blocking")]
    #[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
    pub fn build_sync(self) -> HFResult<crate::blocking::HFClientSync> {
        let async_client = self.build()?;
        let client = crate::blocking::HFClientSync::from_inner(async_client)?;
        Ok(client)
    }
}

impl Default for HFClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HFClient {
    /// Creates a client with default settings, reading the token and endpoint from the
    /// environment. Equivalent to `HFClient::builder().build()`.
    ///
    /// # Errors
    ///
    /// Fails if the resolved endpoint URL is invalid or the HTTP client cannot be built.
    pub fn new() -> HFResult<Self> {
        HFClientBuilder::new().build()
    }

    /// Returns an [`HFClientBuilder`] for fine-grained configuration.
    pub fn builder() -> HFClientBuilder {
        HFClientBuilder::new()
    }

    pub(crate) fn http_client(&self) -> &reqwest::Client {
        &self.inner.client
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn no_redirect_client(&self) -> &reqwest::Client {
        &self.inner.no_redirect_client
    }

    pub(crate) fn retry_config(&self) -> &RetryConfig {
        &self.inner.retry_config
    }

    /// Hub base URL this client targets, with any trailing slash trimmed.
    ///
    /// Resolved at [`build`](HFClientBuilder::build) time from
    /// [`HFClientBuilder::endpoint`] → `HF_ENDPOINT` → the default
    /// (`https://huggingface.co`).
    pub fn endpoint(&self) -> &str {
        &self.inner.endpoint
    }

    /// Local cache directory used for downloaded files.
    ///
    /// Resolved at [`build`](HFClientBuilder::build) time from
    /// [`HFClientBuilder::cache_dir`] → `HF_HUB_CACHE` → `$HF_HOME/hub` →
    /// `~/.cache/huggingface/hub`. Returned even when caching is disabled —
    /// see [`cache_enabled`](Self::cache_enabled) to check that.
    pub fn cache_dir(&self) -> &std::path::Path {
        &self.inner.cache_dir
    }

    /// Whether the local file cache is enabled. Set via
    /// [`HFClientBuilder::cache_enabled`]; defaults to `true`.
    pub fn cache_enabled(&self) -> bool {
        self.inner.cache_enabled
    }

    /// Build authorization headers for requests
    pub(crate) fn auth_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(ref token) = self.inner.token
            && let Ok(val) = HeaderValue::from_str(&format!("Bearer {token}"))
        {
            headers.insert(AUTHORIZATION, val);
        }
        headers
    }

    /// Build a URL for the API: {endpoint}/api/{api_segment}/{repo_id}.
    ///
    /// `api_segment` is the plural form of the repo type — `"models"`, `"datasets"`,
    /// `"spaces"`, or `"kernels"`. Pass [`crate::repository::RepoType::plural`] for the
    /// concrete repo type at the call site.
    pub(crate) fn api_url(&self, api_segment: &str, repo_id: &str) -> String {
        format!("{}/api/{}/{}", self.endpoint(), api_segment, repo_id)
    }

    /// Build a download URL: `{endpoint}/{url_prefix}{repo_id}/resolve/{revision}/{filename}`.
    ///
    /// `url_prefix` is `""` for models or `"<plural>/"` for the other repo types. Pass
    /// [`crate::repository::RepoType::url_prefix`] for the concrete repo type at the call
    /// site. `revision` is fully percent-encoded (including `/`) so it occupies exactly one
    /// path segment; `filename` is percent-encoded path-segment by path-segment, so values
    /// like `subdir/file name #1.bin` produce a correctly escaped URL.
    pub(crate) fn download_url(
        &self,
        url_prefix: &str,
        repo_id: &str,
        revision: &str,
        filename: &str,
    ) -> HFResult<String> {
        let base = format!(
            "{}/{}{}/resolve/{}",
            self.endpoint(),
            url_prefix,
            repo_id,
            encode_ref(revision)
        );
        let mut url = Url::parse(&base)?;
        append_path_segments(&mut url, filename)?;
        Ok(url.into())
    }

    /// Build a bucket API URL: `{endpoint}/api/buckets/{bucket_id}`
    pub(crate) fn bucket_api_url(&self, bucket_id: &str) -> String {
        format!("{}/api/buckets/{}", self.endpoint(), bucket_id)
    }

    /// Check an HTTP response and map error status codes to HFError variants.
    /// Returns the response on success (2xx).
    ///
    /// `repo_id` and `not_found_ctx` control how 404s are mapped:
    /// - `NotFoundContext::Repo` → `HFError::RepoNotFound`
    /// - `NotFoundContext::Entry { path }` → `HFError::EntryNotFound`
    /// - `NotFoundContext::Revision { revision }` → `HFError::RevisionNotFound`
    /// - `NotFoundContext::Generic` → `HFError::Http`
    pub(crate) async fn check_response(
        &self,
        response: reqwest::Response,
        repo_id: Option<&str>,
        not_found_ctx: NotFoundContext,
    ) -> HFResult<reqwest::Response> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let retry_after = if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            crate::error::parse_retry_after(response.headers())
        } else {
            None
        };
        let context = Box::new(HttpErrorContext::from_response(response).await);
        let repo_id_str = repo_id.unwrap_or("").to_string();

        match status.as_u16() {
            401 => Err(HFError::AuthRequired { context }),
            403 => Err(HFError::Forbidden { context }),
            404 => match not_found_ctx {
                NotFoundContext::Repo => Err(HFError::RepoNotFound {
                    repo_id: repo_id_str,
                    context: Some(context),
                }),
                NotFoundContext::Bucket => Err(HFError::BucketNotFound {
                    bucket_id: repo_id_str,
                    context: Some(context),
                }),
                NotFoundContext::Entry { path } => Err(HFError::EntryNotFound {
                    path,
                    repo_id: repo_id_str,
                    context: Some(context),
                }),
                NotFoundContext::Revision { revision } => Err(HFError::RevisionNotFound {
                    revision,
                    repo_id: repo_id_str,
                    context: Some(context),
                }),
                NotFoundContext::Generic => Err(HFError::Http { context }),
            },
            409 => Err(HFError::Conflict { context }),
            429 => Err(HFError::RateLimited {
                retry_after,
                context,
            }),
            _ => Err(HFError::Http { context }),
        }
    }

    /// Get or lazily create the cached XetSession.
    ///
    /// Returns `(session, generation)`. The generation is an opaque counter
    /// that identifies which session instance this is. Pass it to
    /// [`replace_xet_session`](Self::replace_xet_session) so that only the
    /// caller that observed the error triggers a replacement — concurrent
    /// callers that already got a session won't clobber it.
    pub(crate) fn xet_session(&self) -> HFResult<(xet::xet_session::XetSession, u64)> {
        let mut guard = self
            .inner
            .xet_state
            .lock()
            .map_err(|e| HFError::Other(format!("xet session mutex poisoned: {e}")))?;

        if let Some(ref session) = guard.session {
            return Ok((session.clone(), guard.generation));
        }

        #[cfg(not(target_family = "wasm"))]
        let builder = xet::xet_session::XetSessionBuilder::new();
        // Cap transfer concurrency on wasm to avoid exhausting linear memory.
        #[cfg(target_family = "wasm")]
        let builder = {
            let mut config = xet::xet_session::XetConfig::new();
            config.client.ac_max_upload_concurrency = 8;
            config.client.ac_max_download_concurrency = 8;
            xet::xet_session::XetSessionBuilder::new_with_config(config)
        };
        let session = builder
            .build()
            .map_err(|e| HFError::xet(crate::error::XetOperation::Session, e))?;
        guard.session = Some(session.clone());
        guard.generation += 1;
        Ok((session, guard.generation))
    }

    /// Replace the cached XetSession only if the generation matches.
    ///
    /// Called by xet call sites when a factory method returns an error.
    /// The generation check ensures that if another thread already replaced
    /// the session, this call is a no-op rather than discarding the fresh one.
    pub(crate) fn replace_xet_session(&self, generation: u64, err: &xet::error::XetError) {
        tracing::warn!(error = %err, generation, "replacing cached XetSession");
        let Ok(mut guard) = self.inner.xet_state.lock() else {
            return;
        };
        if guard.generation == generation {
            guard.session = None;
        }
    }
}

/// Resolve token from environment or a token file.
/// Priority: HF_TOKEN env → HF_TOKEN_PATH file → $HF_HOME/token file.
fn resolve_token() -> Option<String> {
    if let Ok(val) = std::env::var(constants::HF_HUB_DISABLE_IMPLICIT_TOKEN)
        && !val.is_empty()
    {
        debug!("implicit token disabled via HF_HUB_DISABLE_IMPLICIT_TOKEN");
        return None;
    }

    if let Ok(token) = std::env::var(constants::HF_TOKEN)
        && !token.is_empty()
    {
        debug!("resolved token from HF_TOKEN env var");
        return Some(token);
    }

    if let Ok(path) = std::env::var(constants::HF_TOKEN_PATH)
        && let Ok(token) = std::fs::read_to_string(&path)
    {
        let token = token.trim().to_string();
        if !token.is_empty() {
            debug!("resolved token from HF_TOKEN_PATH file");
            return Some(token);
        }
    }

    let hf_home = constants::hf_home();
    let token_path = hf_home.join(constants::TOKEN_FILENAME);
    if let Ok(token) = std::fs::read_to_string(&token_path) {
        let token = token.trim().to_string();
        if !token.is_empty() {
            debug!("resolved token from stored token file");
            return Some(token);
        }
    }

    debug!("no token found");
    None
}

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use url::Url;

    use super::{
        HFClientBuilder, append_path_segments, encode_ref, is_relative_location, split_id,
    };

    #[test]
    fn append_path_segments_encodes_special_chars() {
        let mut url = Url::parse("https://example.com/base").unwrap();
        append_path_segments(&mut url, "with space/and#hash?and+plus").unwrap();
        assert_eq!(
            url.as_str(),
            "https://example.com/base/with%20space/and%23hash%3Fand+plus"
        );
    }

    #[test]
    fn append_path_segments_handles_leading_trailing_and_repeated_slashes() {
        let mut url = Url::parse("https://example.com/base").unwrap();
        append_path_segments(&mut url, "/a//b/c/").unwrap();
        assert_eq!(url.as_str(), "https://example.com/base/a/b/c");
    }

    #[test]
    fn append_path_segments_encodes_percent_literals() {
        let mut url = Url::parse("https://example.com/base").unwrap();
        append_path_segments(&mut url, "a%2Fb/c").unwrap();
        assert_eq!(url.as_str(), "https://example.com/base/a%252Fb/c");
    }

    #[test]
    fn append_path_segments_handles_unicode() {
        let mut url = Url::parse("https://example.com/base").unwrap();
        append_path_segments(&mut url, "ümlaut/файл.txt").unwrap();
        assert_eq!(
            url.as_str(),
            "https://example.com/base/%C3%BCmlaut/%D1%84%D0%B0%D0%B9%D0%BB.txt"
        );
    }

    #[test]
    fn append_path_segments_empty_path_is_noop() {
        let mut url = Url::parse("https://example.com/base").unwrap();
        append_path_segments(&mut url, "").unwrap();
        assert_eq!(url.as_str(), "https://example.com/base");
    }

    #[test]
    fn encode_ref_encodes_slash_and_reserved_chars() {
        assert_eq!(encode_ref("feature/x"), "feature%2Fx");
        assert_eq!(encode_ref("fix#123"), "fix%23123");
        assert_eq!(encode_ref("wip?draft"), "wip%3Fdraft");
        assert_eq!(encode_ref("v1.0+beta"), "v1.0%2Bbeta");
        assert_eq!(encode_ref("with space"), "with%20space");
        assert_eq!(encode_ref("100%done"), "100%25done");
    }

    #[test]
    fn encode_ref_preserves_unreserved_chars() {
        assert_eq!(encode_ref("main"), "main");
        assert_eq!(encode_ref("v1.0-rc.2_final~ok"), "v1.0-rc.2_final~ok");
        assert_eq!(encode_ref("abc123def456"), "abc123def456");
    }

    #[test]
    fn encode_ref_keeps_compare_dots_literal() {
        assert_eq!(encode_ref("main..feature/x"), "main..feature%2Fx");
        assert_eq!(encode_ref("v1.0...v2.0"), "v1.0...v2.0");
    }

    #[test]
    fn encode_ref_encodes_unicode() {
        assert_eq!(encode_ref("ветка"), "%D0%B2%D0%B5%D1%82%D0%BA%D0%B0");
    }

    #[test]
    fn is_relative_location_classification() {
        assert!(is_relative_location(
            "/Snowflake/repo/resolve/main/config.json"
        ));
        assert!(is_relative_location("relative/path"));
        assert!(!is_relative_location(
            "https://cdn-lfs.huggingface.co/repos/x"
        ));
        assert!(!is_relative_location("//cdn-lfs.huggingface.co/repos/x"));
    }

    #[test]
    fn download_url_encodes_revision() {
        let client = HFClientBuilder::new()
            .endpoint("https://huggingface.co")
            .build()
            .unwrap();
        let url = client
            .download_url("", "owner/repo", "feature/x", "file.bin")
            .unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/owner/repo/resolve/feature%2Fx/file.bin"
        );

        let url = client
            .download_url("", "owner/repo", "fix#123", "file.bin")
            .unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/owner/repo/resolve/fix%23123/file.bin"
        );
    }

    #[test]
    fn download_url_encodes_filename_segments() {
        let client = HFClientBuilder::new()
            .endpoint("https://huggingface.co")
            .build()
            .unwrap();
        let url = client
            .download_url("", "owner/repo", "main", "subdir/file name #1.bin")
            .unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/owner/repo/resolve/main/subdir/file%20name%20%231.bin"
        );
    }

    #[test]
    fn download_url_dataset_prefix() {
        let client = HFClientBuilder::new()
            .endpoint("https://huggingface.co")
            .build()
            .unwrap();
        let url = client
            .download_url("datasets/", "owner/ds", "v1.0", "data/train.csv")
            .unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/datasets/owner/ds/resolve/v1.0/data/train.csv"
        );
    }

    #[test]
    fn split_id_splits_owner_and_name() {
        assert_eq!(
            split_id("openai-community/gpt2"),
            ("openai-community", "gpt2")
        );
        assert_eq!(
            split_id("HuggingFaceFW/fineweb"),
            ("HuggingFaceFW", "fineweb")
        );
    }

    #[test]
    fn split_id_ownerless_short_form() {
        assert_eq!(split_id("gpt2"), ("", "gpt2"));
        assert_eq!(split_id(""), ("", ""));
    }

    #[test]
    fn split_id_only_splits_first_slash() {
        assert_eq!(split_id("owner/sub/name"), ("owner", "sub/name"));
    }

    #[test]
    fn split_id_empty_owner_or_name() {
        assert_eq!(split_id("/name"), ("", "name"));
        assert_eq!(split_id("owner/"), ("owner", ""));
    }

    #[test]
    fn test_builder_cache_dir_explicit() {
        let client = HFClientBuilder::new()
            .cache_dir("/tmp/my-cache")
            .build()
            .unwrap();
        assert_eq!(client.cache_dir(), std::path::Path::new("/tmp/my-cache"));
    }

    // `#[serial]` because the precedence tests below mutate `HF_HOME`, and the default
    // cache dir is derived from it.
    #[test]
    #[serial]
    fn test_builder_cache_dir_default() {
        let client = HFClientBuilder::new().build().unwrap();
        let path_str = client.cache_dir().to_string_lossy();
        assert!(path_str.contains("huggingface") && path_str.ends_with("hub"));
    }

    #[test]
    fn test_xet_session_lazy_creation() {
        let client = HFClientBuilder::new().build().unwrap();
        assert!(client.inner.xet_state.lock().unwrap().session.is_none());
        let (_s1, _gen) = client.xet_session().unwrap();
        assert!(client.inner.xet_state.lock().unwrap().session.is_some());
    }

    #[test]
    fn test_xet_session_shared_across_clones() {
        let client = HFClientBuilder::new().build().unwrap();
        let clone = client.clone();
        let (_s1, _gen) = client.xet_session().unwrap();
        assert!(clone.inner.xet_state.lock().unwrap().session.is_some());
    }

    #[test]
    fn test_xet_session_recovers_after_abort() {
        let client = HFClientBuilder::new().build().unwrap();

        let (session, generation) = client.xet_session().unwrap();
        session.abort().unwrap();

        match session.new_file_download_group() {
            Ok(_) => panic!("expected error after abort"),
            Err(e) => client.replace_xet_session(generation, &e),
        }

        let (recovered, _) = client.xet_session().unwrap();
        assert!(recovered.new_file_download_group().is_ok());
    }

    #[test]
    fn test_xet_session_recovers_after_sigint_abort() {
        let client = HFClientBuilder::new().build().unwrap();

        let (session, generation) = client.xet_session().unwrap();
        session.sigint_abort().unwrap();

        client.replace_xet_session(generation, &xet::error::XetError::KeyboardInterrupt);

        let (recovered, _) = client.xet_session().unwrap();
        assert!(recovered.new_file_download_group().is_ok());
    }

    /// Simulates the call-site retry pattern used in xet.rs:
    /// 1. Get session + generation, factory call fails
    /// 2. Call replace_xet_session(generation) to drop the bad session
    /// 3. Get a fresh session, factory call succeeds
    #[test]
    fn test_replace_and_retry_after_abort() {
        let client = HFClientBuilder::new().build().unwrap();

        let (session, generation) = client.xet_session().unwrap();
        assert!(session.new_file_download_group().is_ok());

        session.abort().unwrap();

        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                client.replace_xet_session(generation, &e);
                client
                    .xet_session()
                    .unwrap()
                    .0
                    .new_file_download_group()
                    .expect("fresh session factory call should succeed")
            }
        };
        drop(group);
    }

    /// Verifies that replace_xet_session with a stale generation is a no-op.
    #[test]
    fn test_replace_with_stale_generation_is_noop() {
        let client = HFClientBuilder::new().build().unwrap();

        let (session, gen1) = client.xet_session().unwrap();
        session.abort().unwrap();

        // First replace succeeds
        client.replace_xet_session(gen1, &xet::error::XetError::KeyboardInterrupt);

        // Get the fresh session with a new generation
        let (_fresh, gen2) = client.xet_session().unwrap();
        assert_ne!(gen1, gen2);

        // Attempting to replace with the old generation is a no-op
        client.replace_xet_session(gen1, &xet::error::XetError::KeyboardInterrupt);

        // The fresh session is still cached
        let (still_fresh, gen3) = client.xet_session().unwrap();
        assert_eq!(gen2, gen3);
        assert!(still_fresh.new_file_download_group().is_ok());
    }

    #[test]
    fn test_xet_session_reuse_without_replacement() {
        let client = HFClientBuilder::new().build().unwrap();

        let (s1, g1) = client.xet_session().unwrap();
        let (s2, g2) = client.xet_session().unwrap();

        assert_eq!(g1, g2);
        assert!(s1.new_file_download_group().is_ok());
        assert!(s2.new_file_download_group().is_ok());
    }

    /// Token precedence is exercised in child processes with a clean environment.
    /// No test changes the parent process environment or reads the user's token file.
    mod token_precedence {
        use std::ffi::OsStr;
        use std::path::Path;
        use std::process::Command;

        use super::HFClientBuilder;
        use crate::constants::{
            HF_HOME, HF_HUB_DISABLE_IMPLICIT_TOKEN, HF_TOKEN, HF_TOKEN_PATH, TOKEN_FILENAME,
        };

        const CHILD: &str = "SKIPPY_HF_TOKEN_TEST_CHILD";
        const EXPLICIT: &str = "SKIPPY_HF_TOKEN_TEST_EXPLICIT";
        const EXPECTED: &str = "SKIPPY_HF_TOKEN_TEST_EXPECTED";

        fn assert_in_child(
            home: &Path,
            vars: &[(&str, &OsStr)],
            explicit: Option<&str>,
            expected: Option<&str>,
        ) {
            let mut command = Command::new(std::env::current_exe().unwrap());
            command.args([
                "--exact",
                "client::tests::token_precedence::token_resolution_child",
                "--nocapture",
            ]);
            for key in [
                HF_HOME,
                HF_TOKEN,
                HF_TOKEN_PATH,
                HF_HUB_DISABLE_IMPLICIT_TOKEN,
                EXPLICIT,
                EXPECTED,
            ] {
                command.env_remove(key);
            }
            command.env(CHILD, "1").env(HF_HOME, home);
            for (key, value) in vars {
                command.env(key, value);
            }
            if let Some(token) = explicit {
                command.env(EXPLICIT, token);
            }
            if let Some(token) = expected {
                command.env(EXPECTED, token);
            }
            let output = command.output().expect("spawn token precedence child");
            assert!(
                output.status.success(),
                "token precedence child failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("1 passed"),
                "child test filter must execute its assertion"
            );
        }

        #[test]
        fn token_resolution_child() {
            if std::env::var_os(CHILD).is_none() {
                return;
            }
            let mut builder = HFClientBuilder::new();
            if let Ok(token) = std::env::var(EXPLICIT) {
                builder = builder.token(token);
            }
            let client = builder.build().unwrap();
            assert_eq!(client.inner.token, std::env::var(EXPECTED).ok());
        }

        #[test]
        fn test_explicit_token_overrides_env() {
            let home = tempfile::tempdir().unwrap();
            assert_in_child(
                home.path(),
                &[(HF_TOKEN, OsStr::new("env-token"))],
                Some("explicit-token"),
                Some("explicit-token"),
            );
        }

        #[test]
        fn test_env_token_used_when_no_explicit() {
            let home = tempfile::tempdir().unwrap();
            assert_in_child(
                home.path(),
                &[(HF_TOKEN, OsStr::new("env-token"))],
                None,
                Some("env-token"),
            );
        }

        #[test]
        fn test_env_token_overrides_token_path_file() {
            let home = tempfile::tempdir().unwrap();
            let path = home.path().join("tok");
            std::fs::write(&path, "file-token").unwrap();
            assert_in_child(
                home.path(),
                &[
                    (HF_TOKEN, OsStr::new("env-token")),
                    (HF_TOKEN_PATH, path.as_os_str()),
                ],
                None,
                Some("env-token"),
            );
        }

        #[test]
        fn test_token_path_file_used_when_no_env() {
            let home = tempfile::tempdir().unwrap();
            let path = home.path().join("tok");
            std::fs::write(&path, "file-token\n").unwrap();
            assert_in_child(
                home.path(),
                &[(HF_TOKEN_PATH, path.as_os_str())],
                None,
                Some("file-token"),
            );
        }

        #[test]
        fn test_token_from_hf_home_file() {
            let home = tempfile::tempdir().unwrap();
            std::fs::write(home.path().join(TOKEN_FILENAME), "home-token").unwrap();
            assert_in_child(home.path(), &[], None, Some("home-token"));
        }

        #[test]
        fn test_disable_implicit_token_returns_none() {
            let home = tempfile::tempdir().unwrap();
            assert_in_child(
                home.path(),
                &[
                    (HF_TOKEN, OsStr::new("env-token")),
                    (HF_HUB_DISABLE_IMPLICIT_TOKEN, OsStr::new("1")),
                ],
                None,
                None,
            );
        }

        #[test]
        fn test_disable_implicit_token_does_not_block_explicit() {
            let home = tempfile::tempdir().unwrap();
            assert_in_child(
                home.path(),
                &[(HF_HUB_DISABLE_IMPLICIT_TOKEN, OsStr::new("1"))],
                Some("explicit-token"),
                Some("explicit-token"),
            );
        }

        #[test]
        fn test_no_token_anywhere_is_none() {
            let home = tempfile::tempdir().unwrap();
            assert_in_child(home.path(), &[], None, None);
        }
    }
}
