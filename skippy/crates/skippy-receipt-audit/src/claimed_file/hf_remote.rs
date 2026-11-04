//! A claimed file on Hugging Face (or `HF_ENDPOINT`), read by range.
//!
//! The resolver names the commit a revision is at and redirects to a
//! signed CDN URL, which the range requests reuse until it expires. The
//! token goes to the resolver's origin only: the redirects of a resolve are
//! read, not followed, and a range request carries the token only to that
//! origin. An `https` resolver is never left for `http`. With a plain
//! `http://` endpoint nothing protects the bytes in transit.

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_RANGE, HeaderMap, LOCATION, RANGE};
use skippy_receipts::sparse_copy::RangeFetch;

use super::hf_file::{HfFile, is_commit};

/// Redirects the resolver may send before it names the file's URL, and the
/// file's server after.
const MAX_REDIRECTS: usize = 5;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// The HTTP clients of every claimed file of the process, so the files of
/// a model and the shards of a split share connections.
#[derive(Clone)]
struct Clients {
    /// For the resolver: redirects are read, not followed.
    resolve: Client,
    /// For the file's bytes.
    fetch: Client,
}

impl Clients {
    fn shared() -> Result<Self> {
        static SHARED: OnceLock<Clients> = OnceLock::new();
        if let Some(clients) = SHARED.get() {
            return Ok(clients.clone());
        }
        let built = Self {
            resolve: Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            fetch: Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .redirect(reqwest::redirect::Policy::custom(follow_redirect))
                .build()?,
        };
        Ok(SHARED.get_or_init(|| built).clone())
    }
}

/// Follows a redirect of the file's server, a few times and never from
/// `https` to `http`. The client drops the token when a redirect leaves the
/// origin.
fn follow_redirect(attempt: reqwest::redirect::Attempt<'_>) -> reqwest::redirect::Action {
    let previous = attempt.previous();
    if previous.len() > MAX_REDIRECTS {
        return attempt.error("the file's server redirected too many times");
    }
    let was_https = previous.last().is_some_and(|url| url.scheme() == "https");
    if was_https && attempt.url().scheme() != "https" {
        return attempt.error("the file's server redirected from https to http");
    }
    attempt.follow()
}

pub(super) struct HfRemote {
    resolver: reqwest::Url,
    token: Option<String>,
    clients: Clients,
    /// The file's URL, resolved once and again when it expires.
    url: Mutex<Option<reqwest::Url>>,
}

/// What the resolver said of the file.
pub(super) struct Resolved {
    pub(super) url: reqwest::Url,
    pub(super) size: Option<u64>,
    /// The commit the revision is at.
    pub(super) commit: Option<String>,
}

/// The resolver's URL of `file`, each part of the name a path segment of
/// its own: a part can add no segment, query or fragment.
pub(super) fn resolve_url(endpoint: &str, file: &HfFile) -> Result<reqwest::Url> {
    file.check()
        .with_context(|| format!("{file} names no file on Hugging Face"))?;
    let mut url = reqwest::Url::parse(endpoint)
        .with_context(|| format!("read the Hugging Face endpoint {endpoint}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("the Hugging Face endpoint {endpoint} is not http or https");
    }
    let (owner, name) = file
        .repo
        .split_once('/')
        .context("the repo is not owner/name")?;
    url.path_segments_mut()
        .map_err(|()| anyhow::anyhow!("the Hugging Face endpoint {endpoint} holds no path"))?
        .pop_if_empty()
        .extend([owner, name, "resolve", file.revision.as_str()])
        .extend(file.file.split('/'));
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

impl HfRemote {
    pub(super) fn new(file: &HfFile, endpoint: &str, token: Option<String>) -> Result<Self> {
        Ok(Self {
            resolver: resolve_url(endpoint, file)?,
            token,
            clients: Clients::shared()?,
            url: Mutex::new(None),
        })
    }

    /// Follows the resolver's redirects on its own origin and stops at the
    /// first URL elsewhere, the CDN's, or at the file.
    pub(super) fn resolve(&self) -> Result<Resolved> {
        let mut url = self.resolver.clone();
        let mut size = None;
        let mut commit = None;
        for _ in 0..=MAX_REDIRECTS {
            let mut request = self.clients.resolve.head(url.clone());
            if let Some(token) = &self.token {
                request = request.header(AUTHORIZATION, format!("Bearer {token}"));
            }
            let response = request
                .send()
                .with_context(|| format!("reach Hugging Face at {url}"))?;
            let headers = response.headers();
            size = size.or_else(|| header_u64(headers, "x-linked-size"));
            if commit.is_none() {
                commit = commit_header(headers)?;
            }
            let status = response.status();
            if status.is_success() {
                size = size.or_else(|| response.content_length().filter(|&n| n > 0));
                return Ok(Resolved { url, size, commit });
            }
            if !status.is_redirection() {
                bail!("Hugging Face answered {status} for {url}");
            }
            let location = header_str(headers, LOCATION.as_str())
                .with_context(|| format!("Hugging Face redirected {url} nowhere"))?;
            let next = url.join(&location)?;
            if !matches!(next.scheme(), "http" | "https")
                || (url.scheme() == "https" && next.scheme() != "https")
            {
                bail!("Hugging Face redirected {url} to {next}, which is not as safe");
            }
            if !self.is_resolver(&next) {
                return Ok(Resolved {
                    url: next,
                    size,
                    commit,
                });
            }
            url = next;
        }
        bail!("Hugging Face redirected {} too many times", self.resolver)
    }

    /// Whether `url` is at the resolver's origin (scheme, host and port),
    /// which may see the token.
    fn is_resolver(&self, url: &reqwest::Url) -> bool {
        url.origin() == self.resolver.origin()
    }

    /// The file's size from a one-byte range request.
    pub(super) fn probe_size(&self, url: &reqwest::Url) -> Result<u64> {
        let response = self.range_request(url, 0, 1)?;
        let range = header_str(response.headers(), CONTENT_RANGE.as_str())
            .context("the file's server sent no Content-Range")?;
        range
            .rsplit_once('/')
            .and_then(|(_, total)| total.parse().ok())
            .with_context(|| format!("unreadable Content-Range {range}"))
    }

    /// Keeps `url` as the file's, for the range requests.
    pub(super) fn set_url(&self, url: reqwest::Url) -> Result<()> {
        *self
            .url
            .lock()
            .map_err(|_| anyhow::anyhow!("url lock poisoned"))? = Some(url);
        Ok(())
    }

    fn range_request(
        &self,
        url: &reqwest::Url,
        offset: u64,
        len: u64,
    ) -> Result<reqwest::blocking::Response> {
        let last = len
            .checked_sub(1)
            .and_then(|n| offset.checked_add(n))
            .context("a range request of no bytes, or past any offset")?;
        let mut request = self
            .clients
            .fetch
            .get(url.clone())
            .header(RANGE, format!("bytes={offset}-{last}"));
        if self.is_resolver(url)
            && let Some(token) = &self.token
        {
            request = request.header(AUTHORIZATION, format!("Bearer {token}"));
        }
        let response = request.send()?;
        match response.status() {
            StatusCode::PARTIAL_CONTENT => Ok(response),
            status => bail!("a range request answered {status}, not 206"),
        }
    }

    fn current_url(&self) -> Result<reqwest::Url> {
        let mut url = self
            .url
            .lock()
            .map_err(|_| anyhow::anyhow!("url lock poisoned"))?;
        match &*url {
            Some(url) => Ok(url.clone()),
            None => {
                let resolved = self.resolve()?.url;
                *url = Some(resolved.clone());
                Ok(resolved)
            }
        }
    }

    fn fetch_from(&self, url: &reqwest::Url, offset: u64, len: u64) -> Result<Vec<u8>> {
        let bytes = self.range_request(url, offset, len)?.bytes()?;
        if bytes.len() as u64 != len {
            bail!("a range request sent {} bytes, not {len}", bytes.len());
        }
        Ok(bytes.to_vec())
    }
}

impl RangeFetch for HfRemote {
    fn fetch(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let url = self.current_url()?;
        match self.fetch_from(&url, offset, len) {
            Ok(bytes) => Ok(bytes),
            Err(first) => {
                // a signed URL expires after about an hour: resolve it again
                let fresh = self.resolve().context(first)?.url;
                if let Ok(mut url) = self.url.lock() {
                    *url = Some(fresh.clone());
                }
                self.fetch_from(&fresh, offset, len)
            }
        }
    }
}

/// The commit the resolver says the revision is at. It becomes a directory
/// of the cache and the pin of the copy, so anything but a commit id is an
/// error.
fn commit_header(headers: &HeaderMap) -> Result<Option<String>> {
    match header_str(headers, "x-repo-commit") {
        Some(commit) if is_commit(&commit) => Ok(Some(commit)),
        Some(other) => bail!("Hugging Face named the commit {other:?}, which is not a commit"),
        None => Ok(None),
    }
}

fn header_str(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim_matches('"').to_owned())
        .filter(|v| !v.is_empty())
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    header_str(headers, name)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(revision: &str, file: &str) -> HfFile {
        HfFile {
            repo: "owner/model".into(),
            revision: revision.into(),
            file: file.into(),
        }
    }

    #[test]
    fn the_resolve_url_holds_each_part_as_a_segment() {
        let url = resolve_url("https://huggingface.co", &file("main", "sub/m.gguf")).unwrap();
        assert_eq!(
            url.as_str(),
            "https://huggingface.co/owner/model/resolve/main/sub/m.gguf"
        );
        // an endpoint with a path of its own, with or without a slash
        for endpoint in ["http://mirror.test:8080/hf", "http://mirror.test:8080/hf/"] {
            let url = resolve_url(endpoint, &file("main", "m.gguf")).unwrap();
            assert_eq!(
                url.as_str(),
                "http://mirror.test:8080/hf/owner/model/resolve/main/m.gguf"
            );
        }
        assert!(resolve_url("ftp://mirror.test", &file("main", "m.gguf")).is_err());
        assert!(resolve_url("not a url", &file("main", "m.gguf")).is_err());
    }

    #[test]
    fn a_name_that_would_change_the_url_makes_none() {
        for (revision, path) in [
            ("main", "../../other/model/resolve/main/m.gguf"),
            ("main", "%2e%2e/m.gguf"),
            ("main", "m.gguf?download=1"),
            ("main", "m.gguf#x"),
            ("main", "/m.gguf"),
            ("main", "a\\m.gguf"),
            ("..", "m.gguf"),
            ("main?x", "m.gguf"),
            ("main#x", "m.gguf"),
            ("ma%69n", "m.gguf"),
        ] {
            let made = resolve_url("https://huggingface.co", &file(revision, path));
            assert!(made.is_err(), "{revision} {path}: {made:?}");
        }
        let mut other_host = file("main", "m.gguf");
        other_host.repo = "owner@evil.test/model".into();
        assert!(resolve_url("https://huggingface.co", &other_host).is_err());
    }

    #[test]
    fn only_a_commit_id_is_taken_from_the_resolver() {
        let with = |value: &str| {
            let mut headers = HeaderMap::new();
            headers.insert("x-repo-commit", value.parse().unwrap());
            commit_header(&headers)
        };
        let commit = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(with(commit).unwrap().as_deref(), Some(commit));
        for value in ["main", "../../../etc", "0123", "..", &commit.to_uppercase()] {
            assert!(with(value).is_err(), "{value}");
        }
        assert_eq!(commit_header(&HeaderMap::new()).unwrap(), None);
    }
}
