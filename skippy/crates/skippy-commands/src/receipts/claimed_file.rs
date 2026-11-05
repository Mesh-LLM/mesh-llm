//! The claimed GGUF an audit checks against: a local file, or a Hugging
//! Face file named `repo@revision/file` that the audit reads by range into
//! a sparse local copy, so the client never downloads the model.
//!
//! The copy is pinned by the commit Hugging Face resolves the revision to.
//! The resolver gives one signed CDN URL per file, which the range requests
//! reuse until it expires; the token goes to the resolver only.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_RANGE, HeaderMap, LOCATION, RANGE};
use skippy_receipts::shards::shard_files;
use skippy_receipts::sparse_copy::{RangeFetch, RemoteIdentity, SparseCopy};
use skippy_receipts::weights::{ByteSource, LocalFile, WeightFile};

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
/// Redirects the resolver may send before it names the file's URL.
const MAX_REDIRECTS: usize = 5;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// The claimed GGUF, opened.
pub struct ClaimedFile {
    /// What the native loader reads: the file, or its sparse copy.
    pub path: PathBuf,
    pub weights: WeightFile,
    /// What names this file and no other: `repo@commit/file`, or a local
    /// file's path, size and modification time.
    pub identity: String,
}

/// A file of a Hugging Face model repo at a revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfFile {
    pub repo: String,
    pub revision: String,
    pub file: String,
}

impl HfFile {
    /// Parses `owner/name@revision/path/to/file.gguf`.
    pub fn parse(text: &str) -> Option<Self> {
        let (repo, rest) = text.split_once('@')?;
        let (revision, file) = rest.split_once('/')?;
        let (owner, name) = repo.split_once('/')?;
        let part = |s: &str| !s.is_empty() && !s.contains(['/', '@']);
        (part(owner) && part(name) && part(revision) && !file.is_empty() && !file.contains(".."))
            .then(|| Self {
                repo: repo.to_owned(),
                revision: revision.to_owned(),
                file: file.to_owned(),
            })
    }
}

/// Where the sparse copies live, unless the caller names a place.
pub fn default_cache_root() -> PathBuf {
    skippy_model_hf::application_cache_dir()
        .join("receipts")
        .join("claimed")
}

/// Opens `claimed`: a path when it names a file here, else a Hugging Face
/// file, whose sparse copy lives under `cache_root`.
pub fn open(claimed: &Path, cache_root: &Path) -> Result<ClaimedFile> {
    if claimed.exists() {
        let weights = open_local(claimed)?;
        return Ok(ClaimedFile {
            path: claimed.to_owned(),
            weights,
            identity: local_identity(claimed)?,
        });
    }
    let text = claimed.to_string_lossy();
    let Some(file) = HfFile::parse(&text) else {
        bail!(
            "the claimed model {text} is neither a file here nor repo@revision/file on Hugging Face"
        );
    };
    open_remote(&file, cache_root)
}

/// Opens `file` on Hugging Face (or `HF_ENDPOINT`), never a file here: for
/// a claimed model a node named, which must not choose what this machine
/// reads.
pub fn open_remote(file: &HfFile, cache_root: &Path) -> Result<ClaimedFile> {
    open_hf(
        file,
        &endpoint(),
        skippy_model_hf::hf_token_override(),
        cache_root,
    )
}

/// The Hugging Face endpoint: `HF_ENDPOINT`, when set and not empty.
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim().trim_end_matches('/').to_owned())
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned())
}

/// Opens the sparse copy of `file` and reads its header; for the first
/// shard of a split GGUF, the copies of every shard, at the commit the
/// first resolves to.
pub fn open_hf(
    file: &HfFile,
    endpoint: &str,
    token: Option<String>,
    cache_root: &Path,
) -> Result<ClaimedFile> {
    let (first, commit) = open_hf_copy(file, endpoint, token.clone(), cache_root)?;
    let path = first.path().to_owned();
    let source = format!("{}@{commit}/{}", file.repo, file.file);
    let weights = match split_shards(&file.file)? {
        None => WeightFile::open(Box::new(first)),
        Some(names) => {
            let mut shards: Vec<Box<dyn ByteSource>> = vec![Box::new(first)];
            for name in &names[1..] {
                let shard = HfFile {
                    repo: file.repo.clone(),
                    revision: commit.clone(),
                    file: name.clone(),
                };
                shards.push(Box::new(
                    open_hf_copy(&shard, endpoint, token.clone(), cache_root)?.0,
                ));
            }
            WeightFile::open_shards(shards)
        }
    }
    .with_context(|| format!("read the GGUF header of {source}"))?;
    Ok(ClaimedFile {
        path,
        weights,
        identity: source,
    })
}

/// Opens the sparse copy of one file, and the commit it is pinned to.
fn open_hf_copy(
    file: &HfFile,
    endpoint: &str,
    token: Option<String>,
    cache_root: &Path,
) -> Result<(SparseCopy, String)> {
    let remote = HfRemote::new(file, endpoint, token)?;
    let resolved = remote.resolve()?;
    let commit = resolved
        .commit
        .clone()
        .unwrap_or_else(|| file.revision.clone());
    if is_commit(&file.revision) && commit != file.revision {
        bail!(
            "Hugging Face resolved {}@{} to commit {commit}",
            file.repo,
            file.revision
        );
    }
    let size = match resolved.size {
        Some(size) => size,
        None => remote.probe_size(&resolved.url)?,
    };
    *remote
        .url
        .lock()
        .map_err(|_| anyhow::anyhow!("url lock poisoned"))? = Some(resolved.url);
    let path = cache_root
        .join(file.repo.replace('/', "--"))
        .join(&commit)
        .join(&file.file);
    let source = format!("{}@{commit}/{}", file.repo, file.file);
    let identity = RemoteIdentity {
        source: source.clone(),
        size,
    };
    let copy = SparseCopy::open(&path, identity, Box::new(remote))?;
    Ok((copy, commit))
}

/// Reads a local GGUF, with every shard of a split one.
fn open_local(path: &Path) -> Result<WeightFile> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let Some(names) = split_shards(name)? else {
        return WeightFile::open(Box::new(LocalFile::open(path)?));
    };
    let dir = path.parent().unwrap_or(Path::new("."));
    let shards = names
        .iter()
        .map(|name| Ok(Box::new(LocalFile::open(&dir.join(name))?) as Box<dyn ByteSource>))
        .collect::<Result<Vec<_>>>()?;
    WeightFile::open_shards(shards)
}

/// The shard files of the split GGUF `file` names, or `None` for a whole
/// GGUF. A claim names a split by its first shard.
fn split_shards(file: &str) -> Result<Option<Vec<String>>> {
    let name = file.rsplit('/').next().unwrap_or(file);
    match skippy_model_ref::split_gguf_shard_info(name) {
        None => Ok(None),
        Some(shard) if shard.total == "00001" => Ok(None),
        Some(shard) if shard.part != "00001" => {
            bail!(
                "{file} is shard {} of a split GGUF: claim its first shard",
                shard.part
            )
        }
        Some(_) => Ok(shard_files(file)),
    }
}

fn local_identity(path: &Path) -> Result<String> {
    let canonical = path.canonicalize()?;
    let meta = std::fs::metadata(&canonical)?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    Ok(format!("{}:{}:{modified}", canonical.display(), meta.len()))
}

fn is_commit(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

struct HfRemote {
    resolver: reqwest::Url,
    token: Option<String>,
    /// For the resolver: redirects are read, not followed.
    resolve_client: Client,
    client: Client,
    /// The file's URL, resolved once and again when it expires.
    url: Mutex<Option<reqwest::Url>>,
}

/// What the resolver said of the file.
struct Resolved {
    url: reqwest::Url,
    size: Option<u64>,
    commit: Option<String>,
}

impl HfRemote {
    fn new(file: &HfFile, endpoint: &str, token: Option<String>) -> Result<Self> {
        let resolver = reqwest::Url::parse(&format!(
            "{endpoint}/{}/resolve/{}/{}",
            file.repo, file.revision, file.file
        ))
        .context("form the Hugging Face resolve URL")?;
        Ok(Self {
            resolver,
            token,
            resolve_client: Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            client: Client::builder().timeout(REQUEST_TIMEOUT).build()?,
            url: Mutex::new(None),
        })
    }

    /// Follows the resolver's redirects on its own host and stops at the
    /// first URL elsewhere, the CDN's, or at the file.
    fn resolve(&self) -> Result<Resolved> {
        let mut url = self.resolver.clone();
        let mut size = None;
        let mut commit = None;
        for _ in 0..=MAX_REDIRECTS {
            let mut request = self.resolve_client.head(url.clone());
            if let Some(token) = &self.token {
                request = request.header(AUTHORIZATION, format!("Bearer {token}"));
            }
            let response = request
                .send()
                .with_context(|| format!("reach Hugging Face at {url}"))?;
            let headers = response.headers();
            size = size.or_else(|| header_u64(headers, "x-linked-size"));
            commit = commit.or_else(|| header_str(headers, "x-repo-commit"));
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

    /// Whether `url` is on the resolver's host, which may see the token.
    fn is_resolver(&self, url: &reqwest::Url) -> bool {
        url.host_str() == self.resolver.host_str() && url.port() == self.resolver.port()
    }

    /// The file's size from a one-byte range request.
    fn probe_size(&self, url: &reqwest::Url) -> Result<u64> {
        let response = self.range_request(url, 0, 1)?;
        let range = header_str(response.headers(), CONTENT_RANGE.as_str())
            .context("the file's server sent no Content-Range")?;
        range
            .rsplit_once('/')
            .and_then(|(_, total)| total.parse().ok())
            .with_context(|| format!("unreadable Content-Range {range}"))
    }

    fn range_request(
        &self,
        url: &reqwest::Url,
        offset: u64,
        len: u64,
    ) -> Result<reqwest::blocking::Response> {
        let mut request = self
            .client
            .get(url.clone())
            .header(RANGE, format!("bytes={offset}-{}", offset + len - 1));
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
        if url.is_none() {
            *url = Some(self.resolve()?.url);
        }
        Ok(url.clone().expect("resolved"))
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
mod tests;
