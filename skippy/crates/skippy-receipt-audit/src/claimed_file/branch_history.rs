//! Whether a commit a node named is in the history of the branch the
//! request named, as Hugging Face (or `HF_ENDPOINT`) lists it.
//!
//! Hugging Face serves a file at any commit of a repo by its id, the
//! commits of an open pull request included, and anyone can open a pull
//! request on a public repo. So a commit id alone does not say the repo's
//! owner published those bytes: a node could open a pull request that
//! replaces the file, serve that, and name the pull request's commit.
//! When the request named a branch, or no revision (the default branch
//! `main`), the commit must be in that branch's history.
//!
//! The history is the commit list of the branch,
//! `GET /api/models/{repo}/commits/{branch}`, newest first, one page of up
//! to [`PAGE_LIMIT`] commits at a time, following the `next` link of each
//! answer for at most [`MAX_PAGES`] pages. The commit list of a pull
//! request's ref holds the commits of its branch too, so only the branch's
//! own list counts. A commit found is kept for the process
//! ([`ACCEPTED`]), so an audit of a model already checked asks nothing.
//! A commit not in a list that ended is not on the branch
//! ([`NotOnBranch`]). A list that did not end within the pages, or did
//! not answer, cannot say: that is no audit.

use std::collections::HashSet;
use std::fmt;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, LINK};
use serde::Deserialize;
use skippy_receipts::weights::limits::CannotAudit;

use super::hf_file::{HfFile, is_commit};
use super::hf_remote::read_at_most;

/// Commits asked for per page. Hugging Face answers 50 without a limit.
const PAGE_LIMIT: u32 = 1000;
/// Pages read before the check gives up: the newest 4,000 commits of the
/// branch, or 200 when the server keeps its own page of 50.
pub(super) const MAX_PAGES: usize = 4;
/// The largest page read, in bytes. A commit takes about 350 bytes.
const MAX_PAGE_BYTES: u64 = 4 << 20;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Accepted (repo, branch, commit) triples kept for the process; past this
/// the set starts over.
const MAX_ACCEPTED: usize = 4096;

/// The (repo, branch, commit) triples found so far. A commit stays in the
/// history of a branch unless the owner rewrites it.
static ACCEPTED: Mutex<Option<HashSet<(String, String, String)>>> = Mutex::new(None);

/// The commit a node named is not in the history of the branch the request
/// named: what it served is not what was asked for.
#[derive(Debug)]
pub struct NotOnBranch(String);

impl NotOnBranch {
    pub fn new(reason: impl Into<String>) -> Self {
        Self(reason.into())
    }

    /// Whether `error` is, or was caused by, this mismatch.
    pub fn caused(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| cause.is::<Self>())
    }
}

impl fmt::Display for NotOnBranch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NotOnBranch {}

#[derive(Deserialize)]
struct Commit {
    id: String,
}

/// Checks that `file`'s revision, a commit, is in the history of `branch`
/// of its repo, on the Hugging Face at `endpoint`.
pub(super) fn check(
    file: &HfFile,
    branch: &str,
    endpoint: &str,
    token: Option<&str>,
) -> Result<()> {
    if !file.is_pinned() {
        bail!("{file} is not at a commit, so no history holds it");
    }
    let triple = (
        file.repo.to_ascii_lowercase(),
        branch.to_owned(),
        file.revision.clone(),
    );
    if accepted(&triple) {
        return Ok(());
    }
    let first = commits_url(endpoint, &file.repo, branch)?;
    let mut next = Some(first.clone());
    let mut pages = 0;
    while let Some(url) = next {
        if pages == MAX_PAGES {
            return Err(CannotAudit::new(format!(
                "the commit {} is not among the newest {MAX_PAGES} pages of commits of {}@{branch}, and older ones are not read",
                file.revision, file.repo
            ))
            .into());
        }
        pages += 1;
        let (ids, link) = page(&url, &first, token)?;
        if ids.contains(&file.revision) {
            accept(triple);
            return Ok(());
        }
        next = link;
    }
    Err(NotOnBranch(format!(
        "the receipt names {}@{}, a commit that is not in the history of the branch {branch} the request named: Hugging Face serves it, but it may be of a pull request anyone can open",
        file.repo, file.revision
    ))
    .into())
}

fn accepted(triple: &(String, String, String)) -> bool {
    ACCEPTED
        .lock()
        .is_ok_and(|set| set.as_ref().is_some_and(|set| set.contains(triple)))
}

fn accept(triple: (String, String, String)) {
    if let Ok(mut set) = ACCEPTED.lock() {
        let set = set.get_or_insert_with(HashSet::new);
        if set.len() >= MAX_ACCEPTED {
            set.clear();
        }
        set.insert(triple);
    }
}

/// The first page of the commit list of `branch` of `repo`, each part of
/// the name a path segment of its own.
pub(super) fn commits_url(endpoint: &str, repo: &str, branch: &str) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(endpoint)
        .with_context(|| format!("read the Hugging Face endpoint {endpoint}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("the Hugging Face endpoint {endpoint} is not http or https");
    }
    let (owner, name) = repo.split_once('/').context("the repo is not owner/name")?;
    if branch.is_empty() {
        bail!("the branch is empty");
    }
    url.path_segments_mut()
        .map_err(|()| anyhow::anyhow!("the Hugging Face endpoint {endpoint} holds no path"))?
        .pop_if_empty()
        .extend(["api", "models", owner, name, "commits", branch]);
    url.set_query(Some(&format!("limit={PAGE_LIMIT}")));
    url.set_fragment(None);
    Ok(url)
}

fn client() -> Result<Client> {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    if let Some(client) = CLIENT.get() {
        return Ok(client.clone());
    }
    let built = Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    Ok(CLIENT.get_or_init(|| built).clone())
}

/// The commit ids of one page, and the next page's URL. The next page must
/// be at the origin of the first, which may see the token.
fn page(
    url: &reqwest::Url,
    first: &reqwest::Url,
    token: Option<&str>,
) -> Result<(Vec<String>, Option<reqwest::Url>)> {
    let mut request = client()?.get(url.clone());
    if let Some(token) = token {
        request = request.header(AUTHORIZATION, format!("Bearer {token}"));
    }
    let response = request
        .send()
        .with_context(|| format!("list the commits at {url}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(CannotAudit::new(format!(
            "Hugging Face answered {status} for the commit list {url}"
        ))
        .into());
    }
    let link = response
        .headers()
        .get(LINK)
        .and_then(|v| v.to_str().ok())
        .and_then(next_link)
        .map(|next| url.join(&next))
        .transpose()?;
    if let Some(next) = &link
        && next.origin() != first.origin()
    {
        bail!("the commit list {url} links to {next}, at another origin");
    }
    let body = read_at_most(response, MAX_PAGE_BYTES)
        .with_context(|| format!("read the commit list {url}"))?;
    let commits: Vec<Commit> =
        serde_json::from_slice(&body).with_context(|| format!("decode the commit list {url}"))?;
    let ids = commits
        .into_iter()
        .map(|commit| commit.id)
        .filter(|id| is_commit(id))
        .collect();
    Ok((ids, link))
}

/// The target of the `rel="next"` link of a `Link` header.
fn next_link(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let (target, params) = part.split_once(';')?;
        let next = params
            .split(';')
            .any(|p| matches!(p.trim(), "rel=\"next\"" | "rel=next"));
        let target = target.trim().strip_prefix('<')?.strip_suffix('>')?;
        next.then(|| target.to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_commit_list_url_holds_each_part_as_a_segment() {
        let url = commits_url("https://huggingface.co", "o/m", "main").unwrap();
        assert_eq!(
            url.as_str(),
            "https://huggingface.co/api/models/o/m/commits/main?limit=1000"
        );
        // a ref with slashes stays one segment
        let url = commits_url("http://mirror.test/hf/", "o/m", "refs/pr/1").unwrap();
        assert_eq!(
            url.as_str(),
            "http://mirror.test/hf/api/models/o/m/commits/refs%2Fpr%2F1?limit=1000"
        );
        assert!(commits_url("ftp://mirror.test", "o/m", "main").is_err());
        assert!(commits_url("https://huggingface.co", "o/m", "").is_err());
    }

    #[test]
    fn only_the_next_link_is_followed() {
        let header = "<https://h.test/api/models/o/m/commits/main?p=1&limit=50>; rel=\"next\"";
        assert_eq!(
            next_link(header).as_deref(),
            Some("https://h.test/api/models/o/m/commits/main?p=1&limit=50")
        );
        let both = "<https://h.test/a?p=0>; rel=\"prev\", <https://h.test/a?p=2>; rel=\"next\"";
        assert_eq!(next_link(both).as_deref(), Some("https://h.test/a?p=2"));
        assert_eq!(next_link("<https://h.test/a?p=0>; rel=\"prev\""), None);
        assert_eq!(next_link("garbage"), None);
    }
}
