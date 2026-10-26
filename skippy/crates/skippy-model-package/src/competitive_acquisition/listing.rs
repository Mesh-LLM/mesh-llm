//! Competitive tokenizer inventory only; bytes use the existing native HF stream owner.
use super::contract::{check, path, pin};
use anyhow::{Result, bail};
use futures::StreamExt as _;
use hf_hub::repository::RepoTreeEntry;
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};
pub(crate) struct Listing {
    #[cfg(test)]
    endpoint: Option<String>,
    client: reqwest::Client,
    token: String,
}
impl Listing {
    pub(crate) fn new(token: String) -> Result<Self> {
        skippy_model_hf::configure_hf_tls_provider();
        Ok(Self {
            #[cfg(test)]
            endpoint: None,
            client: reqwest::Client::builder()
                .https_only(true)
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .map_err(|_| anyhow::anyhow!("listing client refused"))?,
            token,
        })
    }
    pub(crate) async fn roster(
        &self,
        repo: &str,
        revision: &str,
        deadline: Instant,
    ) -> Result<BTreeMap<String, u64>> {
        let (owner, name) = repo
            .split_once('/')
            .filter(|(_, name)| !name.contains('/'))
            .ok_or_else(|| anyhow::anyhow!("repo grammar"))?;
        if !path(repo) || !pin(revision, 40) {
            bail!("immutable listing source refused");
        }
        #[cfg(test)]
        let base = self.endpoint.as_deref().unwrap_or("https://huggingface.co");
        #[cfg(not(test))]
        let base = "https://huggingface.co";
        let mut first = reqwest::Url::parse(base)?;
        first
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("URL segments"))?
            .extend(["api", "models", owner, name, "tree", revision]);
        first.set_query(Some("recursive=true&expand=true&limit=1000"));
        self.pages(first, deadline).await
    }
    async fn pages(&self, first: reqwest::Url, deadline: Instant) -> Result<BTreeMap<String, u64>> {
        let mut pending = Some(first.clone());
        let mut visited = BTreeSet::new();
        let mut rows = BTreeMap::new();
        let mut entries = 0;
        let mut pages = 0;
        let mut received = 0;
        while let Some(url) = pending.take() {
            check(deadline)?;
            pages += 1;
            if pages > 32 || !visited.insert(url.to_string()) {
                bail!("snapshot pagination bound/cycle");
            }
            let mut request = self.client.get(url);
            if !self.token.is_empty() {
                request = request.bearer_auth(&self.token);
            }
            let response = request
                .send()
                .await
                .map_err(|_| anyhow::anyhow!("snapshot inventory request failed"))?;
            if response.status() != reqwest::StatusCode::OK {
                bail!("snapshot inventory HTTP refusal");
            }
            let links: Vec<_> = response
                .headers()
                .get_all(reqwest::header::LINK)
                .iter()
                .map(|s| s.to_str().map(str::to_owned))
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| anyhow::anyhow!("snapshot link encoding refused"))?;
            pending = next(&first, &links)?;
            let mut stream = response.bytes_stream();
            let mut bytes = Vec::new();
            while let Some(chunk) = stream.next().await {
                check(deadline)?;
                let chunk = chunk.map_err(|_| anyhow::anyhow!("snapshot inventory body failed"))?;
                received += chunk.len();
                if bytes.len() + chunk.len() > 2 * 1024 * 1024 || received > 16 * 1024 * 1024 {
                    bail!("snapshot metadata byte bound");
                }
                bytes.extend_from_slice(&chunk);
            }
            let parsed: Vec<RepoTreeEntry> = serde_json::from_value(super::json::unique(&bytes)?)?;
            for entry in parsed {
                entries += 1;
                if entries > 16384 {
                    bail!("snapshot entry bound");
                }
                match entry {
                    RepoTreeEntry::File {
                        path: name, size, ..
                    } => {
                        if !path(&name) || rows.insert(name, size).is_some() {
                            bail!("snapshot path/duplicate refused");
                        }
                    }
                    RepoTreeEntry::Directory { path: name, .. } => {
                        if !path(&name) {
                            bail!("snapshot directory grammar");
                        }
                    }
                }
            }
        }
        check(deadline)?;
        if rows.is_empty() {
            bail!("empty immutable snapshot roster");
        }
        Ok(rows)
    }
}
fn next(first: &reqwest::Url, headers: &[String]) -> Result<Option<reqwest::Url>> {
    let mut result = None;
    for header in headers {
        for part in header.split(',') {
            let Some((raw, tail)) = part.trim().split_once('>') else {
                bail!("snapshot pagination grammar");
            };
            let raw = raw
                .strip_prefix('<')
                .ok_or_else(|| anyhow::anyhow!("snapshot pagination grammar"))?;
            if tail.split(';').map(str::trim).any(|s| s == "rel=\"next\"") {
                let url = first.join(raw)?;
                if result.is_some()
                    || url.origin() != first.origin()
                    || url.path() != first.path()
                    || url.fragment().is_some()
                    || !url.username().is_empty()
                    || url.password().is_some()
                {
                    bail!("snapshot pagination origin/path ambiguity");
                }
                result = Some(url);
            }
        }
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immutable_inventory_next_links_keep_origin_revision_and_unique_bounded_chain() {
        let first=reqwest::Url::parse("https://huggingface.co/api/models/a/b/tree/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa?recursive=true").unwrap();
        let good = format!("<{}&cursor=1>; rel=\"next\"", first);
        assert!(next(&first, std::slice::from_ref(&good)).unwrap().is_some());
        for bad in [
            "<https://other.invalid/api/models/a/b/tree/a>; rel=\"next\"".to_string(),
            "</api/models/a/b/tree/main>; rel=\"next\"".to_string(),
            format!("{good},{good}"),
        ] {
            assert!(next(&first, &[bad]).is_err());
        }
    }
}

#[cfg(all(test, unix))]
mod body_tests {
    use super::*;
    use crate::competitive_acquisition::download_tests::Peer;
    #[test]
    fn native_typed_inventory_body_cap_and_duplicate_paths_refuse_without_artifact_get() {
        let revision = "a".repeat(40);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        for body in [serde_json::to_vec(&serde_json::json!([{"type":"file","path":"tokenizer.json","oid":"a","size":4},{"type":"file","path":"tokenizer.json","oid":"b","size":4}])).unwrap(),vec![b'x';2*1024*1024+1]] {let n=body.len();let peer=Peer::new(vec![(revision.clone(),body,n)]);let listing=Listing{endpoint:Some(peer.endpoint.clone()),client:reqwest::Client::builder().no_proxy().timeout(std::time::Duration::from_secs(2)).build().unwrap(),token:String::new()};assert!(rt.block_on(listing.pages(reqwest::Url::parse(&peer.endpoint).unwrap(),Instant::now()+std::time::Duration::from_secs(3))).is_err());drop(peer);}
    }
}

#[cfg(all(test, unix))]
impl Listing {
    pub(crate) fn fixture(endpoint: &str) -> Self {
        Self {
            endpoint: Some(endpoint.into()),
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(std::time::Duration::from_secs(2))
                .build()
                .unwrap(),
            token: String::new(),
        }
    }
}
