//! Literal immutable projector acquisition, sharing existing trusted repository transport.
use super::SourceClient;
use anyhow::{Result, bail};
use serde::Serialize;
use std::time::Instant;
#[derive(Serialize)]
pub struct ProjectorIdentity {
    pub repo: String,
    pub revision: String,
    pub path: String,
    pub byte_size: u64,
    pub expected_sha256: Option<String>,
}
impl SourceClient {
    pub async fn projector_until(
        &self,
        repo: &str,
        pin: &str,
        path: &str,
        until: Instant,
    ) -> Result<ProjectorIdentity> {
        super::repo(repo)?;
        super::revision(pin)?;
        selection(path)?;
        tokio::time::timeout_at(until.into(), self.projector_info(repo, pin, path, until))
            .await
            .map_err(|_| anyhow::anyhow!("projector metadata deadline"))?
    }
    async fn projector_info(
        &self,
        repo: &str,
        pin: &str,
        path: &str,
        until: Instant,
    ) -> Result<ProjectorIdentity> {
        let (owner, name) = repo
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("projector repo refused"))?;
        let mut url = self.url(&["api", "models", owner, name, "revision", pin])?;
        url.query_pairs_mut().append_pair("blobs", "true");
        let response = self.request(url, until).await?;
        if response.status() != reqwest::StatusCode::OK {
            bail!("projector metadata HTTP refusal");
        }
        let value: serde_json::Value =
            serde_json::from_slice(&super::source::body(response, 2 * 1024 * 1024, until).await?)
                .map_err(|_| anyhow::anyhow!("projector metadata JSON refused"))?;
        if value["sha"] != pin {
            bail!("projector immutable source mismatch");
        }
        let siblings = value["siblings"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("projector sibling roster absent"))?;
        let selected: Vec<_> = siblings.iter().filter(|r| r["rfilename"] == path).collect();
        if selected.len() != 1 {
            bail!("projector literal sibling missing or duplicate");
        }
        let row = selected[0];
        let size = row["size"]
            .as_u64()
            .filter(|n| *n > 0 && *n <= 1024_u64.pow(4))
            .ok_or_else(|| anyhow::anyhow!("projector admitted size absent"))?;
        let expected = match row.get("lfs").filter(|v| !v.is_null()) {
            Some(lfs) => {
                let sha = lfs["sha256"]
                    .as_str()
                    .filter(|s| {
                        s.len() == 64
                            && s.bytes()
                                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    })
                    .ok_or_else(|| anyhow::anyhow!("projector LFS pin malformed"))?;
                if lfs["size"].as_u64() != Some(size) {
                    bail!("projector LFS size mismatch");
                }
                Some(sha.to_owned())
            }
            None => None,
        };
        super::source::guard(until)?;
        Ok(ProjectorIdentity {
            repo: repo.into(),
            revision: pin.into(),
            path: path.into(),
            byte_size: size,
            expected_sha256: expected,
        })
    }
}

pub(super) fn selection(path: &str) -> Result<()> {
    if path.is_empty()
        || path.len() > 4096
        || path.chars().any(char::is_control)
        || path
            .split('/')
            .any(|p| p.is_empty() || matches!(p, "." | ".."))
        || path.contains('\\')
        || !path.to_ascii_lowercase().ends_with(".gguf")
    {
        bail!("literal projector selection refused");
    }
    Ok(())
}
