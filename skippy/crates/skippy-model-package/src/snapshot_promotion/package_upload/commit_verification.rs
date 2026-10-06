//! Complete immutable quant roster byte verification/acquisition, using existing scoped transport.
use super::{Plan, Publisher, RepositoryKind, policy};
use crate::snapshot_promotion::{lfs_transfer::RepositoryRead, regular_publication};
use anyhow::{Result, anyhow, bail};
use futures::{Future, FutureExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    io::Write as _,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommitArtifact {
    pub path: String,
    pub sha256: String,
    pub byte_size: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommitRequest {
    pub schema_version: u32,
    pub repo: String,
    pub commit: String,
    pub gguf_prefix: String,
    pub basename: String,
    pub expected_splits: u32,
    pub artifacts: Vec<CommitArtifact>,
}
#[derive(Serialize)]
pub struct CommitReceipt {
    pub repo: String,
    pub commit: String,
    pub artifact_root: PathBuf,
    pub verified: Vec<CommitArtifact>,
    pub completed: bool,
    pub error: Option<String>,
}
impl CommitRequest {
    pub fn validate(&self) -> Result<BTreeSet<String>> {
        if self.schema_version != 1
            || !regular_publication::contract::hex(&self.commit, 40)
            || !(1..=1024).contains(&self.expected_splits)
            || self.artifacts.len() < self.expected_splits as usize
            || self.artifacts.len() > 4096
            || [&self.gguf_prefix, &self.basename].iter().any(|s| {
                s.is_empty()
                    || s.len() > 128
                    || s.as_str() == "."
                    || s.as_str() == ".."
                    || !s
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            })
        {
            bail!("immutable complete quant roster admission refused");
        }
        let expected: BTreeSet<_> = (1..=self.expected_splits)
            .map(|i| {
                format!(
                    "{}/{}-{i:05}-of-{:05}.gguf",
                    self.gguf_prefix, self.basename, self.expected_splits
                )
            })
            .collect();
        let mut paths = BTreeSet::new();
        for a in &self.artifacts {
            let p = Plan {
                repo: self.repo.clone(),
                kind: RepositoryKind::Model,
                revision: self.commit.clone(),
                path: a.path.clone(),
                create_pr: false,
                maximum_attempts: 1,
                expected_parent: None,
            };
            p.validate()?;
            if !regular_publication::contract::hex(&a.sha256, 64)
                || a.byte_size == 0
                || a.byte_size > 1024_u64.pow(4)
                || !paths.insert(a.path.clone())
            {
                bail!("immutable quant identities duplicate/invalid");
            }
        }
        let declared: BTreeSet<_> = paths
            .into_iter()
            .filter(|p| {
                p.starts_with(&format!("{}/", self.gguf_prefix))
                    && p.to_ascii_lowercase().ends_with(".gguf")
            })
            .collect();
        if declared != expected {
            bail!("immutable final quant shard roster incomplete/foreign");
        }
        Ok(expected)
    }
}
impl Publisher {
    pub async fn acquire_quant_commit_until<C: Future<Output = ()>>(
        &self,
        input: &CommitRequest,
        root: &Path,
        until: Instant,
        cancellation: C,
    ) -> CommitReceipt {
        let mut receipt = CommitReceipt {
            repo: input.repo.clone(),
            commit: input.commit.clone(),
            artifact_root: root.into(),
            verified: Vec::new(),
            completed: false,
            error: None,
        };
        let mut cancel = Box::pin(cancellation.fuse());
        let pending = self.acquire_commit(input, root, until, &mut receipt);
        let result = match futures::future::select(
            cancel.as_mut(),
            Box::pin(tokio::time::timeout_at(until.into(), pending)),
        )
        .await
        {
            futures::future::Either::Left(_) => Err(anyhow!("quant commit acquisition cancelled")),
            futures::future::Either::Right((Err(_), _)) => {
                Err(anyhow!("quant commit acquisition deadline"))
            }
            futures::future::Either::Right((Ok(r), _)) => r,
        };
        let result = if cancel.as_mut().now_or_never().is_some() {
            Err(anyhow!("quant commit final cancellation"))
        } else {
            policy::check(until).and(result)
        };
        match result {
            Ok(()) => receipt.completed = true,
            Err(e) => receipt.error = Some(e.to_string()),
        }
        receipt
    }
    async fn acquire_commit(
        &self,
        input: &CommitRequest,
        root: &Path,
        until: Instant,
        receipt: &mut CommitReceipt,
    ) -> Result<()> {
        let expected = input.validate()?;
        policy::check(until)?;
        if !root.is_absolute()
            || root
                .parent()
                .ok_or_else(|| anyhow!("quant artifact parent"))?
                .canonicalize()?
                != root.parent().unwrap()
            || std::fs::symlink_metadata(root).is_ok()
        {
            bail!("fresh canonical quant artifact root required");
        }
        let plan = Plan {
            repo: input.repo.clone(),
            kind: RepositoryKind::Model,
            revision: input.commit.clone(),
            path: input.artifacts[0].path.clone(),
            create_pr: false,
            maximum_attempts: 1,
            expected_parent: None,
        };
        let response = self
            .client
            .http
            .get(self.url(&plan, &["revision", &input.commit], true)?)
            .bearer_auth(&self.client.token)
            .send()
            .await
            .map_err(|_| anyhow!("immutable quant roster lookup failed"))?;
        let info: serde_json::Value =
            serde_json::from_slice(&self.client.bounded(response, until).await?)?;
        if info["sha"] != input.commit {
            bail!("immutable quant roster commit differs");
        }
        let siblings = info["siblings"]
            .as_array()
            .ok_or_else(|| anyhow!("immutable quant sibling roster absent"))?;
        if siblings.len() > 10000 {
            bail!("immutable quant sibling roster bound");
        }
        let mut actual = BTreeSet::new();
        for row in siblings {
            let p = row["rfilename"]
                .as_str()
                .ok_or_else(|| anyhow!("quant sibling path malformed"))?;
            if p.starts_with(&format!("{}/", input.gguf_prefix))
                && p.to_ascii_lowercase().ends_with(".gguf")
                && !actual.insert(p.to_owned())
            {
                bail!("quant sibling duplicated");
            }
        }
        if actual != expected {
            bail!("immutable quant complete remote GGUF roster differs");
        }
        policy::check(until)?;
        std::fs::create_dir(root)?;
        for a in &input.artifacts {
            policy::check(until)?;
            let path = root.join(&a.path);
            std::fs::create_dir_all(path.parent().unwrap())?;
            let temporary = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
            let mut held = temporary.persist_noclobber(&path)?;
            self.client
                .acquire_repository_until(
                    RepositoryRead {
                        repo: &input.repo,
                        dataset: false,
                        commit: &input.commit,
                        path: &a.path,
                        size: a.byte_size,
                        expected_sha256: Some(&a.sha256),
                    },
                    &mut held,
                    until,
                )
                .await?;
            held.flush()?;
            held.sync_all()?;
            policy::check(until)?;
            receipt.verified.push(a.clone());
        }
        policy::check(until)
    }
}
