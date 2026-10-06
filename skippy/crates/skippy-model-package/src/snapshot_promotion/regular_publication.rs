//! Parent-bound regular-file publication precursor; GGUF/LFS model upload is unsupported.
pub(super) mod contract;
pub(super) mod exchange;
use anyhow::{Result, bail};
use contract::Staged;
pub use contract::{LocalFile, Plan, Secret};
use futures::{Future, FutureExt};
use serde::Serialize;
use std::time::Instant;
#[derive(Serialize)]
pub struct Receipt {
    pub schema_version: u32,
    pub repo: String,
    pub parent_commit: String,
    pub commit_oid: Option<String>,
    pub mutation_attempted: bool,
    pub remote_verified_paths: Vec<String>,
    pub source_custody_verified: bool,
    pub completed: bool,
    pub error: Option<String>,
}
pub struct Publisher {
    http: reqwest::Client,
    origin: reqwest::Url,
    token: Secret,
}
impl Publisher {
    pub fn new(token: Secret) -> Result<Self> {
        let _ = skippy_model_hf::configure_hf_tls_provider();
        Ok(Self {
            http: reqwest::Client::builder()
                .no_proxy()
                .pool_max_idle_per_host(0)
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            origin: reqwest::Url::parse("https://huggingface.co")?,
            token,
        })
    }
    /// No ambient endpoint/auth fallback. Cancellation drops owned transport futures.
    /// A failed/uncertain mutation is never automatically retried.
    pub async fn publish_until<C: Future<Output = ()>>(
        &self,
        plan: &Plan,
        files: Vec<LocalFile>,
        deadline: Instant,
        cancellation: C,
    ) -> Receipt {
        self.publish_observed_until(plan, files, deadline, cancellation, &mut |_| Ok(()))
            .await
    }
    /// Observe conservative mutation attempts and known immutable commit/verification state.
    /// Observer failure refuses completion; the existing owned transport is not duplicated.
    pub async fn publish_observed_until<C: Future<Output = ()>>(
        &self,
        plan: &Plan,
        files: Vec<LocalFile>,
        deadline: Instant,
        cancellation: C,
        observer: &mut dyn FnMut(&Receipt) -> Result<()>,
    ) -> Receipt {
        let mut receipt = Receipt {
            schema_version: 1,
            repo: plan.repo.clone(),
            parent_commit: plan.parent_commit.clone(),
            commit_oid: None,
            mutation_attempted: false,
            remote_verified_paths: Vec::new(),
            source_custody_verified: false,
            completed: false,
            error: None,
        };
        let result = {
            let operation = tokio::time::timeout_at(
                tokio::time::Instant::from_std(deadline),
                self.attempt(plan, files, deadline, &mut receipt, observer),
            );
            match futures::future::select(cancellation.boxed_local(), operation.boxed_local()).await
            {
                futures::future::Either::Right((Ok(result), cancellation)) => {
                    terminal(deadline, cancellation.now_or_never().is_some()).and(result)
                }
                futures::future::Either::Right((Err(_), _)) => Err(anyhow::anyhow!(
                    "publication deadline expired; mutation may be unconfirmed"
                )),
                futures::future::Either::Left(((), _)) => Err(anyhow::anyhow!(
                    "publication cancelled; mutation may be unconfirmed"
                )),
            }
        };
        match result {
            Ok(()) => receipt.completed = true,
            Err(error) => receipt.error = Some(error.to_string()),
        }
        receipt
    }
    async fn attempt(
        &self,
        plan: &Plan,
        files: Vec<LocalFile>,
        deadline: Instant,
        receipt: &mut Receipt,
        observer: &mut dyn FnMut(&Receipt) -> Result<()>,
    ) -> Result<()> {
        plan.validate()?;
        let mut staged = Staged::new(plan, files, deadline)?;
        self.classify(plan, &staged, deadline).await?;
        let mut payload = tempfile::NamedTempFile::new()?;
        serde_json::to_writer(
            payload.as_file_mut(),
            &serde_json::json!({"key":"header","value":{"summary":"Publish pinned regular metadata","description":"","parentCommit":plan.parent_commit}}),
        )?;
        use std::io::{Seek, Write};
        payload.write_all(b"\n")?;
        for (path, artifact) in &mut staged.files {
            contract::check(deadline)?;
            artifact.spool.as_file_mut().rewind()?;
            super::commit_payload::write_regular_row(
                path,
                &artifact.identity,
                artifact.spool.as_file_mut(),
                payload.as_file_mut(),
            )?;
        }
        payload.flush()?;
        payload.as_file().sync_all()?;
        let size = payload.as_file().metadata()?.len();
        if size > 12 * 1024 * 1024 {
            bail!("regular commit payload bound exceeded");
        }
        contract::check(deadline)?;
        let request = self
            .http
            .post(self.url(plan, &["commit", "main"], true)?)
            .bearer_auth(self.token.expose())
            .header("Content-Type", "application/x-ndjson")
            .header("Content-Length", size.to_string())
            .body(exchange::file_body(payload.reopen()?));
        receipt.mutation_attempted = true;
        observer(receipt)?;
        // The synchronous checkpoint can consume the allowance; refuse before send.
        contract::check(deadline)?;
        let response = request
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("commit request outcome unconfirmed"))?;
        let value: serde_json::Value =
            serde_json::from_slice(&exchange::bounded(response, 65536, deadline).await?)
                .map_err(|_| anyhow::anyhow!("malformed commit receipt; outcome unconfirmed"))?;
        let oid = value["commitOid"]
            .as_str()
            .filter(|v| contract::hex(v, 40))
            .ok_or_else(|| anyhow::anyhow!("invalid commit receipt identity; outcome unconfirmed"))?
            .to_owned();
        receipt.commit_oid = Some(oid.clone());
        observer(receipt)?;
        for (path, artifact) in &staged.files {
            self.verify_remote(plan, path, artifact, &oid, deadline)
                .await?;
            receipt.remote_verified_paths.push(path.clone());
            observer(receipt)?;
        }
        staged.recheck(deadline)?;
        contract::check(deadline)?;
        receipt.source_custody_verified = true;
        observer(receipt)?;
        Ok(())
    }
    fn url(&self, plan: &Plan, tail: &[&str], api: bool) -> Result<reqwest::Url> {
        let mut url = self.origin.clone();
        let mut parts = url
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("publication origin refused"))?;
        parts.pop_if_empty();
        if api {
            parts.extend(["api", "models"]);
        }
        parts.extend(plan.repo.split('/'));
        parts.extend(tail.iter().copied());
        drop(parts);
        Ok(url)
    }
}
#[cfg(test)]
#[path = "regular_publication/tests.rs"]
mod tests;

fn terminal(deadline: Instant, cancelled: bool) -> Result<()> {
    if cancelled {
        bail!("publication cancelled at terminal boundary; mutation may be unconfirmed");
    }
    contract::check(deadline)
}

impl Publisher {
    /// Read only the bounded immutable JSON bytes through the same origin/redirect/TLS owner.
    /// This performs no commit and does not interpret remote Jobs state as certification.
    pub async fn retrieve_json_until<C: Future<Output = ()>>(
        &self,
        plan: &Plan,
        path: &str,
        oid: &str,
        identity: &super::policy::ArtifactIdentity,
        deadline: Instant,
        cancellation: C,
    ) -> Result<Vec<u8>> {
        plan.validate()?;
        if plan.paths != [path]
            || !path.ends_with(".json")
            || !contract::hex(oid, 40)
            || !contract::hex(&identity.sha256, 64)
            || identity.byte_size > 1024 * 1024
        {
            bail!("immutable JSON retrieval admission refused");
        }
        let operation = tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            self.read_remote(plan, path, identity, oid, deadline),
        );
        match futures::future::select(cancellation.boxed_local(), operation.boxed_local()).await {
            futures::future::Either::Left(_) => bail!("immutable JSON retrieval cancelled"),
            futures::future::Either::Right((Err(_), _)) => {
                bail!("immutable JSON retrieval deadline expired")
            }
            futures::future::Either::Right((Ok(result), retained)) => {
                terminal(deadline, retained.now_or_never().is_some())?;
                result
            }
        }
    }
}
#[cfg(test)]
#[path = "regular_publication/retrieval_tests.rs"]
mod retrieval_tests;

#[cfg(test)]
pub(crate) fn fixture_retrieval_publisher(origin: &str) -> Publisher {
    Publisher {
        http: reqwest::Client::builder()
            .no_proxy()
            .pool_max_idle_per_host(0)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap(),
        origin: reqwest::Url::parse(origin).unwrap(),
        token: Secret::new("fixture-token".into()).unwrap(),
    }
}

#[cfg(test)]
#[path = "regular_publication/fixture.rs"]
mod fixture;
