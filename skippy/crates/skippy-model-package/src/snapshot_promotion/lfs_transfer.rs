//! Actual bounded basic/multipart LFS object upload; repository commit is a separate owner.
mod custody;
mod protocol;
mod transfer;
mod verification;
use anyhow::{Result, anyhow};
pub use custody::Object;
use futures::{Future, FutureExt};
use serde::Serialize;
use std::time::Instant;
pub(crate) use verification::RepositoryRead;
#[derive(Serialize)]
pub struct Receipt {
    pub oid: String,
    pub size: u64,
    pub mutation_attempted: bool,
    pub uploaded_parts: usize,
    pub object_present: bool,
    pub source_custody_verified: bool,
    pub completed: bool,
    pub error: Option<String>,
}
pub struct Client {
    pub(super) http: reqwest::Client,
    pub(super) origin: reqwest::Url,
    pub(super) token: String,
}
impl Client {
    pub fn new(token: String) -> Result<Self> {
        if token.is_empty() || token.len() > 8192 || token.bytes().any(|b| !b.is_ascii_graphic()) {
            return Err(anyhow!("LFS explicit credential refused"));
        }
        let _ = skippy_model_hf::configure_hf_tls_provider();
        Ok(Self {
            http: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|_| anyhow!("LFS client initialization failed"))?,
            origin: reqwest::Url::parse("https://huggingface.co")?,
            token,
        })
    }
    /// Drops the owned pending transport on caller cancellation. No retry or final commit.
    pub async fn upload_until<C: Future<Output = ()>>(
        &self,
        repo: &str,
        mut object: Object,
        until: Instant,
        cancellation: C,
    ) -> Receipt {
        let mut receipt = Receipt {
            oid: object.oid.clone(),
            size: object.size,
            mutation_attempted: false,
            uploaded_parts: 0,
            object_present: false,
            source_custody_verified: false,
            completed: false,
            error: None,
        };
        let result = {
            let operation = tokio::time::timeout_at(
                until.into(),
                self.execute(repo, &mut object, until, &mut receipt),
            );
            match futures::future::select(cancellation.boxed_local(), operation.boxed_local()).await
            {
                futures::future::Either::Left(_) => {
                    Err(anyhow!("LFS cancelled; object mutation may be unconfirmed"))
                }
                futures::future::Either::Right((Err(_), _)) => Err(anyhow!(
                    "LFS deadline expired; object mutation may be unconfirmed"
                )),
                futures::future::Either::Right((Ok(result), cancellation)) => {
                    if cancellation.now_or_never().is_some() {
                        Err(anyhow!("LFS cancelled at terminal boundary"))
                    } else {
                        custody::check(until).and(result)
                    }
                }
            }
        };
        match result {
            Ok(()) => receipt.completed = true,
            Err(error) => receipt.error = Some(error.to_string()),
        }
        receipt
    }
    async fn execute(
        &self,
        repo: &str,
        object: &mut Object,
        until: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        custody::repo(repo)?;
        object.verify(until)?;
        let batch = self.batch(repo, object, until).await?;
        self.transfer(object, batch, until, receipt).await?;
        object.verify(until)?;
        custody::check(until)?;
        receipt.source_custody_verified = true;
        Ok(())
    }
}
#[cfg(test)]
#[path = "lfs_transfer/tests.rs"]
mod tests;
