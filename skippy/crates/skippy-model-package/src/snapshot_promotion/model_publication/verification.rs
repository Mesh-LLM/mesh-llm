use super::{ArtifactIdentity, Publisher};
use anyhow::Result;
use std::time::Instant;
impl Publisher {
    pub(super) async fn verify(
        &self,
        repo: &str,
        oid: &str,
        path: &str,
        identity: &ArtifactIdentity,
        until: Instant,
    ) -> Result<()> {
        self.client
            .verify_repository_until(repo, false, oid, path, identity, until)
            .await
    }
}
