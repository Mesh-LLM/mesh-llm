use super::{
    Publisher,
    contract::{self, Artifact, Plan, Staged},
};
use anyhow::{Result, bail};
use base64::Engine as _;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, time::Instant};
pub(in crate::snapshot_promotion) fn file_body(file: File) -> reqwest::Body {
    let stream = futures::stream::try_unfold(file, |mut file| async move {
        let mut bytes = vec![0_u8; 65536];
        let count = file.read(&mut bytes)?;
        if count == 0 {
            return Ok::<_, std::io::Error>(None);
        }
        bytes.truncate(count);
        Ok(Some((bytes::Bytes::from(bytes), file)))
    });
    reqwest::Body::wrap_stream(stream)
}
pub(super) async fn bounded(
    mut response: reqwest::Response,
    cap: usize,
    deadline: Instant,
) -> Result<Vec<u8>> {
    if !response.status().is_success() {
        bail!("publication HTTP status {}", response.status().as_u16());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| anyhow::anyhow!("publication response stream failed"))?
    {
        contract::check(deadline)?;
        if bytes
            .len()
            .checked_add(chunk.len())
            .is_none_or(|len| len > cap)
        {
            bail!("publication response bound exceeded");
        }
        bytes.extend_from_slice(&chunk);
    }
    contract::check(deadline)?;
    Ok(bytes)
}
impl Publisher {
    pub(super) async fn classify(
        &self,
        plan: &Plan,
        staged: &Staged,
        deadline: Instant,
    ) -> Result<()> {
        let mut entries = Vec::new();
        for (path, file) in &staged.files {
            contract::check(deadline)?;
            let mut input = file.spool.reopen()?;
            let mut sample = [0; 512];
            let count = input.read(&mut sample)?;
            entries.push(serde_json::json!({"path":path,"size":file.identity.byte_size,"sample":base64::engine::general_purpose::STANDARD.encode(&sample[..count])}));
        }
        let response = self
            .http
            .post(self.url(plan, &["preupload", "main"], true)?)
            .bearer_auth(self.token.expose())
            .json(&serde_json::json!({"files":entries}))
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("preupload classification request failed"))?;
        let response: serde_json::Value =
            serde_json::from_slice(&bounded(response, 65536, deadline).await?)
                .map_err(|_| anyhow::anyhow!("malformed preupload classification"))?;
        let rows = response["files"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("preupload classification roster absent"))?;
        if rows.len() != staged.files.len() {
            bail!("preupload classification roster mismatch");
        }
        let mut seen = std::collections::BTreeSet::new();
        for row in rows {
            let path = row["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("preupload path absent"))?;
            if !staged.files.contains_key(path)
                || !seen.insert(path)
                || row["uploadMode"] != "regular"
                || row["shouldIgnore"].as_bool() == Some(true)
            {
                bail!("preupload regular mode refused; LFS upload remains unsupported");
            }
        }
        Ok(())
    }
    pub(super) async fn verify_remote(
        &self,
        plan: &Plan,
        path: &str,
        file: &Artifact,
        oid: &str,
        deadline: Instant,
    ) -> Result<()> {
        self.read_remote(plan, path, &file.identity, oid, deadline)
            .await
            .map(|_| ())
    }
    pub(super) async fn read_remote(
        &self,
        plan: &Plan,
        path: &str,
        identity: &super::super::policy::ArtifactIdentity,
        oid: &str,
        deadline: Instant,
    ) -> Result<Vec<u8>> {
        if identity.byte_size > 1024 * 1024 || !contract::hex(&identity.sha256, 64) {
            bail!("regular verification byte/pin bound refused");
        }
        let mut tail = vec!["resolve", oid];
        tail.extend(path.split('/'));
        let mut url = self.url(plan, &tail, false)?;
        for _ in 0..4 {
            contract::check(deadline)?;
            if url.origin() != self.origin.origin()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.fragment().is_some()
            {
                bail!("regular verification redirect origin refused");
            }
            let mut response = self
                .http
                .get(url.clone())
                .bearer_auth(self.token.expose())
                .send()
                .await
                .map_err(|_| anyhow::anyhow!("immutable regular verification request failed"))?;
            if response.status().is_redirection() {
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| anyhow::anyhow!("verification redirect location invalid"))?;
                url = url
                    .join(location)
                    .map_err(|_| anyhow::anyhow!("verification redirect malformed"))?;
                continue;
            }
            if response.status() != reqwest::StatusCode::OK {
                bail!(
                    "immutable regular verification HTTP status {}",
                    response.status().as_u16()
                );
            }
            let mut bytes = Vec::new();
            let mut hash = Sha256::new();
            let mut size = 0_u64;
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|_| anyhow::anyhow!("immutable verification body failed"))?
            {
                contract::check(deadline)?;
                size = size
                    .checked_add(chunk.len() as u64)
                    .ok_or_else(|| anyhow::anyhow!("verification byte overflow"))?;
                if size > identity.byte_size {
                    bail!("immutable verification body exceeds expected size");
                }
                hash.update(&chunk);
                bytes.extend_from_slice(&chunk);
            }
            let digest = hash
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            if size != identity.byte_size || digest != identity.sha256 {
                bail!("immutable remote byte identity mismatch");
            }
            contract::check(deadline)?;
            return Ok(bytes);
        }
        bail!("immutable verification redirect limit exceeded")
    }
}
