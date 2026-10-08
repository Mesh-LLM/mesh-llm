use super::{Client, Object, Receipt, custody, protocol::Instructions};
use anyhow::{Result, anyhow, bail};
use std::time::Instant;
impl Client {
    pub(super) async fn transfer(
        &self,
        object: &mut Object,
        instructions: Instructions,
        until: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        let Some(upload) = instructions.upload else {
            receipt.object_present = true;
            return Ok(());
        };
        let address = self.action(&upload)?;
        if let Some(chunk) = upload["header"].get("chunk_size") {
            let chunk = chunk
                .as_str()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|n| *n > 0 && *n <= 5 * 1024 * 1024 * 1024)
                .ok_or_else(|| anyhow!("LFS multipart chunk size refused"))?;
            let count = object.size.div_ceil(chunk);
            if count > 10000 {
                bail!("LFS multipart part bound refused");
            }
            let headers = upload["header"]
                .as_object()
                .ok_or_else(|| anyhow!("LFS multipart headers invalid"))?;
            if headers.len() != count as usize + 1 {
                bail!("LFS multipart part roster mismatch");
            }
            let mut urls = Vec::new();
            for n in 1..=count {
                urls.push(
                    self.address(
                        headers
                            .get(&n.to_string())
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| anyhow!("LFS numbered part URL absent"))?,
                    )?,
                );
            }
            let mut parts = Vec::new();
            for (index, url) in urls.into_iter().enumerate() {
                custody::check(until)?;
                let start = index as u64 * chunk;
                let size = chunk.min(object.size - start);
                let request = self
                    .http
                    .put(url)
                    .header("Content-Type", "application/octet-stream")
                    .header("Content-Length", size.to_string())
                    .body(object.body(start, size, until)?);
                receipt.mutation_attempted = true;
                let response = request
                    .send()
                    .await
                    .map_err(|_| anyhow!("LFS part upload outcome unconfirmed"))?;
                let etag = response
                    .headers()
                    .get(reqwest::header::ETAG)
                    .and_then(|v| v.to_str().ok())
                    .filter(|v| !v.is_empty() && v.len() <= 1024)
                    .ok_or_else(|| anyhow!("LFS part ETag absent/oversized"))?
                    .to_owned();
                self.bounded(response, until).await?;
                receipt.uploaded_parts += 1;
                parts.push(serde_json::json!({"partNumber":index+1,"etag":etag}));
            }
            custody::check(until)?;
            // Signed completion action is used without adding the user's HF bearer credential.
            let response = self
                .http
                .post(address)
                .header("Accept", "application/vnd.git-lfs+json")
                .header("Content-Type", "application/vnd.git-lfs+json")
                .json(&serde_json::json!({"oid":object.oid,"parts":parts}))
                .send()
                .await
                .map_err(|_| anyhow!("LFS completion outcome unconfirmed"))?;
            self.bounded(response, until).await?;
        } else {
            let headers = self.headers(&upload)?;
            custody::check(until)?;
            let request = self
                .http
                .put(address)
                .headers(headers)
                .header("Content-Type", "application/octet-stream")
                .header("Content-Length", object.size.to_string())
                .body(object.body(0, object.size, until)?);
            receipt.mutation_attempted = true;
            let response = request
                .send()
                .await
                .map_err(|_| anyhow!("LFS basic upload outcome unconfirmed"))?;
            self.bounded(response, until).await?;
            receipt.uploaded_parts = 1;
        }
        if let Some(verify) = instructions.verify {
            custody::check(until)?;
            let address = self.action(&verify)?;
            let same_origin = address.origin() == self.origin.origin();
            let request = self
                .http
                .post(address)
                .headers(self.headers(&verify)?)
                .header("Accept", "application/vnd.git-lfs+json")
                .header("Content-Type", "application/vnd.git-lfs+json")
                .json(&serde_json::json!({"oid":object.oid,"size":object.size}));
            let request = if same_origin {
                request.bearer_auth(&self.token)
            } else {
                request
            };
            let response = request
                .send()
                .await
                .map_err(|_| anyhow!("LFS verification outcome unconfirmed"))?;
            if response.status() != reqwest::StatusCode::OK {
                bail!("LFS verification did not confirm presence");
            }
            self.bounded(response, until).await?;
        }
        receipt.object_present = true;
        Ok(())
    }
}
