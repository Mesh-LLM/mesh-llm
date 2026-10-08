use super::{Client, Object, custody};
use anyhow::{Result, anyhow, bail};
use serde_json::Value;
use std::time::Instant;
pub(super) struct Instructions {
    pub upload: Option<Value>,
    pub verify: Option<Value>,
}
impl Client {
    pub(in crate::snapshot_promotion) async fn bounded(
        &self,
        mut response: reqwest::Response,
        until: Instant,
    ) -> Result<Vec<u8>> {
        if !response.status().is_success() {
            bail!("LFS HTTP status {}", response.status().as_u16());
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| anyhow!("LFS response stream failed"))?
        {
            custody::check(until)?;
            if chunk.len() > 65536_usize.saturating_sub(bytes.len()) {
                bail!("LFS response bound exceeded");
            }
            bytes.extend_from_slice(&chunk);
        }
        custody::check(until)?;
        Ok(bytes)
    }
    pub(super) async fn batch(
        &self,
        repo: &str,
        object: &Object,
        until: Instant,
    ) -> Result<Instructions> {
        let mut url = self.origin.clone();
        {
            let mut parts = url
                .path_segments_mut()
                .map_err(|_| anyhow!("LFS origin refused"))?;
            parts.pop_if_empty();
            let (owner, name) = repo.split_once('/').unwrap();
            parts.extend([
                owner,
                &format!("{name}.git"),
                "info",
                "lfs",
                "objects",
                "batch",
            ]);
        }
        let response=self.http.post(url).bearer_auth(&self.token)
            .header("Accept","application/vnd.git-lfs+json").header("Content-Type","application/vnd.git-lfs+json")
            .json(&serde_json::json!({"operation":"upload","transfers":["basic","multipart"],"hash_algo":"sha256","objects":[{"oid":object.oid,"size":object.size}]})).send().await.map_err(|_|anyhow!("LFS batch request failed"))?;
        let value: Value = serde_json::from_slice(&self.bounded(response, until).await?)
            .map_err(|_| anyhow!("LFS batch JSON invalid"))?;
        if !matches!(
            value["transfer"].as_str(),
            None | Some("basic" | "multipart")
        ) {
            bail!("LFS selected transfer unsupported; no inline fallback");
        }
        let rows = value["objects"]
            .as_array()
            .filter(|rows| rows.len() == 1)
            .ok_or_else(|| anyhow!("LFS batch object roster mismatch"))?;
        let row = &rows[0];
        if row["oid"] != object.oid
            || row["size"].as_u64() != Some(object.size)
            || row.get("error").is_some()
        {
            bail!("LFS batch object identity/refusal");
        }
        let Some(actions) = row.get("actions") else {
            return Ok(Instructions {
                upload: None,
                verify: None,
            });
        };
        let actions = actions
            .as_object()
            .ok_or_else(|| anyhow!("LFS actions invalid"))?;
        if actions.keys().any(|key| key != "upload" && key != "verify") {
            bail!("LFS action unsupported");
        }
        let upload = actions
            .get("upload")
            .cloned()
            .ok_or_else(|| anyhow!("LFS upload action absent"))?;
        // Admit every address before the first object mutation.
        self.action(&upload)?;
        if let Some(verify) = actions.get("verify") {
            self.action(verify)?;
        }
        Ok(Instructions {
            upload: Some(upload),
            verify: actions.get("verify").cloned(),
        })
    }
    pub(in crate::snapshot_promotion) fn address(&self, value: &str) -> Result<reqwest::Url> {
        if value.len() > 8192 || value.bytes().any(|b| b.is_ascii_control()) {
            bail!("LFS action URL bound refused");
        }
        let url = reqwest::Url::parse(value).map_err(|_| anyhow!("LFS action URL invalid"))?;
        let host = url.host_str().unwrap_or("");
        let admitted = host == "huggingface.co"
            || host.ends_with(".huggingface.co")
            || host == "hf.co"
            || host.ends_with(".hf.co")
            || host.ends_with(".amazonaws.com");
        if (url.origin() != self.origin.origin()
            && (url.scheme() != "https" || !admitted || url.port().is_some()))
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
        {
            bail!("LFS action HTTPS storage origin refused");
        }
        Ok(url)
    }
    pub(super) fn action(&self, value: &Value) -> Result<reqwest::Url> {
        self.address(
            value["href"]
                .as_str()
                .ok_or_else(|| anyhow!("LFS action address absent"))?,
        )
    }
    pub(super) fn headers(&self, value: &Value) -> Result<reqwest::header::HeaderMap> {
        let mut output = reqwest::header::HeaderMap::new();
        let mut total = 0;
        if let Some(headers) = value.get("header") {
            let headers = headers
                .as_object()
                .ok_or_else(|| anyhow!("LFS action headers invalid"))?;
            if headers.len() > 32 {
                bail!("LFS action header count refused");
            }
            for (name, value) in headers {
                let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| anyhow!("LFS action header name refused"))?;
                if matches!(
                    name.as_str(),
                    "host"
                        | "cookie"
                        | "proxy-authorization"
                        | "content-length"
                        | "transfer-encoding"
                        | "connection"
                ) {
                    bail!("LFS action header reserved");
                }
                let value = value
                    .as_str()
                    .ok_or_else(|| anyhow!("LFS action header value refused"))?;
                total += name.as_str().len() + value.len();
                if total > 16384 {
                    bail!("LFS action header bytes refused");
                }
                output.insert(
                    name,
                    reqwest::header::HeaderValue::from_str(value)
                        .map_err(|_| anyhow!("LFS action header value refused"))?,
                );
            }
        }
        Ok(output)
    }
}
