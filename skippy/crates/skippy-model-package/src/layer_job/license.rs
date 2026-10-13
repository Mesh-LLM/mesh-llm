//! Source-card license first, then the first explicitly declared base model; no architectural guess.
use super::SourceClient;
use anyhow::{Result, bail};
use serde::Serialize;
use serde_json::Value;
use std::time::Instant;
#[derive(Serialize)]
pub struct License {
    pub value: Option<String>,
    pub repo: Option<String>,
    pub revision: Option<String>,
    pub warning: Option<&'static str>,
}
fn text(value: &Value) -> Option<String> {
    value
        .as_str()
        .filter(|s| !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control))
        .map(str::to_owned)
}
fn base(value: &Value) -> Option<String> {
    let value = value.as_array().and_then(|a| a.first()).unwrap_or(value);
    text(value).or_else(|| {
        value
            .as_object()
            .and_then(|o| text(o.get("id").or_else(|| o.get("modelId"))?))
    })
}
impl SourceClient {
    async fn card_info(
        &self,
        repo: &str,
        reference: &str,
        until: Instant,
    ) -> Result<(String, Value)> {
        super::repo(repo)?;
        let (owner, name) = repo
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("license repo refused"))?;
        let response = self
            .request(
                self.url(&["api", "models", owner, name, "revision", reference])?,
                until,
            )
            .await?;
        if response.status() != reqwest::StatusCode::OK {
            bail!("license metadata HTTP refusal");
        }
        let value: Value =
            serde_json::from_slice(&super::source::body(response, 262144, until).await?)
                .map_err(|_| anyhow::anyhow!("license metadata schema refused"))?;
        let pin = value["sha"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("license metadata identity absent"))?;
        super::revision(pin)?;
        if reference != "main" && pin != reference {
            bail!("license source pin mismatch");
        }
        Ok((pin.into(), value["cardData"].clone()))
    }
    pub async fn license_until(&self, repo: &str, pin: &str, until: Instant) -> License {
        let result = async {
            super::revision(pin)?;
            let (source_pin, card) = self.card_info(repo, pin, until).await?;
            if let Some(value) = text(&card["license"]) {
                return Ok(Some((value, repo.to_owned(), source_pin)));
            }
            if let Some(base_repo) = base(&card["base_model"]) {
                let (base_pin, base_card) = self.card_info(&base_repo, "main", until).await?;
                if let Some(value) = text(&base_card["license"]) {
                    return Ok(Some((value, base_repo, base_pin)));
                }
            }
            Ok::<_, anyhow::Error>(None)
        }
        .await;
        match result {
            Ok(Some((value, repo, revision))) => License {
                value: Some(value),
                repo: Some(repo),
                revision: Some(revision),
                warning: None,
            },
            Ok(None) => License {
                value: None,
                repo: None,
                revision: None,
                warning: None,
            },
            Err(_) => License {
                value: None,
                repo: None,
                revision: None,
                warning: Some("upstream license metadata unavailable; no license inferred"),
            },
        }
    }
}
