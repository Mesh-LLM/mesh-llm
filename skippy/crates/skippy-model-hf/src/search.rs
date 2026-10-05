//! Hugging Face repository discovery shared by the two model CLIs.

use anyhow::{Context, Result};
use hf_hub::repository::ModelInfo;
use regex_lite::Regex;
use std::sync::LazyLock;
use tokio_stream::StreamExt;

use crate::HfModelRepository;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactFilter {
    Gguf,
    Mlx,
}

impl ArtifactFilter {
    fn hub_filter(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Mlx => "mlx",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sort {
    Trending,
    Downloads,
    Likes,
    Created,
    Updated,
    ParametersDesc,
    ParametersAsc,
}

impl Sort {
    fn api_key(self) -> Option<&'static str> {
        match self {
            Self::Trending => Some("trendingScore"),
            Self::Downloads => Some("downloads"),
            Self::Likes => Some("likes"),
            Self::Created => Some("createdAt"),
            Self::Updated => Some("lastModified"),
            Self::ParametersDesc | Self::ParametersAsc => None,
        }
    }
}

pub async fn search_repositories(
    query: &str,
    limit: usize,
    filter: ArtifactFilter,
    sort: Sort,
) -> Result<Vec<ModelInfo>> {
    let repo_limit = match sort {
        Sort::ParametersDesc | Sort::ParametersAsc => (limit.saturating_mul(5)).clamp(1, 100),
        _ => limit.clamp(1, 100),
    };
    let client = HfModelRepository::from_env()?.api;
    let request = client
        .list_models()
        .search(query.to_string())
        .filter(filter.hub_filter().to_string())
        .full(true)
        .limit(repo_limit);
    if let Some(key) = sort.api_key() {
        collect_repositories(
            request
                .sort(key.to_string())
                .send()
                .context("search Hugging Face")?,
        )
        .await
    } else {
        collect_repositories(request.send().context("search Hugging Face")?).await
    }
}

async fn collect_repositories(
    stream: impl tokio_stream::Stream<Item = std::result::Result<ModelInfo, hf_hub::HFError>>,
) -> Result<Vec<ModelInfo>> {
    tokio::pin!(stream);
    let mut repos = Vec::new();
    while let Some(repo) = stream.next().await {
        repos.push(repo.context("read Hugging Face repository summary")?);
    }
    Ok(repos)
}

pub fn approximate_parameter_count_b_from_text(text: &str) -> Option<f64> {
    static MULTIPLIED: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)([bm])").expect("constant regex")
    });
    static SIMPLE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)([bm])").expect("constant regex"));
    let mut best: Option<f64> = None;
    for captures in MULTIPLIED.captures_iter(text) {
        let (Some(left), Some(right), Some(unit)) = (
            captures
                .get(1)
                .and_then(|value| value.as_str().parse::<f64>().ok()),
            captures
                .get(2)
                .and_then(|value| value.as_str().parse::<f64>().ok()),
            captures
                .get(3)
                .map(|value| value.as_str().to_ascii_lowercase()),
        ) else {
            continue;
        };
        let value = if unit == "b" {
            left * right
        } else {
            left * right / 1000.0
        };
        best = Some(best.map_or(value, |current| current.max(value)));
    }
    for captures in SIMPLE.captures_iter(text) {
        let (Some(number), Some(unit)) = (
            captures
                .get(1)
                .and_then(|value| value.as_str().parse::<f64>().ok()),
            captures
                .get(2)
                .map(|value| value.as_str().to_ascii_lowercase()),
        ) else {
            continue;
        };
        let value = if unit == "b" { number } else { number / 1000.0 };
        best = Some(best.map_or(value, |current| current.max(value)));
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_sort_recognizes_moe_and_dense_names() {
        assert_eq!(
            approximate_parameter_count_b_from_text("Qwen3-235B-A22B"),
            Some(235.0)
        );
        assert_eq!(approximate_parameter_count_b_from_text("8x7B"), Some(56.0));
        assert_eq!(approximate_parameter_count_b_from_text("500M"), Some(0.5));
    }
}
