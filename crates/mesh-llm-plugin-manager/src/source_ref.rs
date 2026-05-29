use std::{error::Error, fmt, str::FromStr};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PluginInstallRef {
    Catalog {
        name: String,
        version: Option<PluginVersion>,
    },
    GitHub {
        source: GitHubPluginSource,
        version: Option<PluginVersion>,
    },
}

impl PluginInstallRef {
    pub fn parse(input: &str) -> Result<Self, PluginInstallRefParseError> {
        parse_install_ref(input)
    }

    pub fn requested_version(&self) -> Option<&PluginVersion> {
        match self {
            Self::Catalog { version, .. } | Self::GitHub { version, .. } => version.as_ref(),
        }
    }

    pub fn install_name(&self) -> &str {
        match self {
            Self::Catalog { name, .. } => name,
            Self::GitHub { source, .. } => &source.repo,
        }
    }
}

impl FromStr for PluginInstallRef {
    type Err = PluginInstallRefParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::parse(input)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitHubPluginSource {
    pub owner: String,
    pub repo: String,
}

impl GitHubPluginSource {
    pub fn from_url(url: &str) -> Result<Self, PluginInstallRefParseError> {
        match parse_install_ref(url)? {
            PluginInstallRef::GitHub { source, .. } => Ok(source),
            PluginInstallRef::Catalog { .. } => Err(PluginInstallRefParseError::new(
                url,
                "expected GitHub repository URL",
            )),
        }
    }

    pub fn repo_slug(&self) -> String {
        format!("{}/{}", self.owner, self.repo)
    }

    pub fn url(&self) -> String {
        format!("https://github.com/{}", self.repo_slug())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginVersion(String);

impl PluginVersion {
    pub fn new(value: impl Into<String>) -> Result<Self, PluginInstallRefParseError> {
        let value = value.into();
        if is_valid_version(&value) {
            Ok(Self(value))
        } else {
            Err(PluginInstallRefParseError::invalid_version(value))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn with_v_prefix(&self) -> String {
        if self.0.starts_with('v') {
            self.0.clone()
        } else {
            format!("v{}", self.0)
        }
    }

    pub fn without_v_prefix(&self) -> &str {
        self.0.strip_prefix('v').unwrap_or(&self.0)
    }

    pub fn matching_segments(&self) -> Vec<String> {
        let mut segments = vec![self.0.clone()];
        let alternate = if self.0.starts_with('v') {
            self.without_v_prefix().to_string()
        } else {
            self.with_v_prefix()
        };
        if alternate != self.0 {
            segments.push(alternate);
        }
        segments
    }
}

impl fmt::Display for PluginVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginInstallRefParseError {
    input: String,
    reason: &'static str,
}

impl PluginInstallRefParseError {
    fn new(input: impl Into<String>, reason: &'static str) -> Self {
        Self {
            input: input.into(),
            reason,
        }
    }

    fn invalid_version(input: impl Into<String>) -> Self {
        Self::new(input, "invalid version segment")
    }
}

impl fmt::Display for PluginInstallRefParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid plugin install reference '{}': {}",
            self.input, self.reason
        )
    }
}

impl Error for PluginInstallRefParseError {}

pub fn parse_install_ref(input: &str) -> Result<PluginInstallRef, PluginInstallRefParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(PluginInstallRefParseError::new(
            input,
            "reference cannot be empty",
        ));
    }

    if let Some(tail) = github_url_tail(trimmed) {
        return parse_github_tail(trimmed, tail);
    }

    if trimmed.contains("://") {
        return Err(PluginInstallRefParseError::new(
            input,
            "only GitHub repository URLs are supported",
        ));
    }

    if trimmed.contains('/') {
        return parse_github_tail(trimmed, trimmed);
    }

    parse_catalog_ref(trimmed)
}

fn parse_catalog_ref(input: &str) -> Result<PluginInstallRef, PluginInstallRefParseError> {
    let (name, version) = split_optional_version(input)?;
    if !is_valid_name(name) {
        return Err(PluginInstallRefParseError::new(
            input,
            "catalog name must contain only ASCII letters, digits, '.', '_', or '-'",
        ));
    }
    Ok(PluginInstallRef::Catalog {
        name: name.to_string(),
        version,
    })
}

fn parse_github_tail(
    input: &str,
    tail: &str,
) -> Result<PluginInstallRef, PluginInstallRefParseError> {
    let clean = tail
        .split_once('?')
        .map(|(left, _)| left)
        .unwrap_or(tail)
        .split_once('#')
        .map(|(left, _)| left)
        .unwrap_or(tail)
        .trim_matches('/');
    let mut parts = clean.split('/');
    let owner = parts.next().unwrap_or_default();
    let repo_and_version = parts.next().unwrap_or_default();
    if parts.next().is_some() || owner.is_empty() || repo_and_version.is_empty() {
        return Err(PluginInstallRefParseError::new(
            input,
            "expected GitHub repository as owner/repo",
        ));
    }

    let (repo, version) = split_optional_version(repo_and_version)?;
    let repo = repo.strip_suffix(".git").unwrap_or(repo);
    if !is_valid_github_owner(owner) || !is_valid_name(repo) {
        return Err(PluginInstallRefParseError::new(
            input,
            "GitHub owner or repository name contains unsupported characters",
        ));
    }

    Ok(PluginInstallRef::GitHub {
        source: GitHubPluginSource {
            owner: owner.to_string(),
            repo: repo.to_string(),
        },
        version,
    })
}

fn github_url_tail(input: &str) -> Option<&str> {
    input
        .strip_prefix("https://github.com/")
        .or_else(|| input.strip_prefix("http://github.com/"))
}

fn split_optional_version(
    value: &str,
) -> Result<(&str, Option<PluginVersion>), PluginInstallRefParseError> {
    let Some((left, right)) = value.rsplit_once('@') else {
        return Ok((value, None));
    };
    if left.is_empty() {
        return Err(PluginInstallRefParseError::new(
            value,
            "name before version cannot be empty",
        ));
    }
    Ok((left, Some(PluginVersion::new(right.to_string())?)))
}

fn is_valid_version(value: &str) -> bool {
    !value.is_empty()
        && !value.contains('/')
        && !value.contains('\\')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'+'))
}

fn is_valid_github_owner(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

pub(crate) fn is_valid_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_catalog_ref() {
        let parsed = parse_install_ref("blackboard").unwrap();
        assert_eq!(
            parsed,
            PluginInstallRef::Catalog {
                name: "blackboard".into(),
                version: None
            }
        );
    }

    #[test]
    fn parses_catalog_ref_with_version() {
        let parsed = parse_install_ref("blackboard@1.2.3").unwrap();
        assert_eq!(parsed.install_name(), "blackboard");
        assert_eq!(parsed.requested_version().unwrap().as_str(), "1.2.3");
    }

    #[test]
    fn parses_owner_repo_ref() {
        let parsed = parse_install_ref("mesh-llm/cool-plugin@v1.1.0").unwrap();
        assert_eq!(
            parsed,
            PluginInstallRef::GitHub {
                source: GitHubPluginSource {
                    owner: "mesh-llm".into(),
                    repo: "cool-plugin".into()
                },
                version: Some(PluginVersion("v1.1.0".into()))
            }
        );
    }

    #[test]
    fn parses_github_url_ref() {
        let parsed = parse_install_ref("https://github.com/mesh-llm/cool-plugin@1.1.0").unwrap();
        let PluginInstallRef::GitHub { source, version } = parsed else {
            panic!("expected github ref");
        };
        assert_eq!(source.repo_slug(), "mesh-llm/cool-plugin");
        assert_eq!(source.url(), "https://github.com/mesh-llm/cool-plugin");
        assert_eq!(
            version.unwrap().matching_segments(),
            vec!["1.1.0", "v1.1.0"]
        );
    }

    #[test]
    fn parses_github_source_from_url() {
        let source =
            GitHubPluginSource::from_url("https://github.com/mesh-llm/cool-plugin.git").unwrap();
        assert_eq!(source.repo_slug(), "mesh-llm/cool-plugin");
    }

    #[test]
    fn rejects_non_github_url() {
        let err = parse_install_ref("https://example.com/mesh-llm/cool-plugin").unwrap_err();
        assert!(err.to_string().contains("only GitHub"));
    }
}
