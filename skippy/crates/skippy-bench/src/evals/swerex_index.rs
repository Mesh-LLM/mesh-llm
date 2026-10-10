//! Exact SWE-ReX Docker pip-index patch; environment pin admission is a separate owner.
use anyhow::{Result, bail};
use std::path::Path;

#[cfg(unix)]
use super::swerex_source as source;

#[cfg(any(unix, test))]
const OLD: &str = r#"f"RUN /root/python3.11/bin/pip3 install --no-cache-dir {PACKAGE_NAME}\n\n""#;
#[cfg(any(unix, test))]
const INDEX_PREFIX: &str = r#"f"RUN /root/python3.11/bin/pip3 install --index-url "#;
#[cfg(any(unix, test))]
const MAX_URL_BYTES: usize = 8 * 1024;

#[cfg(any(unix, test))]
fn replacement(text: &str, index_url: &str) -> Result<Option<String>> {
    // A URL becomes part of an upstream Docker RUN command and a Python literal.
    // Single quotes protect shell globbing; reject delimiters of either source language.
    // Do not log parser errors containing credentials.
    let allowed = |ch: char| {
        !ch.is_control() && !ch.is_whitespace() && !matches!(ch, '\'' | '"' | '\\' | '{' | '}')
    };
    if index_url.is_empty() || index_url.len() > MAX_URL_BYTES || !index_url.chars().all(allowed) {
        bail!("SWE-ReX pip index must be a bounded safe HTTP(S) URL");
    }
    let url = reqwest::Url::parse(index_url)
        .map_err(|_| anyhow::anyhow!("invalid SWE-ReX pip index URL"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none_or(str::is_empty) {
        bail!("SWE-ReX pip index requires HTTP(S) and a nonempty host");
    }
    let new = format!(r#"{INDEX_PREFIX}'{index_url}' --no-cache-dir {{PACKAGE_NAME}}\n\n""#);
    match (
        text.matches(OLD).count(),
        text.matches(INDEX_PREFIX).count(),
        text.matches(&new).count(),
    ) {
        (1, 0, 0) => Ok(Some(text.replacen(OLD, &new, 1))),
        (0, 1, 1) => Ok(None),
        _ => bail!("unknown or ambiguous SWE-ReX Docker pip installation source"),
    }
}

#[cfg(unix)]
pub(super) fn patch(module_source: &Path, environment_root: &Path, index_url: &str) -> Result<()> {
    let admitted = source::AdmittedSource::open(module_source, environment_root)?;
    let text = std::str::from_utf8(&admitted.bytes)
        .map_err(|_| anyhow::anyhow!("SWE-ReX module source is not UTF-8"))?;
    if let Some(updated) = replacement(text, index_url)? {
        admitted.replace(updated.as_bytes())?;
    }
    Ok(())
}

#[cfg(not(unix))]
pub(super) fn patch(
    _module_source: &Path,
    _environment_root: &Path,
    _index_url: &str,
) -> Result<()> {
    bail!("the SWE-ReX adapter source patch requires Unix");
}

#[cfg(test)]
mod tests;
/// Read-only prepared profile admission against the exact official locked SDK bytes.
#[cfg(unix)]
pub(super) fn admit_current(
    module_source: &Path,
    environment_root: &Path,
    index_url: &str,
) -> Result<()> {
    let original = include_str!("swe_environment/docker-original.txt");
    let expected = replacement(original, index_url)?
        .ok_or_else(|| anyhow::anyhow!("compiled SWE Docker preimage is not original"))?;
    let admitted = source::AdmittedSource::open(module_source, environment_root)?;
    if admitted.bytes != expected.as_bytes() {
        bail!("prepared SWE Docker profile differs from current compiled source");
    }
    Ok(())
}
#[cfg(not(unix))]
pub(super) fn admit_current(
    _module_source: &Path,
    _environment_root: &Path,
    _index_url: &str,
) -> Result<()> {
    bail!("prepared SWE Docker profile requires Unix");
}
