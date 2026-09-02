//! Release manifest discovery, download, and verification.

use crate::cache::current_skippy_abi_version;
use crate::discovery::discover_native_runtime_bundle_dirs;
use crate::types::{NATIVE_RUNTIME_MANIFEST_URL_ENV, NativeRuntimeManifestOptions};
use anyhow::{Context, Result, bail};
use mesh_llm_native_runtime::{
    NativeRuntimeArtifact, NativeRuntimeManifest, NativeRuntimeReleaseManifest,
};
use sha2::Digest;
use std::path::PathBuf;
use std::time::Duration;
pub fn default_release_manifest_url(mesh_version: &str) -> String {
    format!(
        "https://github.com/Mesh-LLM/mesh-llm/releases/download/v{mesh_version}/native-runtimes.json"
    )
}

pub fn default_manifest_url(build_version: &str, release_version: &str) -> String {
    if mesh_llm_build_info::is_sha_build(build_version) {
        "https://github.com/Mesh-LLM/mesh-llm/releases/latest/download/native-runtimes.json"
            .to_string()
    } else {
        default_release_manifest_url(release_version)
    }
}

pub(crate) fn request_default_manifest_url(mesh_version: &str) -> String {
    if mesh_version == mesh_llm_build_info::RELEASE_VERSION {
        default_manifest_url(
            mesh_llm_build_info::BUILD_VERSION,
            mesh_llm_build_info::RELEASE_VERSION,
        )
    } else {
        default_release_manifest_url(mesh_version)
    }
}

pub async fn load_release_manifest(
    options: NativeRuntimeManifestOptions,
) -> Result<NativeRuntimeReleaseManifest> {
    Ok(load_release_manifest_with_bundle_dirs(options).await?.0)
}

pub(crate) async fn load_release_manifest_with_bundle_dirs(
    mut options: NativeRuntimeManifestOptions,
) -> Result<(NativeRuntimeReleaseManifest, Vec<PathBuf>)> {
    options.bundle_dirs = discover_native_runtime_bundle_dirs(&options.bundle_dirs)?;
    let mut artifacts = Vec::new();
    let mut mesh_version = options.mesh_version.clone();
    let mut skippy_abi = current_skippy_abi_version();
    if let Some(path) = options.manifest_path {
        let manifest = NativeRuntimeReleaseManifest::read_from_path(&path)?;
        mesh_version = manifest.mesh_version.clone();
        skippy_abi = manifest.skippy_abi.clone();
        artifacts.extend(manifest.artifacts);
    } else if let Some(url) = manifest_url(&options) {
        let manifest = download_release_manifest(&url).await?;
        mesh_version = manifest.mesh_version.clone();
        skippy_abi = manifest.skippy_abi.clone();
        artifacts.extend(manifest.artifacts);
    }
    append_bundle_artifacts(
        &mut artifacts,
        &mut mesh_version,
        &mut skippy_abi,
        &options.bundle_dirs,
    )?;
    Ok((
        NativeRuntimeReleaseManifest {
            mesh_version,
            skippy_abi,
            artifacts,
        },
        options.bundle_dirs,
    ))
}

pub(crate) async fn download_release_manifest(url: &str) -> Result<NativeRuntimeReleaseManifest> {
    let diagnostic_url = url_without_query(url);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("build native runtime manifest HTTP client")?;
    let bytes = client
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| format!("download native runtime release manifest {diagnostic_url}"))?
        .error_for_status()
        .map_err(reqwest::Error::without_url)
        .with_context(|| {
            format!("native runtime release manifest request failed for {diagnostic_url}")
        })?
        .bytes()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| format!("read native runtime release manifest {diagnostic_url}"))?;
    let checksum_url = release_manifest_checksum_url(url);
    let diagnostic_checksum_url = url_without_query(&checksum_url);
    let checksum = client
        .get(&checksum_url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| {
            format!("download native runtime manifest checksum {diagnostic_checksum_url}")
        })?
        .error_for_status()
        .map_err(reqwest::Error::without_url)
        .with_context(|| {
            format!("native runtime manifest checksum request failed for {diagnostic_checksum_url}")
        })?
        .text()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| {
            format!("read native runtime manifest checksum {diagnostic_checksum_url}")
        })?;
    verify_release_manifest_checksum(&bytes, &checksum)
        .with_context(|| format!("verify native runtime release manifest {diagnostic_url}"))?;
    let text = std::str::from_utf8(&bytes)
        .with_context(|| format!("decode native runtime release manifest {diagnostic_url}"))?;
    NativeRuntimeReleaseManifest::from_json_str(text)
        .with_context(|| format!("parse native runtime release manifest {diagnostic_url}"))
}

pub(crate) fn verify_release_manifest_checksum(
    manifest_bytes: &[u8],
    checksum_text: &str,
) -> Result<()> {
    let expected = normalize_sha256(checksum_text)?;
    let actual = hex::encode(sha2::Sha256::digest(manifest_bytes));
    if actual != expected {
        bail!("native runtime manifest checksum mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

pub(crate) fn release_manifest_checksum_url(url: &str) -> String {
    match url.split_once('?') {
        Some((base, query)) => format!("{base}.sha256?{query}"),
        None => format!("{url}.sha256"),
    }
}

/// Strips the query string and redacts any userinfo (`user:pass@`) from a
/// URL before it is surfaced in error context or progress events. Mirrors
/// `redact_url_userinfo` in `mesh-llm-host-runtime::logging::policy`; kept
/// local because this crate does not otherwise depend on host-runtime.
pub(crate) fn url_without_query(url: &str) -> String {
    let without_query = url.split_once('?').map_or(url, |(base, _)| base);
    redact_url_userinfo(without_query)
}

fn redact_url_userinfo(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let authority_start = scheme_end + 3;
    let authority_end = url[authority_start..]
        .find(['/', '#'])
        .map_or(url.len(), |offset| authority_start + offset);
    let authority = &url[authority_start..authority_end];
    let Some(user_info_end) = authority.rfind('@') else {
        return url.to_string();
    };
    format!(
        "{}[REDACTED]@{}{}",
        &url[..authority_start],
        &authority[user_info_end + 1..],
        &url[authority_end..]
    )
}

pub(crate) fn manifest_url(options: &NativeRuntimeManifestOptions) -> Option<String> {
    options
        .manifest_url
        .clone()
        .or_else(|| {
            std::env::var(NATIVE_RUNTIME_MANIFEST_URL_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .or_else(|| {
            (options.allow_default_manifest_url && options.bundle_dirs.is_empty())
                .then(|| request_default_manifest_url(&options.mesh_version))
        })
}

pub(crate) fn append_bundle_artifacts(
    artifacts: &mut Vec<NativeRuntimeArtifact>,
    mesh_version: &mut String,
    skippy_abi: &mut String,
    bundle_dirs: &[PathBuf],
) -> Result<()> {
    for dir in bundle_dirs {
        let manifest = NativeRuntimeManifest::read_from_dir(dir)
            .with_context(|| format!("read bundled native runtime {}", dir.display()))?;
        if let Some(version) = &manifest.runtime.mesh_version {
            *mesh_version = version.clone();
        }
        *skippy_abi = manifest.runtime.skippy_abi.clone();
        artifacts.push(manifest.runtime);
    }
    Ok(())
}

pub(crate) fn normalize_sha256(value: &str) -> Result<String> {
    let trimmed = value.trim().strip_prefix("sha256:").unwrap_or(value.trim());
    let digest = trimmed
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if digest.len() == 64 && digest.chars().all(|ch| ch.is_ascii_hexdigit()) {
        Ok(digest)
    } else {
        bail!("native runtime manifest contains invalid sha256: {value}");
    }
}
