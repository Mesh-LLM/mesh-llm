//! Mesh policy for the shared advisory source-digest cache.
pub(crate) use skippy_api::hash_cache::{SidecarDigestCache, file_ctime_nanos, file_mtime_nanos};
use std::path::PathBuf;
const CACHE_DIR_ENV: &str = "MESH_LLM_HASH_CACHE_DIR";
/// Resolve the default cache location.
///
/// Precedence:
/// 1. `MESH_LLM_HASH_CACHE_DIR` environment variable
/// 2. `~/.mesh-llm/cache/hashes`
/// 3. `None` (caching disabled, digests are always recomputed)
pub(crate) fn open_default() -> Option<SidecarDigestCache> {
    if let Some(dir) = std::env::var_os(CACHE_DIR_ENV) {
        return Some(SidecarDigestCache::open_in(PathBuf::from(dir)));
    }
    Some(SidecarDigestCache::open_in(default_dir()?))
}

/// The default location, `~/.mesh-llm/cache/hashes`.
#[cfg(not(test))]
fn default_dir() -> Option<PathBuf> {
    Some(
        dirs::home_dir()?
            .join(".mesh-llm")
            .join("cache")
            .join("hashes"),
    )
}

/// Under test there is no default location. Package verification runs
/// through `open_default`, so its tests used to leave entries keyed to
/// already deleted temp fixtures in the developer's real cache on every
/// run. A test that exercises caching opens an explicit directory through
/// `open_in`, and `MESH_LLM_HASH_CACHE_DIR` above still redirects the
/// default for a test that wants it.
#[cfg(test)]
fn default_dir() -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_default_cache_never_reaches_the_real_home_under_test() {
        // Measured on Windows 11 before this guard: one full run of the crate's
        // tests wrote six entries into `~/.mesh-llm/cache/hashes`, every one
        // keyed to a temp fixture that no longer existed. An explicit
        // directory in the environment is the one override that still applies.
        if std::env::var_os(CACHE_DIR_ENV).is_none() {
            assert!(open_default().is_none());
        }
    }
}
