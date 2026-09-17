/// Default Hugging Face Hub endpoint
pub(crate) const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

/// Default revision (branch)
pub(crate) const DEFAULT_REVISION: &str = "main";

pub(crate) const HF_ENDPOINT: &str = "HF_ENDPOINT";
pub(crate) const HF_TOKEN: &str = "HF_TOKEN";
pub(crate) const HF_TOKEN_PATH: &str = "HF_TOKEN_PATH";
pub(crate) const HF_HOME: &str = "HF_HOME";
pub(crate) const HF_HUB_CACHE: &str = "HF_HUB_CACHE";
pub(crate) const HUGGINGFACE_HUB_CACHE: &str = "HUGGINGFACE_HUB_CACHE";
pub(crate) const XDG_CACHE_HOME: &str = "XDG_CACHE_HOME";
pub(crate) const HF_HUB_DISABLE_IMPLICIT_TOKEN: &str = "HF_HUB_DISABLE_IMPLICIT_TOKEN";
pub(crate) const HF_HUB_USER_AGENT_ORIGIN: &str = "HF_HUB_USER_AGENT_ORIGIN";

/// Token filename within HF_HOME
pub(crate) const TOKEN_FILENAME: &str = "token";

pub(crate) const HEADER_X_XET_HASH: &str = "x-xet-hash";

pub(crate) const CACHE_LOCK_TIMEOUT_SECS: u64 = 10;
pub(crate) const HEADER_X_REPO_COMMIT: &str = "x-repo-commit";
pub(crate) const HEADER_X_LINKED_ETAG: &str = "x-linked-etag";
pub(crate) const HEADER_X_LINKED_SIZE: &str = "x-linked-size";

pub(crate) fn dirs_or_home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
}

/// Resolve the Hugging Face home directory.
///
/// Order: `HF_HOME` env var, then `XDG_CACHE_HOME/huggingface`, then `~/.cache/huggingface`.
pub fn hf_home() -> std::path::PathBuf {
    if let Ok(path) = std::env::var(HF_HOME) {
        return std::path::PathBuf::from(path);
    }
    if let Ok(xdg) = std::env::var(XDG_CACHE_HOME) {
        return std::path::PathBuf::from(xdg).join("huggingface");
    }
    let home = dirs_or_home();
    std::path::PathBuf::from(format!("{home}/.cache/huggingface"))
}

/// Resolve the Hugging Face Hub cache directory.
///
/// Order: `HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`, then `<hf_home>/hub`.
pub fn resolve_cache_dir() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var(HF_HUB_CACHE) {
        return std::path::PathBuf::from(cache);
    }
    if let Ok(cache) = std::env::var(HUGGINGFACE_HUB_CACHE) {
        return std::path::PathBuf::from(cache);
    }
    hf_home().join("hub")
}
