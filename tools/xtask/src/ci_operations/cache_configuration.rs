//! Existing runner cache admission; this owner never provisions models or credentials.
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub(super) struct Configuration {
    pub(super) hub: PathBuf,
    pub(super) values: Vec<(&'static str, String)>,
}

fn existing<'a>(env: &'a BTreeMap<String, String>, name: &str) -> Option<&'a str> {
    env.get(name)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

fn expand_home(path: &str, env: &BTreeMap<String, String>) -> Result<PathBuf, String> {
    if path == "~" || path.starts_with("~/") {
        return Ok(home(env)?.join(path.strip_prefix("~/").unwrap_or("")));
    }
    if let Some(path) = path.strip_prefix('~') {
        let (name, suffix) = path.split_once('/').unwrap_or((path, ""));
        return Ok(super::cache_home::account_home(Some(name))?.join(suffix));
    }
    Ok(PathBuf::from(path))
}

fn home(env: &BTreeMap<String, String>) -> Result<PathBuf, String> {
    match existing(env, "HOME") {
        Some(home) => Ok(home.into()),
        None => super::cache_home::account_home(None),
    }
}

pub(super) fn configuration(env: &BTreeMap<String, String>) -> Result<Configuration, String> {
    let default = match existing(env, "XDG_CACHE_HOME") {
        Some(path) => PathBuf::from(path),
        None => home(env)?.join(".cache"),
    }
    .join("huggingface");
    let home = match existing(env, "HF_HOME").or_else(|| existing(env, "HF_CACHE")) {
        Some(path) => expand_home(path, env)?,
        None => default,
    };
    let hub = match existing(env, "HF_HUB_CACHE") {
        Some(path) => expand_home(path, env)?,
        None => home.join("hub"),
    };
    let unavailable = || {
        format!(
            "configured Hugging Face hub cache is unavailable: {}; check the runner mount/configuration",
            hub.display()
        )
    };
    let resolved = std::fs::canonicalize(&hub).map_err(|_| unavailable())?;
    let expected = std::fs::canonicalize(home.join("hub")).map_err(|_| unavailable())?;
    if resolved != expected {
        return Err("HF_HUB_CACHE must resolve to HF_HOME/hub for family certification".into());
    }
    if !hub.is_dir() || !accessible(&hub) {
        return Err(unavailable());
    }
    let mut values = vec![
        ("HF_CACHE", path_text(&home)?),
        ("HF_HOME", path_text(&home)?),
        ("HF_HUB_CACHE", path_text(&hub)?),
        ("HF_HUB_OFFLINE", "1".into()),
    ];
    for name in ["HF_TOKEN", "HF_TOKEN_PATH"] {
        if let Some(value) = existing(env, name) {
            values.push((name, value.to_owned()));
        }
    }
    for (name, value) in &values {
        if value.contains(['\r', '\n', '\0']) {
            return Err(format!("invalid multiline value for {name}"));
        }
    }
    Ok(Configuration { hub, values })
}

fn path_text(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| "cache path must be UTF-8".into())
}

#[cfg(unix)]
fn accessible(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: the owned CString remains valid during this read-only access probe.
    unsafe { libc::access(path.as_ptr(), libc::R_OK | libc::X_OK) == 0 }
}

#[cfg(not(unix))]
fn accessible(path: &Path) -> bool {
    std::fs::read_dir(path).is_ok()
}
