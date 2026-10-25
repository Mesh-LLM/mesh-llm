//! Anonymous history reads and explicitly credentialed publication share only
//! the retained process boundary. Auth is never inferred from a user's HF cache.
use crate::{
    automation::private_state::PrivateState,
    command::DynResult,
    process::{self, Value},
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    time::Duration,
};
pub(super) fn repository(repo: &str) -> DynResult<()> {
    let parts = repo.split('/').collect::<Vec<_>>();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || [".", ".."].contains(part)
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        })
    {
        return Err("dataset repository must be OWNER/NAME".into());
    }
    Ok(())
}
pub(super) fn environment(
    state: &PrivateState,
    token: Option<OsString>,
) -> BTreeMap<OsString, Value> {
    let mut environment = BTreeMap::new();
    for key in [
        "PATH",
        "SystemRoot",
        "WINDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ] {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(key.into(), Value::Public(value));
        }
    }
    for (key, path) in [
        ("HOME", state.root().to_path_buf()),
        ("USERPROFILE", state.root().to_path_buf()),
        ("HF_HOME", state.root().join("hf")),
        ("XDG_CACHE_HOME", state.root().join("cache")),
        ("XDG_CONFIG_HOME", state.root().join("config")),
    ] {
        environment.insert(key.into(), Value::Public(path.into_os_string()));
    }
    environment.insert("HF_HUB_OFFLINE".into(), Value::Public("0".into()));
    environment.insert(
        "HF_HUB_DISABLE_IMPLICIT_TOKEN".into(),
        Value::Public(if token.is_some() { "0" } else { "1" }.into()),
    );
    if let Some(token) = token {
        environment.insert("HF_TOKEN".into(), Value::Secret(token));
    }
    environment
}
pub(super) fn execute(
    executable: &Path,
    arguments: Vec<String>,
    state: &PrivateState,
    token: Option<OsString>,
    timeout: Duration,
    cancellation: &process::Cancellation,
) -> DynResult<process::ProcessReport> {
    let spec = process::ProcessSpec {
        executable: super::executable_resolution::executable(executable)?,
        arguments: arguments
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect(),
        cwd: state.root().to_path_buf(),
        environment: environment(state, token),
    };
    Ok(process::supervise(
        &spec,
        &process::Limits {
            execution: timeout,
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        cancellation,
        Default::default(),
    )?)
}
pub(super) fn terminal(report: &process::ProcessReport) -> bool {
    report.outcome == process::Outcome::Exited
        && report.status.is_some()
        && report.failure.is_none()
        && report.cleanup.complete
        && !report.cleanup.forced
        && !report.cleanup.graceful_signal_failed
        && report.cleanup.failure.is_none()
}
pub(super) fn lookup_url(repo: &str) -> DynResult<String> {
    repository(repo)?;
    Ok(format!("https://huggingface.co/api/datasets/{repo}"))
}
pub(super) fn lookup(
    curl: &Path,
    url: &str,
    state: &PrivateState,
    cancellation: &process::Cancellation,
) -> DynResult<u16> {
    let null = if cfg!(windows) { "NUL" } else { "/dev/null" };
    let arguments = vec![
        "--disable".into(),
        "--silent".into(),
        "--show-error".into(),
        "--location".into(),
        "--max-redirs".into(),
        "3".into(),
        "--proto-redir".into(),
        "=https".into(),
        "--max-time".into(),
        "30".into(),
        "--max-filesize".into(),
        "1048576".into(),
        "--output".into(),
        null.into(),
        "--write-out".into(),
        "%{http_code}".into(),
        url.into(),
    ];
    let report = execute(
        curl,
        arguments,
        state,
        None,
        Duration::from_secs(35),
        cancellation,
    )?;
    if !report.success() || !terminal(&report) || report.stdout.truncated {
        return Err("anonymous dataset lookup failed before a verified HTTP response".into());
    }
    let status = std::str::from_utf8(&report.stdout.bytes_retained)?.trim();
    if status.len() != 3 || !status.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("anonymous dataset lookup returned malformed HTTP status".into());
    }
    let status = status.parse::<u16>()?;
    if !(100..=599).contains(&status) {
        return Err("anonymous dataset lookup returned no HTTP status".into());
    }
    Ok(status)
}
pub(super) fn tool(value: Option<&str>, name: &str) -> DynResult<PathBuf> {
    match value {
        Some(value) => Ok(super::executable_resolution::executable(Path::new(value))?),
        None => super::executable_resolution::tool(name),
    }
}
pub(super) fn seconds(value: &str) -> DynResult<Duration> {
    let seconds = value.parse::<u64>()?;
    if !(1..=600).contains(&seconds) {
        return Err("history Hub timeout must be in 1..=600 seconds".into());
    }
    Ok(Duration::from_secs(seconds))
}
