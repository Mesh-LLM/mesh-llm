//! PR runner authority checks performed before untrusted source is admitted.
use super::authority_audit::audit_docker_sources;
use crate::repository::check_report::CheckReport;
use std::path::PathBuf;

const FORBIDDEN: &[&str] = &[
    "DEPOT_CACHE_TOKEN",
    "DEPOT_TOKEN",
    "DEPOT_REGISTRY_PULL_TOKEN",
    "DEPOT_REGISTRY_TOKEN",
    "DEPOT_CACHE_URL",
    "DEPOT_CACHE_API_URL",
    "DEPOT_REGISTRY_URL",
    "DEPOT_REGISTRY_HOST",
    "SCCACHE_WEBDAV_ENDPOINT",
    "SCCACHE_WEBDAV_TOKEN",
    "SCCACHE_WEBDAV_USERNAME",
    "SCCACHE_WEBDAV_PASSWORD",
    "SCCACHE_BUCKET",
    "SCCACHE_ENDPOINT",
    "TURBO_TOKEN",
    "TURBO_API",
    "TURBO_TEAM",
    "GOCACHEPROG",
    "REGISTRY_TOKEN",
    "REGISTRY_USERNAME",
    "REGISTRY_PASSWORD",
    "REGISTRY_AUTH_TOKEN",
    "NPM_TOKEN",
    "NODE_AUTH_TOKEN",
    "CARGO_REGISTRIES_CRATES_IO_TOKEN",
];

struct Policy {
    depot: bool,
    native_cache: bool,
    remote_cache: bool,
}

fn flag(name: &str, value: Option<String>) -> Result<bool, String> {
    match value.as_deref() {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        _ => Err(format!("{name}: requires true or false")),
    }
}

impl Policy {
    fn read(get: &impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let policy = Self {
            depot: flag("INPUT_DEPOT_SELECTED", get("INPUT_DEPOT_SELECTED"))?,
            native_cache: flag(
                "INPUT_ALLOW_NATIVE_GITHUB_CACHE",
                get("INPUT_ALLOW_NATIVE_GITHUB_CACHE"),
            )?,
            remote_cache: flag(
                "INPUT_ALLOW_DEPOT_REMOTE_CACHE",
                get("INPUT_ALLOW_DEPOT_REMOTE_CACHE"),
            )?,
        };
        if policy.depot && policy.remote_cache {
            return Err(
                "INPUT_ALLOW_DEPOT_REMOTE_CACHE: Depot PR remote cache must be disabled".into(),
            );
        }
        Ok(policy)
    }

    fn endpoint(&self, value: &str) -> Result<(), &'static str> {
        if value.is_empty() {
            return Ok(());
        }
        if value.chars().any(char::is_whitespace) {
            return Err("whitespace");
        }
        let uri = url::Url::parse(value).map_err(|_| "malformed")?;
        if !matches!(uri.scheme(), "http" | "https") || uri.host().is_none() {
            return Err("malformed");
        }
        let remainder = value.split_once("://").ok_or("malformed")?.1;
        let boundary = remainder.find(['/', '?', '#']).unwrap_or(remainder.len());
        let authority = &remainder[..boundary];
        if authority.contains('@') || !uri.username().is_empty() || uri.password().is_some() {
            return Err("userinfo");
        }
        let host = uri.host_str().ok_or("malformed")?.to_ascii_lowercase();
        let depot_host = host == "depot.dev" || host.ends_with(".depot.dev");
        if depot_host && !(self.depot && self.native_cache) {
            return Err("unapproved Depot endpoint");
        }
        if !self.depot || self.native_cache {
            return Ok(());
        }
        let github = uri.scheme() == "https"
            && (host == "actions.githubusercontent.com"
                || host.ends_with(".actions.githubusercontent.com"))
            && host.split('.').all(|label| {
                !label.is_empty()
                    && label
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            });
        if github {
            return Ok(());
        }
        // Explicit loopback proxies are allowed, including default-valued ports.
        // Inspect the authority before Url removes a default :80 or :443.
        let explicit_port = authority
            .rsplit_once(':')
            .and_then(|(_, port)| port.parse::<u16>().ok())
            .is_some_and(|port| port > 0);
        let explicit_path = remainder[boundary..].starts_with('/');
        let loopback = match uri.host().ok_or("malformed")? {
            url::Host::Domain(name) => name == "localhost",
            url::Host::Ipv4(address) => address.is_loopback(),
            url::Host::Ipv6(address) => {
                address.is_loopback()
                    || address
                        .to_ipv4_mapped()
                        .is_some_and(|mapped| mapped.is_loopback())
            }
        };
        if loopback && explicit_port && explicit_path {
            Ok(())
        } else {
            Err("endpoint must be GitHub-owned HTTPS or an explicit loopback proxy")
        }
    }
}

fn check(get: &impl Fn(&str) -> Option<String>) -> Result<(), String> {
    let policy = Policy::read(get)?;
    let event = get("INPUT_ORIGINAL_EVENT_NAME")
        .filter(|event| !event.is_empty())
        .or_else(|| get("GITHUB_EVENT_NAME"))
        .unwrap_or_default();
    if !matches!(event.as_str(), "pull_request" | "pull_request_target") {
        return Ok(());
    }
    for name in FORBIDDEN {
        if get(name).is_some_and(|value| !value.is_empty()) {
            return Err(format!("{name}: forbidden PR runner authority"));
        }
    }
    for name in [
        "ACTIONS_CACHE_URL",
        "ACTIONS_RESULTS_URL",
        "ACTIONS_RUNTIME_URL",
    ] {
        if let Some(value) = get(name) {
            policy
                .endpoint(&value)
                .map_err(|reason| format!("{name}: {reason}"))?;
        }
    }
    let directory = get("DOCKER_CONFIG")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(get("HOME").unwrap_or_else(|| "/".into())).join(".docker")
        });
    let auth = get("DOCKER_AUTH_CONFIG").unwrap_or_default();
    audit_docker_sources(&auth, &directory.join("config.json"), policy.depot)
        .map_err(|reason| format!("DOCKER_AUTH_CONFIG/config.json: {reason}"))
}

pub(super) fn run(args: &[String]) -> CheckReport {
    if args == ["--help"] {
        return CheckReport::success("usage: cargo xtool ci-ops pr-authority-audit\nValidates INPUT_DEPOT_SELECTED, INPUT_ALLOW_NATIVE_GITHUB_CACHE, INPUT_ALLOW_DEPOT_REMOTE_CACHE, original event, Actions endpoints, and Docker authentication. Success is silent.\n".into());
    }
    if !args.is_empty() {
        return CheckReport::failure(String::new(), "usage: ci-ops pr-authority-audit\n".into());
    }
    // Values are read only in memory. Diagnostics contain field names and reasons.
    // Non-Unicode values cannot be silently treated as absent authority.
    let mut invalid = None;
    let get = |name: &str| std::env::var(name).ok();
    for name in FORBIDDEN.iter().copied().chain([
        "INPUT_ORIGINAL_EVENT_NAME",
        "GITHUB_EVENT_NAME",
        "INPUT_DEPOT_SELECTED",
        "INPUT_ALLOW_NATIVE_GITHUB_CACHE",
        "INPUT_ALLOW_DEPOT_REMOTE_CACHE",
        "ACTIONS_CACHE_URL",
        "ACTIONS_RESULTS_URL",
        "ACTIONS_RUNTIME_URL",
        "DOCKER_CONFIG",
        "HOME",
        "DOCKER_AUTH_CONFIG",
    ]) {
        if matches!(std::env::var(name), Err(std::env::VarError::NotUnicode(_))) {
            invalid = Some(format!("{name}: invalid text"));
            break;
        }
    }
    match invalid.map_or_else(|| check(&get), Err) {
        Ok(()) => CheckReport::success(String::new()),
        Err(message) => CheckReport::failure(String::new(), format!("{message}\n")),
    }
}

#[cfg(test)]
#[path = "pr_authority_tests.rs"]
mod tests;
