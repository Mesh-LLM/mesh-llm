use serde_json::Value;
use std::net::{IpAddr, Ipv6Addr};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityFailure {
    Missing,
    Whitespace,
    Scheme,
    Authority,
    Userinfo,
    Github,
    Loopback,
    Parser,
    Path,
    Port,
}

impl std::fmt::Display for AuthorityFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}",
            match self {
                Self::Missing => "missing",
                Self::Whitespace => "whitespace",
                Self::Scheme => "scheme",
                Self::Authority => "authority",
                Self::Userinfo => "userinfo",
                Self::Github => "github",
                Self::Loopback => "loopback",
                Self::Parser => "parser",
                Self::Path => "path",
                Self::Port => "port",
            }
        )
    }
}

pub fn attest_endpoint(endpoint: &str) -> Result<(), AuthorityFailure> {
    if endpoint.is_empty() {
        return Err(AuthorityFailure::Missing);
    }
    if endpoint.chars().any(char::is_whitespace) {
        return Err(AuthorityFailure::Whitespace);
    }
    let lower = endpoint.to_ascii_lowercase();
    let remainder = lower
        .strip_prefix("http://")
        .ok_or(AuthorityFailure::Scheme)?;
    let boundary = remainder.find(['/', '?', '#']).unwrap_or(remainder.len());
    let (authority, suffix) = remainder.split_at(boundary);
    if authority.is_empty() {
        return Err(AuthorityFailure::Authority);
    }
    if authority.contains('@') {
        return Err(AuthorityFailure::Userinfo);
    }
    let (host, port) = split_authority(authority)?;
    if host == "actions.githubusercontent.com" || host.ends_with(".actions.githubusercontent.com") {
        return Err(AuthorityFailure::Github);
    }
    if host == "localhost" || loopback(host)? {
        return Err(AuthorityFailure::Loopback);
    }
    if !suffix.starts_with('/') {
        return Err(AuthorityFailure::Path);
    }
    if port.is_empty() || port.len() > 5 || !port.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(AuthorityFailure::Port);
    }
    match port.parse::<u16>() {
        Ok(1..=u16::MAX) => Ok(()),
        _ => Err(AuthorityFailure::Port),
    }
}

fn split_authority(authority: &str) -> Result<(&str, &str), AuthorityFailure> {
    if let Some(bracketed) = authority.strip_prefix('[') {
        let (host, suffix) = bracketed.split_once(']').ok_or(AuthorityFailure::Parser)?;
        host.parse::<Ipv6Addr>()
            .map_err(|_| AuthorityFailure::Parser)?;
        return Ok((
            host,
            suffix.strip_prefix(':').ok_or(AuthorityFailure::Port)?,
        ));
    }
    let (host, port) = authority.rsplit_once(':').ok_or(AuthorityFailure::Port)?;
    if host.is_empty() || host.contains(':') {
        return Err(AuthorityFailure::Authority);
    }
    Ok((host, port))
}

fn loopback(host: &str) -> Result<bool, AuthorityFailure> {
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(address)) => Ok(address.is_loopback()),
        Ok(IpAddr::V6(address)) => Ok(address.is_loopback()
            || address
                .to_ipv4_mapped()
                .is_some_and(|mapped| mapped.is_loopback())),
        Err(_) if host.contains(':') => Err(AuthorityFailure::Parser),
        Err(_) => Ok(false),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockerFailure {
    Malformed,
    Unreadable,
    Authentication,
    DepotAuthentication,
}

impl std::fmt::Display for DockerFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Malformed => "malformed",
            Self::Unreadable => "unreadable",
            Self::Authentication => "authentication",
            Self::DepotAuthentication => "depot-authentication",
        })
    }
}

pub fn audit_docker_json(raw: &str, depot_selected: bool) -> Result<(), DockerFailure> {
    let document: Value = serde_json::from_str(raw).map_err(|_| DockerFailure::Malformed)?;
    let object = document.as_object().ok_or(DockerFailure::Malformed)?;
    if depot_selected {
        return if ["auths", "credHelpers", "credsStore"]
            .iter()
            .any(|key| object.contains_key(*key))
        {
            Err(DockerFailure::Authentication)
        } else {
            Ok(())
        };
    }
    for key in ["auths", "credHelpers"] {
        if let Some(section) = object.get(key) {
            let section = section.as_object().ok_or(DockerFailure::Malformed)?;
            if section
                .keys()
                .any(|key| key.to_lowercase().contains("depot.dev"))
            {
                return Err(DockerFailure::DepotAuthentication);
            }
        }
    }
    if let Some(store) = object.get("credsStore") {
        let store = store.as_str().ok_or(DockerFailure::Malformed)?;
        if store.to_lowercase().contains("depot.dev") {
            return Err(DockerFailure::DepotAuthentication);
        }
    }
    Ok(())
}

pub fn audit_docker_sources(auth: &str, path: &Path, depot: bool) -> Result<(), DockerFailure> {
    if !auth.is_empty() {
        if depot {
            return Err(DockerFailure::Authentication);
        }
        audit_docker_json(auth, depot)?;
    }
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => {
            let raw = std::fs::read_to_string(path).map_err(|error| match error.kind() {
                std::io::ErrorKind::InvalidData => DockerFailure::Malformed,
                _ => DockerFailure::Unreadable,
            })?;
            audit_docker_json(&raw, depot)
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(DockerFailure::Unreadable),
    }
}
