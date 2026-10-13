use super::authority_audit::{attest_endpoint, audit_docker_sources};
use crate::repository::check_report::CheckReport;
use std::path::Path;

const USAGE: &str = "usage: ci-ops authority-audit endpoint ENV_NAME | docker CONFIG_PATH --depot-selected {true|false}\n";

pub(super) fn run(args: &[String]) -> CheckReport {
    let result = match args {
        [help] if help == "--help" => return CheckReport::success(USAGE.into()),
        [mode, name] if mode == "endpoint" && valid_name(name) => match std::env::var(name) {
            Ok(value) => attest_endpoint(&value).map_err(|failure| format!("{name}: {failure}")),
            Err(_) => Err(format!("{name}: missing")),
        },
        [mode, path, flag, selected]
            if mode == "docker"
                && flag == "--depot-selected"
                && matches!(selected.as_str(), "true" | "false") =>
        {
            match std::env::var("DOCKER_AUTH_CONFIG") {
                Ok(auth) => audit_docker_sources(&auth, Path::new(path), selected == "true")
                    .map_err(|failure| format!("DOCKER_AUTH_CONFIG/config.json: {failure}")),
                Err(std::env::VarError::NotPresent) => {
                    audit_docker_sources("", Path::new(path), selected == "true")
                        .map_err(|failure| format!("DOCKER_AUTH_CONFIG/config.json: {failure}"))
                }
                Err(std::env::VarError::NotUnicode(_)) => {
                    Err("DOCKER_AUTH_CONFIG: malformed".into())
                }
            }
        }
        _ => return CheckReport::failure(String::new(), USAGE.into()),
    };
    match result {
        Ok(()) => CheckReport::success(String::new()),
        Err(message) => CheckReport::failure(String::new(), format!("{message}\n")),
    }
}

fn valid_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}
