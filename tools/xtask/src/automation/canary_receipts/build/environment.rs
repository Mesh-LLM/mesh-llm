use super::contract::Input;
use super::package::Identity;
use crate::process::Value;
use std::{collections::BTreeMap, ffi::OsString, path::Path};

pub(super) fn wrapper(
    input: &Input,
    previous: Option<(&Path, &Identity)>,
) -> BTreeMap<OsString, Value> {
    let mut environment = inherited(std::env::vars_os());
    let values = [
        ("CANARY_HARNESS_MODE", input.mode.text().to_owned()),
        ("CANARY_PASS_ID", input.pass_id.clone()),
        ("CANARY_CONTROLLER_SHA", input.controller_revision.clone()),
        ("CANARY_MESH_SOURCE", input.mesh_source.clone()),
        ("UPSTREAM_SHA_INPUT", input.upstream_revision.clone()),
        ("GITHUB_RUN_ID", input.run_id.clone()),
        ("GITHUB_RUN_ATTEMPT", input.run_attempt.clone()),
        (
            "CANARY_AGENT_TIMEOUT_SECONDS",
            input.agent_timeout_seconds.to_string(),
        ),
        (
            "CANARY_VERIFICATION_TIMEOUT_SECONDS",
            input.verification_timeout_seconds.to_string(),
        ),
    ];
    for (key, value) in values {
        environment.insert(key.into(), Value::Public(value.into()));
    }
    environment.insert(
        "CANARY_SOURCE_ROOT".into(),
        Value::Public(input.source_root.clone().into()),
    );
    environment.insert(
        "CANARY_EXPORT_DIR".into(),
        Value::Public(input.export.clone().into()),
    );
    for key in [
        "CANARY_INPUT_BUNDLE",
        "CANARY_CANDIDATE_BRANCH",
        "CANARY_CANDIDATE_SHA",
        "CANARY_PREVIOUS_PACKAGE",
        "CANARY_PREVIOUS_IDENTITY",
        "CANARY_VERIFIED_WORKLOAD_PRODUCER",
    ] {
        environment.remove(std::ffi::OsStr::new(key));
    }
    if let Some((package, identity)) = previous {
        environment.insert(
            "CANARY_INPUT_BUNDLE".into(),
            Value::Public(package.join("candidate.bundle").into()),
        );
        environment.insert(
            "CANARY_CANDIDATE_BRANCH".into(),
            Value::Public(identity.branch.clone().into()),
        );
        environment.insert(
            "CANARY_CANDIDATE_SHA".into(),
            Value::Public(identity.candidate.clone().into()),
        );
    }
    environment
}

fn inherited(values: impl IntoIterator<Item = (OsString, OsString)>) -> BTreeMap<OsString, Value> {
    values
        .into_iter()
        .filter(|(key, _)| {
            !["GH_TOKEN", "GITHUB_TOKEN", "CANARY_REPAIR_TOKEN"]
                .iter()
                .any(|name| key == name)
        })
        .map(|(key, value)| {
            let name = key.to_string_lossy();
            let secret = !value.is_empty()
                && (name.ends_with("_TOKEN")
                    || name.ends_with("_KEY")
                    || name.ends_with("_SECRET")
                    || name.ends_with("_PASSWORD"));
            (
                key,
                if secret {
                    Value::Secret(value)
                } else {
                    Value::Public(value)
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inherited_runner_auth_is_redacted_and_publication_credentials_are_removed() {
        let environment = inherited([
            ("GH_TOKEN".into(), "github-secret".into()),
            ("CANARY_REPAIR_TOKEN".into(), "publisher-secret".into()),
            ("GITHUB_TOKEN".into(), "github-secret".into()),
            ("ZAI_API_KEY".into(), "agent-secret".into()),
            ("OPTIONAL_API_KEY".into(), "".into()),
            ("HF_CACHE".into(), "/read-only-cache".into()),
        ]);
        for key in ["GH_TOKEN", "GITHUB_TOKEN", "CANARY_REPAIR_TOKEN"] {
            assert!(!environment.contains_key(std::ffi::OsStr::new(key)));
        }
        assert!(matches!(
            environment.get(std::ffi::OsStr::new("ZAI_API_KEY")),
            Some(Value::Secret(_))
        ));
        assert!(matches!(
            environment.get(std::ffi::OsStr::new("HF_CACHE")),
            Some(Value::Public(_))
        ));
        assert!(matches!(
            environment.get(std::ffi::OsStr::new("OPTIONAL_API_KEY")),
            Some(Value::Public(_))
        ));
    }
}
