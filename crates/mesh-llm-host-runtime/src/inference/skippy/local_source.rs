use super::StageLoadRequest;
use anyhow::{Context, Result};
pub(crate) use skippy_api::source_registry::{
    into_content_addressed_identity, is_content_addressed_gguf_ref,
    verify_registered_content_source,
};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
static LOCAL_SOURCE_POLICIES: OnceLock<Mutex<HashMap<String, HashMap<String, bool>>>> =
    OnceLock::new();
fn local_source_policy_registry() -> &'static Mutex<HashMap<String, HashMap<String, bool>>> {
    LOCAL_SOURCE_POLICIES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record the current local source policy for a logical model profile.
///
/// Each exact profile remains independent. Only local lifecycle code calls
/// this function, so replacing a prior value lets a failed start or config
/// reload move from strict back to fallback without an immortal process-wide
/// policy bit. Profile-unaware inbound requests still fail closed whenever any
/// currently recorded profile for the model is strict.
pub(crate) fn register_local_source_policy(
    model_id: &str,
    runtime_profile: &str,
    local_source_required: bool,
) {
    if model_id.is_empty() {
        return;
    }
    if let Ok(mut models) = local_source_policy_registry().lock() {
        let profiles = models.entry(model_id.to_string()).or_default();
        profiles.insert(runtime_profile.to_string(), local_source_required);
    }
}

/// Forget a policy profile after its last local runtime has stopped.
///
/// `source_policy` contributes to the derived runtime profile, so retaining a
/// stopped strict profile would make later profile-unaware fallback requests
/// fail closed forever. Callers must keep the entry registered while another
/// runtime with the same model/profile pair is still active.
pub(crate) fn unregister_local_source_policy(model_id: &str, runtime_profile: &str) {
    let Ok(mut models) = local_source_policy_registry().lock() else {
        return;
    };
    let remove_model = if let Some(profiles) = models.get_mut(model_id) {
        profiles.remove(runtime_profile);
        profiles.is_empty()
    } else {
        false
    };
    if remove_model {
        models.remove(model_id);
    }
}

pub(crate) fn local_source_required_for_model(
    model_id: &str,
    runtime_profile: Option<&str>,
) -> bool {
    let Ok(models) = local_source_policy_registry().lock() else {
        return true;
    };
    let Some(profiles) = models.get(model_id) else {
        return false;
    };
    if let Some(runtime_profile) = runtime_profile
        && let Some(required) = profiles.get(runtime_profile)
    {
        return *required;
    }
    // A legacy or unknown profile cannot safely select between local policies.
    // If any known profile is strict, fail closed until the sender provides an
    // exact profile that has been registered as fallback.
    profiles.values().any(|required| *required)
}

pub(crate) fn effective_local_source_required(
    model_id: &str,
    runtime_profile: Option<&str>,
    requested: bool,
) -> bool {
    requested || local_source_required_for_model(model_id, runtime_profile)
}

/// Apply the effective local source policy and resolve a verified worker-local
/// path without consulting catalogs, Hugging Face, or peer artifact transfer.
pub(crate) fn apply_verified_local_source(load: &mut StageLoadRequest) -> Result<bool> {
    let local_source_required = effective_local_source_required(
        &load.model_id,
        load.runtime_profile.as_deref(),
        load.local_source_required || is_content_addressed_gguf_ref(&load.package_ref),
    );
    if !local_source_required {
        return Ok(false);
    }
    load.local_source_required = true;
    anyhow::ensure!(
        load.load_mode == skippy_protocol::LoadMode::RuntimeSlice
            && is_content_addressed_gguf_ref(&load.package_ref),
        "local-required stage source must be a content-addressed RuntimeSlice GGUF"
    );
    let expected_source_sha256 = load
        .source_model_sha256
        .as_deref()
        .context("local-required stage source is missing expected SHA-256")?;
    let identity = verify_registered_content_source(
        &load.model_id,
        &load.package_ref,
        &load.manifest_sha256,
        expected_source_sha256,
    )?;
    load.model_path = Some(
        identity
            .source_model_path
            .to_str()
            .context("verified local GGUF path is not valid UTF-8")?
            .to_string(),
    );
    load.source_model_bytes = Some(identity.source_model_bytes);
    load.source_model_sha256 = Some(identity.source_model_sha256);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_required_policy_is_profile_scoped_and_strengthens_legacy_requests() {
        let model_id = format!("strict-model-{}", std::process::id());
        assert!(!effective_local_source_required(
            &model_id,
            Some("strict"),
            false
        ));
        assert!(effective_local_source_required(
            &model_id,
            Some("strict"),
            true
        ));
        register_local_source_policy(&model_id, "strict", true);
        register_local_source_policy(&model_id, "fallback", false);
        assert!(effective_local_source_required(
            &model_id,
            Some("strict"),
            false
        ));
        assert!(!effective_local_source_required(
            &model_id,
            Some("fallback"),
            false
        ));
        assert!(effective_local_source_required(
            &model_id,
            Some("unknown"),
            false
        ));
        assert!(effective_local_source_required(&model_id, None, false));

        register_local_source_policy(&model_id, "", false);
        assert!(!effective_local_source_required(&model_id, Some(""), false));
        assert!(effective_local_source_required(&model_id, None, false));

        register_local_source_policy(&model_id, "strict", false);
        assert!(!effective_local_source_required(
            &model_id,
            Some("strict"),
            false
        ));
        assert!(!effective_local_source_required(&model_id, None, false));
    }

    #[test]
    fn stopped_strict_profile_does_not_poison_later_fallback_loads() {
        let model_id = format!("reloaded-policy-model-{}", std::process::id());
        register_local_source_policy(&model_id, "strict", true);

        unregister_local_source_policy(&model_id, "strict");
        register_local_source_policy(&model_id, "fallback", false);

        assert!(!effective_local_source_required(&model_id, None, false));
        assert!(!effective_local_source_required(
            &model_id,
            Some("fallback"),
            false
        ));
    }
}
