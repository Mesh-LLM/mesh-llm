//! Operator-owned local service bindings; payloads have no domain semantics.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PluginOperationBinding {
    pub name: String,
    /// Provider plugin name. Empty when `capability` names it instead: the
    /// host then resolves the single running plugin serving that capability.
    #[serde(default)]
    pub provider: String,
    /// Capability the provider must serve (for example `wallet.v1`), used
    /// when `provider` is empty. Comes from an installed package's manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
    /// Setup takes `{"directory": <absolute base>}` from the caller and the
    /// host dispatches `<base>/<provider>` (owner-only) instead of
    /// `setup_input_json`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub setup_directory: bool,
    /// Opaque profile identifier. One provider process owns one profile.
    pub profile: String,
    pub setup_operation: String,
    /// Fixed operator payload, never replaced with caller input.
    pub setup_input_json: String,
    pub operations: Vec<String>,
}

impl PluginOperationBinding {
    pub fn validate(&self, caller: &str) -> Result<(), String> {
        match &self.capability {
            Some(capability) if capability.trim().is_empty() || capability.len() > 256 => {
                return Err("binding capability must contain 1–256 bytes".into());
            }
            Some(_) if !self.provider.is_empty() => {
                return Err("a binding names either a provider or a capability, not both".into());
            }
            None if self.provider.trim().is_empty() => {
                return Err("binding identifiers must contain 1–256 bytes".into());
            }
            _ => {}
        }
        for value in [&self.name, &self.profile, &self.setup_operation] {
            if value.trim().is_empty() || value.len() > 256 {
                return Err("binding identifiers must contain 1–256 bytes".into());
            }
        }
        if self.provider.len() > 256 {
            return Err("binding identifiers must contain 1–256 bytes".into());
        }
        if self.provider == caller {
            return Err("self bindings are not permitted".into());
        }
        if self.setup_input_json.len() > 65_536
            || serde_json::from_str::<serde_json::Value>(&self.setup_input_json).is_err()
        {
            return Err("setup_input_json must be valid JSON no larger than 65536 bytes".into());
        }
        if self.operations.is_empty() || self.operations.len() > 64 {
            return Err("bindings require 1–64 allowed operations".into());
        }
        let mut names = std::collections::BTreeSet::new();
        for operation in &self.operations {
            if operation.trim().is_empty()
                || operation.len() > 256
                || operation == &self.setup_operation
                || !names.insert(operation)
            {
                return Err("allowed operations must be unique, bounded and exclude setup".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grant() -> PluginOperationBinding {
        PluginOperationBinding {
            capability: None,
            setup_directory: false,
            name: "state".into(),
            provider: "store".into(),
            profile: "one".into(),
            setup_operation: "open".into(),
            setup_input_json: "{\"profile\":\"one\"}".into(),
            operations: vec!["read".into(), "write".into()],
        }
    }
    #[test]
    fn binding_excludes_self_setup_and_malformed_payload() {
        let mut binding = grant();
        assert!(binding.validate("client").is_ok());
        assert!(binding.validate("store").is_err());
        binding.operations.push("open".into());
        assert!(binding.validate("client").is_err());
        binding = grant();
        binding.setup_input_json = "broken".into();
        assert!(binding.validate("client").is_err());
    }
}
