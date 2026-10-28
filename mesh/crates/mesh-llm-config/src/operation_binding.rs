//! Operator-owned local service bindings; payloads have no domain semantics.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PluginOperationBinding {
    pub name: String,
    pub provider: String,
    /// Opaque profile identifier. One provider process owns one profile.
    pub profile: String,
    pub setup_operation: String,
    /// Fixed operator payload, never replaced with caller input.
    pub setup_input_json: String,
    pub operations: Vec<String>,
}

impl PluginOperationBinding {
    pub fn validate(&self, caller: &str) -> Result<(), String> {
        for value in [
            &self.name,
            &self.provider,
            &self.profile,
            &self.setup_operation,
        ] {
            if value.trim().is_empty() || value.len() > 256 {
                return Err("binding identifiers must contain 1–256 bytes".into());
            }
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
