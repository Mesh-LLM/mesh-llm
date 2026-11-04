use super::{Response, decode};
use serde::Deserialize;

pub(super) fn privacy(bytes: &[u8]) -> Result<(), String> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|_| "status malformed")?;
    fn leaked(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(fields) => fields
                .iter()
                .any(|(key, value)| key == "control_endpoint" || leaked(value)),
            serde_json::Value::Array(values) => values.iter().any(leaked),
            serde_json::Value::String(value) => {
                value.contains("mesh-llm-control/") || value.contains("control://")
            }
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
                false
            }
        }
    }
    if leaked(&value) {
        Err("mixed-version control data leaked".into())
    } else {
        Ok(())
    }
}

pub(super) fn scan(response: &Response) -> Result<(), String> {
    #[derive(Deserialize)]
    struct Scan {
        disposition: String,
        target_node_id: String,
        inventory: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        canonical_model_ref: String,
        metadata: std::collections::BTreeMap<String, serde_json::Value>,
    }
    let scan: Scan = decode(response)?;
    if response.status >= 400
        || !["executed", "coalesced"].contains(&scan.disposition.as_str())
        || scan.target_node_id.is_empty()
        || scan
            .inventory
            .windows(2)
            .any(|entries| entries[0].canonical_model_ref > entries[1].canonical_model_ref)
    {
        return Err("scan inventory identity or ordering invalid".into());
    }
    for entry in scan.inventory {
        drop(entry.metadata);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::privacy;

    #[test]
    fn privacy_rejects_legacy_and_current_control_alpns() {
        assert!(privacy(br#"{"nested":["mesh-llm-control/1"]}"#).is_err());
        assert!(privacy(br#"{"nested":["mesh-llm-control/2"]}"#).is_err());
        assert!(privacy(br#"{"nested":["mesh-llm/1"]}"#).is_ok());
    }
}
