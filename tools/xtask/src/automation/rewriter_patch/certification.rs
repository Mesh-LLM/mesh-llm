use super::ShardError;
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum FamilyIdentity {
    Text(String),
    Integer(String),
    Fraction(u64),
}

impl FamilyIdentity {
    fn parse(value: &Value) -> Result<Self, ShardError> {
        match value {
            Value::String(text) => Ok(Self::Text(text.clone())),
            Value::Bool(flag) => Ok(Self::Integer(u8::from(*flag).to_string())),
            Value::Number(number) => {
                if let Some(integer) = number.as_i64() {
                    return Ok(Self::Integer(integer.to_string()));
                }
                if let Some(integer) = number.as_u64() {
                    return Ok(Self::Integer(integer.to_string()));
                }
                let float = number.as_f64().ok_or(ShardError::ManifestFamily)?;
                if float == 0.0 {
                    Ok(Self::Integer("0".to_owned()))
                } else if float.fract() == 0.0 {
                    Ok(Self::Integer(format!("{float:.0}")))
                } else {
                    Ok(Self::Fraction(float.to_bits()))
                }
            }
            Value::Null | Value::Array(_) | Value::Object(_) => Err(ShardError::ManifestFamily),
        }
    }

    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Integer(_) | Self::Fraction(_) => None,
        }
    }

    pub(super) fn label(&self) -> String {
        match self {
            Self::Text(text) | Self::Integer(text) => text.clone(),
            Self::Fraction(bits) => f64::from_bits(*bits).to_string(),
        }
    }
}

pub(super) fn causal_families(manifest: &Value) -> Result<BTreeSet<FamilyIdentity>, ShardError> {
    let models = manifest
        .get("models")
        .and_then(Value::as_array)
        .ok_or(ShardError::ManifestModels)?;
    let mut seen = BTreeSet::new();
    for model in models {
        let family = FamilyIdentity::parse(model.get("family").ok_or(ShardError::ManifestFamily)?)?;
        if !seen.insert(family) {
            return Err(ShardError::ManifestFamily);
        }
    }
    let mut causal = BTreeSet::new();
    for model in models {
        let profile = model.get("profile").and_then(Value::as_str);
        match model.get("class").and_then(Value::as_str) {
            Some("causal_generation") => {
                if !matches!(profile, Some("full" | "package-oracle" | "graph-only")) {
                    return Err(ShardError::CausalProfile);
                }
                causal.insert(FamilyIdentity::parse(
                    model.get("family").ok_or(ShardError::ManifestFamily)?,
                )?);
            }
            Some(
                "embedding" | "rerank" | "encoder_decoder" | "ocr" | "speech_synthesis"
                | "speech_recognition",
            ) => {
                if !matches!(profile, Some("workload-smoke" | "workload-oracle")) {
                    return Err(ShardError::WorkloadProfile);
                }
            }
            _ => return Err(ShardError::WorkloadClass),
        }
    }
    Ok(causal)
}
