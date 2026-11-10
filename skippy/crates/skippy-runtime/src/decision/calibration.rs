//! Decision calibration: per-question-kind temperature scaling for typed decisions.
//!
//! Calibration turns the causal backend's relative candidate mass into calibrated
//! probabilities (see `probability_semantics` in the System One wire contract). The
//! artifact schema is owned by `training/decision-calibration` in the monorepo
//! (calibration.json, schema_version 1); this is the serving-side reader.
//!
//! Temperature scaling is exact on log-probabilities: softmax(scores / T). T near 1
//! means the backend is natively calibrated; T > 1 flattens overconfidence.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Artifact schema version this reader understands.
pub const CALIBRATION_SCHEMA_VERSION: &str = "1";

/// One kind's calibration. The canonical artifact form is `{"temperature": 1.64}`;
/// a bare number (`"choice": 1.64`) is also accepted for hand-written files.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
#[serde(untagged)]
pub enum KindCalibration {
    Number(f64),
    Table { temperature: f64 },
}

impl KindCalibration {
    pub fn temperature(&self) -> f64 {
        match self {
            Self::Number(t) => *t,
            Self::Table { temperature } => *temperature,
        }
    }
}

impl<'de> Deserialize<'de> for KindCalibration {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Form {
            Number(f64),
            Table { temperature: f64 },
        }
        match Form::deserialize(deserializer)? {
            Form::Number(t) => Ok(Self::Number(t)),
            Form::Table { temperature } => Ok(Self::Table { temperature }),
        }
    }
}

/// Per-kind temperatures. Kinds absent from the artifact keep T = 1.0 (uncalibrated).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DecisionCalibration {
    #[serde(default)]
    pub schema_version: Option<String>,
    /// Question kind ("noul" | "choice" | "score") → calibration.
    #[serde(default)]
    pub kinds: BTreeMap<String, KindCalibration>,
}

impl DecisionCalibration {
    /// Load a calibration artifact from disk. A missing file means "uncalibrated",
    /// not an error: serving behavior is unchanged without an artifact.
    pub fn load(path: &Path) -> anyhow::Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let calibration: DecisionCalibration = serde_json::from_str(&text)?;
                calibration.validate()?;
                Ok(Some(calibration))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        if let Some(version) = &self.schema_version {
            anyhow::ensure!(
                version == CALIBRATION_SCHEMA_VERSION,
                "unsupported calibration schema_version {version:?}"
            );
        }
        for (kind, t) in &self.kinds {
            anyhow::ensure!(
                matches!(kind.as_str(), "noul" | "choice" | "score"),
                "unknown calibration kind {kind:?}"
            );
            anyhow::ensure!(
                t.temperature() > 0.0,
                "calibration temperature for {kind} must be positive"
            );
        }
        Ok(())
    }

    /// Temperature for a question kind; 1.0 when the kind is not calibrated.
    pub fn temperature(&self, kind: &str) -> f64 {
        self.kinds
            .get(kind)
            .map_or(1.0, KindCalibration::temperature)
    }

    /// True when at least one kind deviates from identity.
    pub fn is_active(&self) -> bool {
        self.kinds
            .values()
            .any(|t| (t.temperature() - 1.0).abs() > f64::EPSILON)
    }
}

/// Discover a calibration artifact beside a model file:
/// `<model-stem>.calibration.json`, then `calibration.json` in the same directory.
pub fn discover_beside_model(model_path: &Path) -> Option<std::path::PathBuf> {
    let dir = model_path.parent()?;
    let stem = model_path.file_stem()?.to_str()?;
    let specific = dir.join(format!("{stem}.calibration.json"));
    if specific.is_file() {
        return Some(specific);
    }
    let generic = dir.join("calibration.json");
    if generic.is_file() {
        return Some(generic);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_none_not_error() {
        let dir = std::env::temp_dir().join("skippy-calibration-absent-test");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            DecisionCalibration::load(&dir.join("calibration.json")).unwrap(),
            None
        );
    }

    #[test]
    fn roundtrip_and_lookup() {
        // Canonical artifact form, as written by training/decision-calibration.
        let json = r#"{
            "schema_version": "1",
            "kinds": {"noul": {"temperature": 0.89}, "choice": {"temperature": 1.64}, "score": {"temperature": 2.09}}
        }"#;
        let cal: DecisionCalibration = serde_json::from_str(json).unwrap();
        assert!(cal.is_active());
        assert_eq!(cal.temperature("noul"), 0.89);
        assert_eq!(cal.temperature("score"), 2.09);
        // Absent kinds are identity.
        assert_eq!(cal.temperature("other"), 1.0);
        // Identity-only artifact is not active.
        let identity: DecisionCalibration =
            serde_json::from_str(r#"{"kinds": {"noul": 1.0}}"#).unwrap();
        assert!(!identity.is_active());
    }

    #[test]
    fn invalid_inputs_rejected() {
        assert!(
            serde_json::from_str::<DecisionCalibration>(r#"{"kinds": {"noul": 0.0}}"#)
                .map(|c| c.validate().is_ok())
                .unwrap_or(false)
                == false
        );
        assert!(
            serde_json::from_str::<DecisionCalibration>(r#"{"schema_version": "99", "kinds": {}}"#)
                .map(|c| c.validate().is_ok())
                .unwrap_or(false)
                == false
        );
    }

    #[test]
    fn discovery_prefers_specific_sidecar() {
        let dir = std::env::temp_dir().join("skippy-calibration-discovery-test");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        std::fs::write(&model, b"").unwrap();
        assert_eq!(discover_beside_model(&model), None);
        let generic = dir.join("calibration.json");
        std::fs::write(&generic, "{}").unwrap();
        assert_eq!(discover_beside_model(&model).unwrap(), generic);
        let specific = dir.join("model.calibration.json");
        std::fs::write(&specific, "{}").unwrap();
        assert_eq!(discover_beside_model(&model).unwrap(), specific);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn bare_number_form_also_accepted() {
    let json = r#"{"kinds": {"noul": 0.89}}"#;
    let cal: DecisionCalibration = serde_json::from_str(json).unwrap();
    assert_eq!(cal.temperature("noul"), 0.89);
}
