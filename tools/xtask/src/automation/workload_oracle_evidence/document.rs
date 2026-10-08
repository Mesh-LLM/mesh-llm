use super::metrics::PcmMetrics;
use super::{Error, WorkloadClass};
use crate::automation::codepoint_json::{parser, value::Value};
use std::path::Path;

pub(super) struct Identity<'a> {
    pub(super) model_class: &'a str,
    pub(super) smoke_lane: &'a str,
    pub(super) oracle_lane: &'a str,
    pub(super) model_id: &'a str,
    pub(super) model_sha256: String,
    pub(super) projector_sha256: Option<String>,
    pub(super) oracle_executable: &'a str,
    pub(super) oracle_executable_sha256: String,
    pub(super) candidate_executable_sha256: String,
    pub(super) pinned_patch_sha: &'a str,
}

impl Identity<'_> {
    fn fields(&self) -> [(&'static str, Option<&str>); 11] {
        [
            ("status", Some("pass")),
            ("class", Some(self.model_class)),
            ("smoke_lane", Some(self.smoke_lane)),
            ("oracle_lane", Some(self.oracle_lane)),
            ("model_id", Some(self.model_id)),
            ("model_sha256", Some(&self.model_sha256)),
            ("projector_sha256", self.projector_sha256.as_deref()),
            ("oracle_executable", Some(self.oracle_executable)),
            (
                "oracle_executable_sha256",
                Some(&self.oracle_executable_sha256),
            ),
            (
                "candidate_executable_sha256",
                Some(&self.candidate_executable_sha256),
            ),
            ("pinned_patch_sha", Some(self.pinned_patch_sha)),
        ]
    }

    pub(super) fn encode(&self, comparison: &str, metrics: Option<PcmMetrics>) -> Vec<u8> {
        let mut fields: Vec<_> = self
            .fields()
            .into_iter()
            .map(|(key, value)| {
                (
                    key.into(),
                    value.map_or(Value::Null, |text| Value::Str(text.into())),
                )
            })
            .collect();
        fields.push(("comparison".into(), Value::Str(comparison.into())));
        if let Some(metrics) = metrics {
            fields.push(("metrics".into(), metrics.into_value()));
        }
        super::serialization::render(&Value::Object(fields)).into_bytes()
    }
}

pub(super) struct EvidenceInput(Value);

impl EvidenceInput {
    pub(super) fn load(path: &Path) -> Result<Self, Error> {
        let value = load(path)?;
        match value {
            Value::Object(_) => Ok(Self(value)),
            Value::Null
            | Value::Bool(_)
            | Value::Int(_)
            | Value::BigInt(_)
            | Value::Float(_)
            | Value::Str(_)
            | Value::Array(_) => Err(Error::Object),
        }
    }

    pub(super) fn verify_identity(&self, expected: &Identity<'_>) -> Result<(), Error> {
        for (field, expected) in expected.fields() {
            let matches = match (self.0.get(field), expected) {
                (Some(Value::Str(actual)), Some(expected)) => actual == expected,
                (None | Some(Value::Null), None) => true,
                _ => false,
            };
            if !matches {
                return Err(Error::Identity(field));
            }
        }
        Ok(())
    }

    pub(super) fn verify_comparison(self, class: WorkloadClass) -> Result<(), Error> {
        let prefix = format!("{} local-monolithic oracle passed: ", class.name());
        if !matches!(self.0.get("comparison"), Some(Value::Str(text)) if text.codepoints().take(prefix.chars().count()).eq(prefix.chars().map(u32::from)))
        {
            return Err(Error::Comparison);
        }
        match class {
            WorkloadClass::SpeechSynthesis => {
                PcmMetrics::from_value(self.0.get("metrics").cloned().unwrap_or(Value::Null))?;
            }
            WorkloadClass::Embedding
            | WorkloadClass::Rerank
            | WorkloadClass::EncoderDecoder
            | WorkloadClass::Ocr
            | WorkloadClass::SpeechRecognition => {}
        }
        Ok(())
    }
}

pub(super) fn tts_result(path: &Path, pinned_patch_sha: &str) -> Result<PcmMetrics, Error> {
    let result = load(path)?;
    if !matches!(result.get("status"), Some(Value::Str(status)) if status == "pass")
        || !matches!(result.get("pinned_patch_sha"), Some(Value::Str(patch)) if patch == pinned_patch_sha)
    {
        return Err(Error::TtsResult);
    }
    PcmMetrics::from_value(result.get("metrics").cloned().unwrap_or(Value::Null))
}

fn load(path: &Path) -> Result<Value, Error> {
    let bytes = std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    parser::parse(&bytes).map_err(Error::Json)
}

pub(super) fn sha256(path: &Path) -> Result<String, Error> {
    crate::product::digest::file_sha256(path).map_err(|failure| Error::Io {
        path: failure.path,
        source: failure.error,
    })
}
