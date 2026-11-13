//! What serving events may say about the model behind a request.
//!
//! A model is named only by a public identity the runtime already holds: the
//! name of the catalog entry it was loaded from. Everything else, including a
//! local file, a configured alias, a private Hugging Face repository, or a
//! name a client typed, is `redacted` and described instead by its
//! architecture and a coarse size, which say what the model is like without
//! saying which one it is.

use std::collections::HashMap;
use std::sync::Mutex;

use mesh_llm_analytics::{Label, Properties};

/// Served name to identity, for every model this process loaded.
///
/// The key is the name skippy and the router see, which may be a file stem
/// or an alias. It is a lookup key and never goes on the wire.
static LOADED_MODELS: Mutex<Option<HashMap<String, ModelIdentity>>> = Mutex::new(None);

/// The reportable description of one model.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModelIdentity {
    /// The catalog name, or `redacted`.
    pub(crate) model: Label,
    /// `general.architecture`, such as `qwen3moe`.
    pub(crate) architecture: Option<Label>,
    /// Parameter count, bucketed by [`parameter_bucket`].
    pub(crate) parameters: Option<&'static str>,
    /// Whether the model routes through experts.
    pub(crate) moe: Option<bool>,
    /// The model family a published DFlash pairing matched, by shape.
    pub(crate) family: Option<Label>,
    /// The public repository of the DFlash draft attached by default.
    pub(crate) dflash_draft: Option<Label>,
    /// The backend of the device the model was loaded onto, when the load
    /// selected one. Reported with the hardware dimensions, not here.
    pub(crate) backend: Option<&'static str>,
}

impl ModelIdentity {
    /// An unknown model: `redacted`, with nothing else known.
    pub(crate) fn redacted() -> Self {
        Self {
            model: Label::redacted(),
            architecture: None,
            parameters: None,
            moe: None,
            family: None,
            dflash_draft: None,
            backend: None,
        }
    }

    /// Attach this identity to an event.
    pub(crate) fn apply(&self, properties: Properties) -> Properties {
        let mut properties = properties.with("model", self.model.clone());
        if let Some(architecture) = &self.architecture {
            properties = properties.with("model_arch", architecture.clone());
        }
        if let Some(parameters) = self.parameters {
            properties = properties.with("model_params", parameters);
        }
        if let Some(moe) = self.moe {
            properties = properties.with("model_moe", moe);
        }
        if let Some(family) = &self.family {
            properties = properties.with("model_family", family.clone());
        }
        if let Some(draft) = &self.dflash_draft {
            properties = properties.with("dflash_draft", draft.clone());
        }
        properties
    }
}

/// Remember how to describe `served_name` on serving events.
pub(crate) fn register(served_name: &str, identity: ModelIdentity) {
    if let Ok(mut models) = LOADED_MODELS.lock() {
        models
            .get_or_insert_with(HashMap::new)
            .insert(served_name.to_owned(), identity);
    }
}

/// The identity for a served or requested model name.
///
/// A model this node loaded is described as it was registered. Any other
/// name is reported only when it is itself a catalog name, which is public by
/// definition; a remote peer's model routed through this node lands there.
pub(crate) fn lookup(name: &str) -> ModelIdentity {
    let registered = LOADED_MODELS
        .lock()
        .ok()
        .and_then(|models| models.as_ref().and_then(|map| map.get(name).cloned()));
    if let Some(identity) = registered {
        return identity;
    }
    match catalog_name_for_request(name) {
        Some(catalog_name) => ModelIdentity {
            model: Label::sanitize_or_redact(&catalog_name),
            ..ModelIdentity::redacted()
        },
        None => ModelIdentity::redacted(),
    }
}

/// The catalog name `name` exactly is, if any.
///
/// Only the curated name or the exact reference count: both are published by
/// the catalog. A file stem is deliberately not matched, because a client's
/// model name can be a local file's stem that merely looks like one.
fn catalog_name_for_request(name: &str) -> Option<String> {
    crate::models::remote_catalog::loaded_models()
        .ok()?
        .into_iter()
        .find(|model| model.name == name || model.exact_ref() == name)
        .map(|model| model.name)
}

/// The backend of a selected device name: `CUDA0` is `cuda`, `CPU` is
/// `cpu`. A device no backend name covers reports nothing rather than a guess.
pub(crate) fn backend_of_selected_device(device: &str) -> Option<&'static str> {
    mesh_llm_analytics::backend_of_device(device).or_else(|| {
        device
            .to_ascii_lowercase()
            .starts_with("cpu")
            .then_some("cpu")
    })
}

/// Bucket a GGUF `general.size_label` (`8B`, `0.6B`, `35B-A3B`, `8x7B`,
/// `270M`) into a coarse parameter class.
///
/// The bucket is the whole number of billions, so edges never overlap: 3.8B
/// is `1-3b`, 4B is `4-9b`. A label that does not parse reports nothing
/// rather than a guess.
pub(crate) fn parameter_bucket(size_label: &str) -> Option<&'static str> {
    let billions = parse_billions(size_label)?;
    Some(match billions {
        b if b < 1.0 => "lt1b",
        b if b < 4.0 => "1-3b",
        b if b < 10.0 => "4-9b",
        b if b < 20.0 => "10-19b",
        b if b < 40.0 => "20-39b",
        b if b < 80.0 => "40-79b",
        b if b < 160.0 => "80-159b",
        b if b < 400.0 => "160-399b",
        _ => "400b+",
    })
}

/// Total parameters in billions from a size label. MoE labels give the total
/// first (`35B-A3B`), and `8x7B` multiplies out.
fn parse_billions(size_label: &str) -> Option<f64> {
    let total = size_label.trim().split('-').next()?;
    let (count, total) = match total.split_once(['x', 'X']) {
        Some((count, rest)) => (count.parse::<f64>().ok()?, rest),
        None => (1.0, total),
    };
    let unit_at = total.find(|c: char| !(c.is_ascii_digit() || c == '.'))?;
    let (number, unit) = total.split_at(unit_at);
    let scale = match unit {
        "T" | "t" => 1000.0,
        "B" | "b" => 1.0,
        "M" | "m" => 0.001,
        "K" | "k" => 0.000_001,
        _ => return None,
    };
    let value = number.parse::<f64>().ok()? * count * scale;
    (value.is_finite() && value > 0.0).then_some(value)
}

#[cfg(test)]
#[path = "model_identity/tests.rs"]
mod tests;
