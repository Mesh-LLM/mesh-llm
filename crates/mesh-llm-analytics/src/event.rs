//! Typed analytics events and a closed property vocabulary.
//!
//! The vocabulary is deliberately closed. Callers cannot hand this crate an
//! arbitrary `String`, so a prompt, a completion, a filesystem path, a peer
//! address, or an API key cannot reach the wire by accident: the only way to
//! attach text is [`Label::sanitize`], which enforces a catalog-shaped grammar
//! and rejects everything else.

use serde::Serialize;
use std::collections::BTreeMap;

/// The closed set of events mesh-llm reports.
///
/// Adding a variant here is the only way to report a new event, which keeps
/// the reported surface reviewable in one place.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Event {
    /// First run of a given install. Emitted once per install identifier.
    InstallFirstRun,
    /// A one-shot CLI command finished. Carries only family and outcome.
    CliCommand,
    /// `mesh-llm serve` reached a serving state.
    ServeStarted,
    /// A `serve` process shut down.
    ServeStopped,
    /// A model became resident on this node.
    ModelLoaded,
    /// A model download was attempted.
    ModelDownload,
    /// The shape of this machine, reported once per serving process.
    HardwareProfile,
}

impl Event {
    /// The event name as PostHog stores it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::InstallFirstRun => "install_first_run",
            Self::CliCommand => "cli_command",
            Self::ServeStarted => "serve_started",
            Self::ServeStopped => "serve_stopped",
            Self::ModelLoaded => "model_loaded",
            Self::ModelDownload => "model_download",
            Self::HardwareProfile => "hardware_profile",
        }
    }
}

/// A grammar-checked string property.
///
/// Construction is only possible through [`Label::sanitize`]. The grammar
/// admits catalog and repository identifiers (`Qwen2.5-7B-Instruct-Q4_K_M`,
/// `Qwen/Qwen2.5-7B`) and rejects anything path-shaped, whitespace-bearing,
/// or over-long.
///
/// ```
/// use mesh_llm_analytics::Label;
///
/// assert!(Label::sanitize("Qwen/Qwen2.5-7B").is_some());
/// assert!(Label::sanitize("/Users/dan/models/private.gguf").is_none());
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct Label(String);

/// Longest accepted label. Catalog names sit far below this.
const MAX_LABEL_LEN: usize = 96;

/// Longest accepted slug. Device names sit well below this; prose does not.
const MAX_SLUG_LEN: usize = 48;

/// Most hyphen-separated segments a slug may have.
///
/// `nvidia-geforce-rtx-4090-laptop-gpu` is six, which is about the longest
/// real device name. A sentence has many more.
const MAX_SLUG_SEGMENTS: usize = 8;

impl Label {
    /// Accept `value` only if it matches the label grammar.
    ///
    /// The grammar is: ASCII alphanumerics plus `.`, `-`, `_`, `+`, and at
    /// most one interior `/`; no leading or trailing separator; no `..`; at
    /// most [`MAX_LABEL_LEN`] characters.
    ///
    /// `+` is admitted for semver build metadata (`0.76.0+gABCDEF.dirty`),
    /// which is the version string this crate reports about itself.
    pub fn sanitize(value: &str) -> Option<Self> {
        if value.is_empty() || value.len() > MAX_LABEL_LEN {
            return None;
        }
        if value.contains("..") {
            return None;
        }
        if value.matches('/').count() > 1 {
            return None;
        }
        if value.starts_with(['/', '-', '.']) || value.ends_with('/') {
            return None;
        }
        let permitted = value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+' | '/'));
        permitted.then(|| Self(value.to_owned()))
    }

    /// Normalize a human-readable device name into a label.
    ///
    /// Hardware probes return names like `Apple M4 Max` or
    /// `NVIDIA GeForce RTX 4090`, which [`Label::sanitize`] rejects for
    /// containing spaces.
    ///
    /// Collapsing separators is dangerous on its own: it would turn
    /// `/home/dan/my model.gguf` into the perfectly valid-looking
    /// `home-dan-my-model.gguf`, laundering a path past the grammar. Two
    /// extra rules prevent that, and they are why this is a distinct
    /// constructor rather than a pre-pass on `sanitize`:
    ///
    /// - Input containing a path separator is refused outright, never
    ///   normalized.
    /// - The result is capped at [`MAX_SLUG_SEGMENTS`] segments and
    ///   [`MAX_SLUG_LEN`] characters, which every real device name fits and
    ///   prose does not.
    ///
    /// ```
    /// use mesh_llm_analytics::Label;
    ///
    /// assert_eq!(Label::slug("Apple M4 Max").unwrap().as_str(), "apple-m4-max");
    /// assert!(Label::slug("/Users/dan/models/private.gguf").is_none());
    /// ```
    pub fn slug(value: &str) -> Option<Self> {
        // Refuse path-shaped input rather than normalizing the separators
        // away. This is the rule that stops slugging from laundering a path.
        if value.contains('/') || value.contains('\\') {
            return None;
        }

        let mut slugged = String::with_capacity(value.len());
        let mut pending_separator = false;
        for ch in value.trim().chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '+') {
                if pending_separator && !slugged.is_empty() {
                    slugged.push('-');
                }
                pending_separator = false;
                slugged.extend(ch.to_lowercase());
            } else {
                // Every non-kept character collapses into a single `-`, so
                // runs of spaces and punctuation do not produce `--`.
                pending_separator = true;
            }
        }

        if slugged.len() > MAX_SLUG_LEN || slugged.split('-').count() > MAX_SLUG_SEGMENTS {
            return None;
        }
        Self::sanitize(&slugged)
    }

    /// A label for a value that failed sanitization, so counts stay complete.
    #[must_use]
    pub fn redacted() -> Self {
        Self("redacted".to_owned())
    }

    /// Sanitize `value`, falling back to [`Label::redacted`].
    ///
    /// Use this when the count matters more than the identity, so an
    /// unrecognized model name still contributes to totals without leaking
    /// whatever the user actually typed.
    #[must_use]
    pub fn sanitize_or_redact(value: &str) -> Self {
        Self::sanitize(value).unwrap_or_else(Self::redacted)
    }

    /// Slug `value`, falling back to [`Label::redacted`].
    #[must_use]
    pub fn slug_or_redact(value: &str) -> Self {
        Self::slug(value).unwrap_or_else(Self::redacted)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A single property value. Deliberately not `From<String>`.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Text(Label),
    /// A compile-time constant, safe by construction.
    Static(&'static str),
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Self::Int(i64::try_from(value).unwrap_or(i64::MAX))
    }
}

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Self::Int(i64::try_from(value).unwrap_or(i64::MAX))
    }
}

impl From<Label> for Value {
    fn from(value: Label) -> Self {
        Self::Text(value)
    }
}

impl From<&'static str> for Value {
    fn from(value: &'static str) -> Self {
        Self::Static(value)
    }
}

/// Properties attached to one event.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Properties(BTreeMap<&'static str, Value>);

impl Properties {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach one property. Keys are `'static` so they cannot be user data.
    #[must_use]
    pub fn with(mut self, key: &'static str, value: impl Into<Value>) -> Self {
        self.0.insert(key, value.into());
        self
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (&'static str, &Value)> {
        self.0.iter().map(|(key, value)| (*key, value))
    }
}

/// Bucket a count so a precise fingerprint never leaves the machine.
///
/// Exact peer or GPU counts are unusual enough at the tail to be identifying;
/// the buckets keep the distribution useful without that.
#[must_use]
pub fn bucket_count(count: u64) -> &'static str {
    match count {
        0 => "0",
        1 => "1",
        2 => "2",
        3..=4 => "3-4",
        5..=8 => "5-8",
        9..=16 => "9-16",
        17..=32 => "17-32",
        _ => "33+",
    }
}

/// Bucket a byte size into human power-of-two ranges.
#[must_use]
pub fn bucket_gigabytes(bytes: u64) -> &'static str {
    const GB: u64 = 1024 * 1024 * 1024;
    match bytes / GB {
        0..=7 => "0-8",
        8..=15 => "8-16",
        16..=31 => "16-32",
        32..=63 => "32-64",
        64..=127 => "64-128",
        128..=255 => "128-256",
        _ => "256+",
    }
}

/// Bucket a duration in seconds into coarse session lengths.
#[must_use]
pub fn bucket_duration_secs(secs: u64) -> &'static str {
    match secs {
        0..=59 => "under_1m",
        60..=899 => "1m-15m",
        900..=3599 => "15m-1h",
        3600..=21599 => "1h-6h",
        21600..=86399 => "6h-24h",
        _ => "over_24h",
    }
}

#[cfg(test)]
#[path = "event/tests.rs"]
mod tests;
