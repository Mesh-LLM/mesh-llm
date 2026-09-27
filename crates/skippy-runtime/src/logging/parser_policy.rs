use skippy_ffi::{
    FEATURE_DEVICE_EVENTS, FEATURE_KV_EVENTS, FEATURE_MODEL_LOAD_EVENTS_V2,
    FEATURE_RUNTIME_EVENT_REPORTER,
};

pub(super) const BACKEND_CATEGORY: u8 = 1 << 0;
pub(super) const MODEL_CATEGORY: u8 = 1 << 1;
const MEMORY_CATEGORY: u8 = 1 << 2;
const KV_CACHE_CATEGORY: u8 = 1 << 3;
const TOKENIZER_CATEGORY: u8 = 1 << 4;
pub(super) const ALL_PRESENTATION_CATEGORIES: u8 =
    BACKEND_CATEGORY | MODEL_CATEGORY | MEMORY_CATEGORY | KV_CACHE_CATEGORY | TOKENIZER_CATEGORY;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeLogParserMode {
    Auto,
    Enabled,
    Disabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeLogParserPolicy {
    pub(super) forwarding_mask: u8,
}

impl NativeLogParserPolicy {
    /// Selects which parsed native-log categories are forwarded as
    /// `NativeLogEvent`s.
    ///
    /// `Enabled` forwards every category as an explicit debug aid and
    /// `Disabled` forwards none. `Auto` uses confirmed structured coverage
    /// for backend, memory, KV-cache, and tokenizer categories, while keeping
    /// model summaries forwarded because SafeTensors opens bypass native
    /// model-open events. Event-capable opens may therefore include both.
    pub fn new(mode: NativeLogParserMode, capabilities: &crate::CapabilityReport) -> Self {
        let forwarding_mask = match mode {
            NativeLogParserMode::Enabled => ALL_PRESENTATION_CATEGORIES,
            NativeLogParserMode::Disabled => 0,
            NativeLogParserMode::Auto => {
                ALL_PRESENTATION_CATEGORIES & !structured_coverage(capabilities)
            }
        };
        Self { forwarding_mask }
    }

    pub fn forwards(self, category: &str) -> bool {
        category_mask(category).is_some_and(|mask| self.forwarding_mask & mask != 0)
    }
}

/// Presentation categories reported through confirmed structured event
/// families. The model category stays uncovered because SafeTensors opens
/// bypass the native model-open event callback.
fn structured_coverage(capabilities: &crate::CapabilityReport) -> u8 {
    if !capabilities.family_confirmed(FEATURE_RUNTIME_EVENT_REPORTER) {
        return 0;
    }
    let mut covered = 0;
    if capabilities.family_confirmed(FEATURE_DEVICE_EVENTS) {
        covered |= BACKEND_CATEGORY;
    }
    if capabilities.family_confirmed(FEATURE_KV_EVENTS) {
        covered |= KV_CACHE_CATEGORY;
    }
    if capabilities.family_confirmed(FEATURE_MODEL_LOAD_EVENTS_V2) {
        covered |= MEMORY_CATEGORY | TOKENIZER_CATEGORY;
    }
    covered
}

pub(super) fn category_mask(category: &str) -> Option<u8> {
    match category {
        "backend" => Some(BACKEND_CATEGORY),
        "model" => Some(MODEL_CATEGORY),
        "memory" => Some(MEMORY_CATEGORY),
        "kv_cache" => Some(KV_CACHE_CATEGORY),
        "tokenizer" => Some(TOKENIZER_CATEGORY),
        _ => None,
    }
}
