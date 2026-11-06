mod dflash;
mod dflash_catalog;
mod request_defaults;
mod resolution;
mod speculative;
mod support;
mod translation;
mod types;

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod native_mtp_tests;

#[cfg(test)]
mod defaults_parity_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod speculative_tests;

#[cfg(test)]
mod recurrent_rollback_tests;

#[cfg(test)]
mod exact_head_tests;

#[cfg(test)]
mod hardware_tests;

#[cfg(test)]
mod hardware_translation_tests;

pub use dflash_catalog::{
    DFLASH_PAIRING_INDEX_FILE, DFLASH_PAIRING_INDEX_REPO, DFLASH_PAIRING_INDEX_REVISION,
    DFlashBackends, DFlashDraft, DFlashPairing, DFlashPairingIndex, DFlashPairingIndexPin,
    DFlashPairingSource, DFlashTarget, DefaultDFlash, check_split_dflash_draft,
    dflash_draft_cache_path, dflash_pairing_index_installed, install_dflash_pairing_index,
    load_dflash_pairing_index,
};
#[cfg(test)]
pub(crate) use resolution::resolve_skippy_config;
pub use resolution::{
    resolve_embedded_request_defaults, resolve_skippy_config_for_selector,
    resolve_skippy_config_for_selector_with_publisher_defaults,
};
pub use support::effective_safety_margin_bytes;
pub use types::{ResolvedEmbeddedOpenAiArgs, ResolvedSkippyConfig, SkippyConfigResolveRequest};
