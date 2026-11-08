mod dflash;
mod dflash_catalog;
mod dflash_index;
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
mod exact_head_tests;

#[cfg(test)]
mod hardware_tests;

#[cfg(test)]
mod hardware_translation_tests;

pub use dflash_catalog::{
    DFlashBackends, DFlashDraft, DFlashPairing, DFlashPairingIndex, DFlashPairingSource,
    DFlashTarget, DefaultDFlash, check_split_dflash_draft, dflash_draft_cache_path,
    dflash_pairing_index_installed, install_dflash_pairing_index,
};
pub use dflash_index::{
    DFLASH_PAIRING_INDEX_REPO, DFLASH_PAIRING_INDEX_SHA256, DFlashPairingIndexPin,
};
#[cfg(test)]
pub(crate) use resolution::resolve_skippy_config;
pub use resolution::{
    resolve_embedded_request_defaults, resolve_skippy_config_for_selector,
    resolve_skippy_config_for_selector_with_publisher_defaults,
};
pub use support::effective_safety_margin_bytes;
pub use types::{ResolvedEmbeddedOpenAiArgs, ResolvedSkippyConfig, SkippyConfigResolveRequest};
