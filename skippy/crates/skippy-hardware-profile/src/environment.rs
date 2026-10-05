//! Environment override parsing for hardware probes.

use std::collections::BTreeSet;
pub(crate) fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok()?.parse().ok()
}

pub(crate) fn env_u32_set(name: &str) -> BTreeSet<u32> {
    env_string_vec(name)
        .into_iter()
        .filter_map(|value| value.parse().ok())
        .collect()
}

pub(crate) fn env_string_set(name: &str) -> BTreeSet<String> {
    env_string_vec(name).into_iter().collect()
}

pub(crate) fn env_string_vec(name: &str) -> Vec<String> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}
