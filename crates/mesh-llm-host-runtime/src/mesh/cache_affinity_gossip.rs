//! Cache-affinity advertisement construction, comparison, and merge policy.

use std::sync::Mutex;

use mesh_llm_routing::cache_inventory::{
    CacheAffinityAdvertisement, CacheInventory, rotating_salt,
};

pub(super) fn advertised_state_changed(
    old: &Option<CacheAffinityAdvertisement>,
    new: &Option<CacheAffinityAdvertisement>,
) -> bool {
    match (old, new) {
        (Some(old), Some(new)) => !old.has_same_cache_state(new),
        (None, None) => false,
        _ => true,
    }
}

pub(super) fn merge_advertisement(
    existing: &mut Option<CacheAffinityAdvertisement>,
    incoming: Option<&CacheAffinityAdvertisement>,
    clear_on_absence: bool,
) {
    match (existing.as_ref(), incoming) {
        (_, None) if clear_on_absence => *existing = None,
        (None, Some(incoming)) => *existing = Some(incoming.clone()),
        (Some(current), Some(incoming)) if incoming.is_newer_than(current) => {
            *existing = Some(incoming.clone());
        }
        _ => {}
    }
}

pub(super) fn local_advertisement(
    inventory: &Mutex<CacheInventory>,
    endpoint_id: &[u8],
    now_unix_ms: u64,
) -> CacheAffinityAdvertisement {
    let salt = rotating_salt(endpoint_id, now_unix_ms);
    inventory
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .advertisement(salt, now_unix_ms)
}
