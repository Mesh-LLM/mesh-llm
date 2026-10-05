//! Bounded-collection primitives shared by every per-category domain map.
//!
//! Each bounded category keeps a `HashMap` plus a `VecDeque` touch order;
//! these helpers keep the two 1:1 and evict oldest-touched first, so
//! eviction is deterministic instead of relying on `HashMap`'s unspecified
//! iteration order.

use std::collections::{HashMap, VecDeque};

use mesh_llm_runtime_event_contracts::OperationId;

use super::SessionRecentEntry;

/// Move `id` to the back of `order` (most-recently-touched); if `id` is
/// new and `map` is already at `bound`, evict the single oldest entry
/// first.
pub(super) fn touch<V>(
    order: &mut VecDeque<String>,
    map: &mut HashMap<String, V>,
    id: &str,
    bound: usize,
) {
    if let Some(position) = order.iter().position(|existing| existing == id) {
        order.remove(position);
    } else if map.len() >= bound
        && let Some(oldest) = order.pop_front()
    {
        map.remove(&oldest);
    }
    order.push_back(id.to_string());
}

pub(super) fn remove_bounded<V>(
    order: &mut VecDeque<String>,
    map: &mut HashMap<String, V>,
    id: &str,
) {
    if let Some(position) = order.iter().position(|existing| existing == id) {
        order.remove(position);
    }
    map.remove(id);
}

/// Move `root` to the back of `order` (most-recently-touched); if `root` is
/// new and `map` is already at `bound`, evict the single oldest root's
/// identity mapping first. Mirrors [`touch`]'s idiom, applied to the root
/// -> current-model-id correlation map instead of a model_id-keyed map.
pub(super) fn touch_root_identity(
    order: &mut VecDeque<OperationId>,
    map: &mut HashMap<OperationId, String>,
    root: OperationId,
    bound: usize,
) {
    if let Some(position) = order.iter().position(|existing| *existing == root) {
        order.remove(position);
    } else if map.len() >= bound
        && let Some(oldest) = order.pop_front()
    {
        map.remove(&oldest);
    }
    order.push_back(root);
}

pub(super) fn push_recent(
    recent: &mut VecDeque<SessionRecentEntry>,
    entry: SessionRecentEntry,
    bound: usize,
) {
    if recent.len() >= bound {
        recent.pop_front();
    }
    recent.push_back(entry);
}
