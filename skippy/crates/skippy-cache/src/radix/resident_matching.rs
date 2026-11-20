use super::{RadixNode, common_prefix_len};

/// Find the longest query prefix backed by any resident sequence.
///
/// Unlike recurrent checkpoints, native resident KV can be sliced when copied:
/// a cached sequence for `[1, 2, 3, 4]` can restore `[1, 2]` for a request that
/// diverges at token 3. Native restore only reads the stored resident source,
/// so multiple restores may hold references concurrently; eviction remains
/// blocked until every reference is released.
pub(super) fn resident_backing_prefix<R, E>(
    root: &RadixNode<R, E>,
    tokens: &[i32],
) -> Option<(usize, Vec<i32>)> {
    let mut node = root;
    let mut remaining = tokens;
    let mut consumed = 0usize;
    let mut path = Vec::new();
    let mut best = None;

    if node.components.resident.is_some() {
        best = Some((0, Vec::new()));
    }

    while let Some(first) = remaining.first() {
        let Some(child) = node.children.get(first) else {
            // The query can diverge exactly at an existing branch point. No
            // child starts with its next token, but any resident descendant
            // still owns a native sequence that can be sliced to `consumed`.
            // Without this fallback, the first two branches split the edge
            // and every later sibling incorrectly becomes a full miss.
            if consumed > 0
                && let Some(stored_tokens) = nearest_resident_descendant(node, &path)
            {
                best = Some((consumed, stored_tokens));
            }
            break;
        };
        let common = common_prefix_len(&child.edge, remaining);
        if common < child.edge.len() {
            let mut child_path = path.clone();
            child_path.extend_from_slice(&child.edge);
            if common > 0
                && let Some(stored_tokens) = nearest_resident_descendant(child, &child_path)
            {
                best = Some((consumed.saturating_add(common), stored_tokens));
            }
            break;
        }
        consumed = consumed.saturating_add(common);
        remaining = &remaining[common..];
        path.extend_from_slice(&child.edge);
        node = child;
        if node.components.resident.is_some() {
            best = Some((consumed, path.clone()));
        }
    }

    if remaining.is_empty()
        && let Some(stored_tokens) = nearest_resident_descendant(node, &path)
    {
        best = Some((consumed, stored_tokens));
    }
    best.filter(|(matched, _)| *matched > 0)
}

fn nearest_resident_descendant<R, E>(
    node: &RadixNode<R, E>,
    node_path: &[i32],
) -> Option<Vec<i32>> {
    if node.components.resident.is_some() {
        return Some(node_path.to_vec());
    }
    for child in node.children.values() {
        let mut child_path = node_path.to_vec();
        child_path.extend_from_slice(&child.edge);
        if let Some(candidate) = nearest_resident_descendant(child, &child_path) {
            return Some(candidate);
        }
    }
    None
}

use super::{
    ComponentKind, RadixMatch, UnifiedRadixCache, longest_component_prefix, node_at, node_at_mut,
};

impl<R: Clone, E> UnifiedRadixCache<R, E> {
    /// Match only complete stored checkpoints that are prefixes of the query.
    /// Recurrent state cannot be truncated to an arbitrary common prefix.
    pub fn peek_resident_checkpoint(
        &self,
        namespace: &str,
        tokens: &[i32],
    ) -> Option<RadixMatch<R>> {
        let root = self.roots.get(namespace)?;
        let matched = longest_component_prefix(root, tokens, ComponentKind::ResidentKv)?;
        let entry = node_at(root, &tokens[..matched])?
            .components
            .resident
            .as_ref()?;
        Some(RadixMatch {
            matched_tokens: matched,
            stored_tokens: tokens[..matched].to_vec(),
            logical_bytes: entry.logical_bytes,
            active_refs: entry.active_refs,
            value: entry.value.clone(),
        })
    }

    /// Pin the exact recurrent checkpoint chosen by the non-mutating probe.
    pub fn acquire_resident_checkpoint(
        &mut self,
        namespace: &str,
        tokens: &[i32],
    ) -> Option<RadixMatch<R>> {
        let mut hit = self.peek_resident_checkpoint(namespace, tokens)?;
        self.clock = self.clock.saturating_add(1);
        let entry = node_at_mut(self.roots.get_mut(namespace)?, &hit.stored_tokens)?
            .components
            .resident
            .as_mut()?;
        entry.last_used = self.clock;
        entry.active_refs = entry.active_refs.saturating_add(1);
        hit.active_refs = entry.active_refs;
        Some(hit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrent_match_never_slices_a_descendant_checkpoint() {
        let mut tree = UnifiedRadixCache::<u32, ()>::new();
        tree.insert_resident_if_vacant("model", &[1, 2, 3, 4], 64, 4)
            .unwrap();
        assert!(tree.peek_resident_checkpoint("model", &[1, 2]).is_none());
        assert!(tree.peek_resident_checkpoint("model", &[1, 2, 8]).is_none());
        tree.insert_resident_if_vacant("model", &[1, 2], 32, 2)
            .unwrap();
        let hit = tree
            .peek_resident_checkpoint("model", &[1, 2, 3, 9])
            .unwrap();
        assert_eq!(hit.matched_tokens, 2);
        assert_eq!(hit.stored_tokens, [1, 2]);
        assert_eq!(
            tree.peek_resident_checkpoint("model", &[1, 2, 3, 4, 5])
                .unwrap()
                .value,
            4
        );
        assert!(
            tree.peek_resident_checkpoint("other", &[1, 2, 3, 4])
                .is_none()
        );
    }

    #[test]
    fn checkpoint_pin_protects_its_native_owner_until_release() {
        let mut tree = UnifiedRadixCache::<u32, ()>::new();
        tree.insert_resident_if_vacant("model", &[1, 2], 32, 2)
            .unwrap();
        let hit = tree
            .acquire_resident_checkpoint("model", &[1, 2, 3])
            .unwrap();
        assert_eq!(hit.active_refs, 1);
        assert!(tree.evict_resident_candidate("model", &[1, 2]).is_none());
        assert!(tree.release_resident("model", &hit.stored_tokens));
        assert!(tree.evict_resident_candidate("model", &[1, 2]).is_some());
    }
}
