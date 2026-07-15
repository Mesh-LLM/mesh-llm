use super::snapshots::LocalInstancesSnapshot;
use crate::models::LocalModelInventorySnapshot;
use crate::runtime::instance::LocalInstanceSnapshot;
use std::fmt;
use tokio::sync::oneshot;

pub(crate) type InventoryScanResult = Result<LocalModelInventorySnapshot, InventoryScanError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum InventoryScanError {
    #[allow(dead_code)]
    LoaderFailed(String),
    TaskPanicked(String),
    TaskCancelled,
}

impl fmt::Display for InventoryScanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoaderFailed(message) => {
                write!(formatter, "local inventory scan failed: {message}")
            }
            Self::TaskPanicked(message) => {
                write!(formatter, "local inventory scan task panicked: {message}")
            }
            Self::TaskCancelled => formatter.write_str("local inventory scan task was cancelled"),
        }
    }
}

impl std::error::Error for InventoryScanError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InventoryScanDisposition {
    Executed,
    Coalesced,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InventoryScanOutcome {
    pub(crate) snapshot: LocalModelInventorySnapshot,
    pub(crate) disposition: InventoryScanDisposition,
}

pub(crate) fn sorted_inventory_entries(
    snapshot: &LocalModelInventorySnapshot,
) -> Vec<crate::proto::node::OwnerControlInventoryEntry> {
    let mut model_refs = snapshot.model_names.iter().cloned().collect::<Vec<_>>();
    model_refs.sort();
    model_refs
        .into_iter()
        .map(|canonical_model_ref| {
            let mut metadata = snapshot.metadata_by_name.get(&canonical_model_ref).cloned();
            if let Some(metadata) = &mut metadata {
                metadata.model_key.clone_from(&canonical_model_ref);
            }
            crate::proto::node::OwnerControlInventoryEntry {
                display_name: snapshot
                    .display_name_by_name
                    .get(&canonical_model_ref)
                    .cloned(),
                total_size_bytes: snapshot
                    .size_by_name
                    .get(&canonical_model_ref)
                    .copied()
                    .unwrap_or_default(),
                metadata,
                canonical_model_ref,
            }
        })
        .collect()
}

#[derive(Default)]
pub(crate) struct InventoryScanCoordinator {
    running: bool,
    waiters: Vec<oneshot::Sender<InventoryScanResult>>,
}

impl InventoryScanCoordinator {
    pub(crate) fn begin_or_join(&mut self) -> (oneshot::Receiver<InventoryScanResult>, bool) {
        let (tx, rx) = oneshot::channel();
        self.waiters.push(tx);
        if self.running {
            (rx, false)
        } else {
            self.running = true;
            (rx, true)
        }
    }

    pub(crate) fn finish(&mut self) -> Vec<oneshot::Sender<InventoryScanResult>> {
        self.running = false;
        std::mem::take(&mut self.waiters)
    }

    #[cfg(test)]
    pub(crate) fn waiters_len(&self) -> usize {
        self.waiters.len()
    }
}

pub(crate) fn replace_local_instances_snapshot(
    current: &mut LocalInstancesSnapshot,
    replacement: Vec<LocalInstanceSnapshot>,
) -> bool {
    if current.instances == replacement {
        return false;
    }

    current.instances = replacement;
    true
}

pub(crate) fn replace_local_inventory_snapshot(
    current: &mut LocalModelInventorySnapshot,
    replacement: LocalModelInventorySnapshot,
) -> bool {
    if *current == replacement {
        return false;
    }

    *current = replacement;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn inventory_entries_are_sorted_and_normalize_metadata_model_key() {
        let snapshot = LocalModelInventorySnapshot {
            model_names: HashSet::from(["z/model".to_string(), "a/model".to_string()]),
            size_by_name: HashMap::from([("a/model".to_string(), 42)]),
            metadata_by_name: HashMap::from([(
                "a/model".to_string(),
                crate::proto::node::CompactModelMetadata {
                    model_key: "stale-key".to_string(),
                    ..Default::default()
                },
            )]),
            display_name_by_name: HashMap::from([("z/model".to_string(), "Zed".to_string())]),
        };

        let entries = sorted_inventory_entries(&snapshot);

        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.canonical_model_ref.as_str())
                .collect::<Vec<_>>(),
            vec!["a/model", "z/model"]
        );
        assert_eq!(entries[0].total_size_bytes, 42);
        assert_eq!(
            entries[0]
                .metadata
                .as_ref()
                .map(|metadata| metadata.model_key.as_str()),
            Some("a/model")
        );
        assert_eq!(entries[1].display_name.as_deref(), Some("Zed"));
    }
}
