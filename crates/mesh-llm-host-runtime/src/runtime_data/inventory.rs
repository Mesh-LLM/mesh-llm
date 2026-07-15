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
