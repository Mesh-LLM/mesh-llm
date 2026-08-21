//! Node-owned accounting for the optional KV disk tier.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

const MAX_NODE_BUDGET_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const FREE_SPACE_PERCENT: u64 = 20;
const MIN_FREE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NodeBudget {
    Explicit(u64),
    Derived(u64),
    InsufficientSpace { free_bytes: u64 },
    Disabled,
}

impl NodeBudget {
    pub(super) fn bytes(self) -> Option<u64> {
        match self {
            Self::Explicit(bytes) | Self::Derived(bytes) => Some(bytes),
            Self::InsufficientSpace { .. } | Self::Disabled => None,
        }
    }
}

pub(super) fn resolve_node_budget(
    explicit_bytes: Option<u64>,
    enabled: bool,
    free_bytes: Option<u64>,
) -> NodeBudget {
    if let Some(bytes) = explicit_bytes {
        return match free_bytes {
            Some(free) if free < bytes.saturating_add(MIN_FREE_BYTES) => {
                NodeBudget::InsufficientSpace { free_bytes: free }
            }
            _ => NodeBudget::Explicit(bytes),
        };
    }
    if !enabled {
        return NodeBudget::Disabled;
    }
    let Some(free) = free_bytes else {
        return NodeBudget::InsufficientSpace { free_bytes: 0 };
    };
    if free < MIN_FREE_BYTES {
        return NodeBudget::InsufficientSpace { free_bytes: free };
    }
    NodeBudget::Derived((free.saturating_mul(FREE_SPACE_PERCENT) / 100).min(MAX_NODE_BUDGET_BYTES))
}

#[derive(Debug)]
struct Pool {
    limit: u64,
    reserved: u64,
}

static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();

/// A lifetime-bound reservation. Dropping the last clone returns its bytes.
#[derive(Clone, Debug)]
pub(super) struct BudgetReservation(Arc<ReservationInner>);

#[derive(Debug)]
struct ReservationInner {
    bytes: u64,
}

impl BudgetReservation {
    pub(super) fn bytes(&self) -> u64 {
        self.0.bytes
    }
}

impl Drop for ReservationInner {
    fn drop(&mut self) {
        if let Some(pool) = POOL.get() {
            let mut pool = pool.lock().expect("KV disk budget pool poisoned");
            pool.reserved = pool.reserved.saturating_sub(self.bytes);
        }
    }
}

/// Reserve a stage allowance from the node total.
///
/// The caller supplies its desired working-set bound. Unused capacity remains
/// available to later stages, and the reservation is returned when its owner
/// is dropped; there are no permanently stranded fixed quarters.
pub(super) fn reserve(node_budget: u64, desired_bytes: u64) -> Option<BudgetReservation> {
    let pool = POOL.get_or_init(|| {
        Mutex::new(Pool {
            limit: node_budget,
            reserved: 0,
        })
    });
    let mut pool = pool.lock().expect("KV disk budget pool poisoned");
    pool.limit = node_budget;
    let available = pool.limit.saturating_sub(pool.reserved);
    let bytes = desired_bytes.min(available);
    if bytes == 0 {
        return None;
    }
    pool.reserved += bytes;
    Some(BudgetReservation(Arc::new(ReservationInner { bytes })))
}

pub(super) fn existing_ancestor(path: &Path) -> PathBuf {
    let mut current = path;
    loop {
        if current.exists() {
            return current.to_path_buf();
        }
        match current.parent() {
            Some(parent) if parent.as_os_str().is_empty() => {
                return PathBuf::from(if path.is_absolute() { "/" } else { "." });
            }
            Some(parent) => current = parent,
            None => return PathBuf::from(if path.is_absolute() { "/" } else { "." }),
        }
    }
}

pub(super) fn free_space_bytes(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
        let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(c_path.as_ptr(), &mut stats) } != 0 {
            return None;
        }
        let block_size = if stats.f_frsize > 0 {
            stats.f_frsize as u64
        } else {
            stats.f_bsize as u64
        };
        Some(stats.f_bavail as u64 * block_size)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn default_budget_is_safe_and_capped() {
        assert_eq!(
            resolve_node_budget(None, true, Some(200 * GIB)),
            NodeBudget::Derived(40 * GIB)
        );
        assert_eq!(
            resolve_node_budget(None, true, Some(4000 * GIB)),
            NodeBudget::Derived(64 * GIB)
        );
        assert!(matches!(
            resolve_node_budget(None, true, Some(4 * GIB)),
            NodeBudget::InsufficientSpace { .. }
        ));
    }

    #[test]
    fn disabled_is_an_explicit_policy() {
        assert_eq!(
            resolve_node_budget(None, false, Some(500 * GIB)),
            NodeBudget::Disabled
        );
    }

    #[test]
    fn reservation_clone_keeps_ownership_until_last_clone_drops() {
        let reservation = BudgetReservation(Arc::new(ReservationInner { bytes: 600 }));
        let worker_reservation = reservation.clone();
        assert_eq!(Arc::strong_count(&reservation.0), 2);

        drop(reservation);
        assert_eq!(Arc::strong_count(&worker_reservation.0), 1);
    }

    #[test]
    fn reservation_is_reclaimed_on_drop() {
        let first = reserve(1000, 600).expect("first reservation");
        let second = reserve(1000, 600).expect("remaining reservation");
        assert_eq!(first.bytes(), 600);
        assert_eq!(second.bytes(), 400);
        drop(first);
        let replacement = reserve(1000, 600).expect("reclaimed reservation");
        assert_eq!(replacement.bytes(), 600);
    }
}
