//! What this client refuses to read of a claimed GGUF, and the error that
//! says so.
//!
//! The node that issued a receipt names the model, so the header, the
//! tensors and the sizes an audit reads are whatever that node chose to
//! publish. Every count, length and size of the file is checked against a
//! limit here before the client allocates or fetches for it. A limit that
//! trips means this client cannot audit the model. It is never a verdict on
//! the node: [`CannotAudit`] is the error for it.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;

/// The largest GGUF header, up to the first tensor's data, that is read.
/// Headers of real models are 8 to 16 MiB with the tokenizer.
pub(crate) const MAX_HEADER_BYTES: u64 = 256 << 20;
/// The most tensors a header may list.
pub(crate) const MAX_TENSORS: u64 = 1 << 16;
/// The most metadata entries a header may hold.
pub(crate) const MAX_METADATA_ENTRIES: u64 = 1 << 16;
/// The longest metadata key or tensor name, in bytes.
pub(crate) const MAX_NAME_BYTES: u64 = 1024;
/// The longest metadata string, in bytes.
pub(crate) const MAX_STRING_BYTES: u64 = 64 << 20;
/// How deep metadata arrays may nest: an array of arrays, and no deeper.
pub(crate) const MAX_ARRAY_DEPTH: u32 = 2;
/// The most dimensions of a tensor, as ggml has them.
pub(crate) const MAX_DIMS: usize = 4;
/// The most shards of a split GGUF.
pub(crate) const MAX_SHARDS: usize = 128;
/// The most values of one weight row that is read and decoded.
pub(crate) const MAX_ROW_VALUES: u64 = 1 << 24;
/// The largest tensor outside the layers that a sparse copy fetches whole.
pub(crate) const WHOLE_TENSOR_LIMIT: u64 = 256 << 20;
/// Bytes a [`WeightFile`] fetches from a remote file before it refuses,
/// unless the caller sets another budget. A head key build streams one
/// whole output matrix, which is a few GiB for the largest models.
///
/// [`WeightFile`]: crate::weights::WeightFile
pub(crate) const DEFAULT_FETCH_BUDGET: u64 = 16 << 30;

/// A claimed GGUF this client will not read: over a limit of this client,
/// or not a GGUF the runtime loads. It says nothing about the node.
#[derive(Debug)]
pub struct CannotAudit(String);

impl CannotAudit {
    pub fn new(reason: impl Into<String>) -> Self {
        Self(reason.into())
    }

    /// Whether `error` is, or was caused by, a refusal of this kind.
    pub fn caused(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| cause.is::<Self>())
    }
}

impl fmt::Display for CannotAudit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "this client cannot audit this model: {}; that is not a verdict on the node",
            self.0
        )
    }
}

impl std::error::Error for CannotAudit {}

/// Refuses the claimed GGUF for `reason`.
pub fn refuse<T>(reason: impl Into<String>) -> Result<T> {
    Err(CannotAudit::new(reason).into())
}

/// How many bytes a [`WeightFile`] may still fetch from a remote file.
///
/// [`WeightFile`]: crate::weights::WeightFile
#[derive(Debug)]
pub(crate) struct FetchBudget {
    limit: AtomicU64,
    spent: AtomicU64,
}

impl FetchBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            limit: AtomicU64::new(limit),
            spent: AtomicU64::new(0),
        }
    }

    /// Starts over with a budget of `limit` bytes.
    pub fn set(&self, limit: u64) {
        self.limit.store(limit, Ordering::SeqCst);
        self.spent.store(0, Ordering::SeqCst);
    }

    /// Starts over with the same budget.
    pub fn reset(&self) {
        self.spent.store(0, Ordering::SeqCst);
    }

    pub fn limit(&self) -> u64 {
        self.limit.load(Ordering::SeqCst)
    }

    pub fn spent(&self) -> u64 {
        self.spent.load(Ordering::SeqCst)
    }

    /// Takes `bytes` from the budget, or refuses when it does not hold them.
    pub fn charge(&self, bytes: u64) -> Result<()> {
        let limit = self.limit();
        let charged = self
            .spent
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |spent| {
                spent.checked_add(bytes).filter(|&total| total <= limit)
            });
        match charged {
            Ok(_) => Ok(()),
            Err(spent) => refuse(format!(
                "the audit would fetch {bytes} more bytes of it after {spent}, past the budget of {limit} bytes"
            )),
        }
    }
}

impl Default for FetchBudget {
    fn default() -> Self {
        Self::new(DEFAULT_FETCH_BUDGET)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_budget_refuses_what_it_does_not_hold() {
        let budget = FetchBudget::new(100);
        budget.charge(60).unwrap();
        let refused = budget.charge(41).unwrap_err();
        assert!(CannotAudit::caused(&refused), "{refused:#}");
        assert!(refused.to_string().contains("not a verdict on the node"));
        // what was refused is not spent
        budget.charge(40).unwrap();
        assert!(budget.charge(1).is_err());
        assert!(budget.charge(u64::MAX).is_err());
        budget.reset();
        budget.charge(100).unwrap();
        budget.set(10);
        assert_eq!((budget.limit(), budget.spent()), (10, 0));
        assert!(budget.charge(11).is_err());
    }

    #[test]
    fn a_refusal_is_found_under_context() {
        let refused = refuse::<()>("too large").unwrap_err().context("open it");
        assert!(CannotAudit::caused(&refused));
        assert!(!CannotAudit::caused(&anyhow::anyhow!("the network failed")));
    }
}
