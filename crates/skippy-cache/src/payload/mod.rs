use std::{borrow::Cow, fmt};

use anyhow::{Result, anyhow};

mod blob_store;
pub(super) mod bytes;

pub use blob_store::{CacheBlobStore, CacheDedupeStats};
pub use bytes::{CacheBytes, CacheBytesReconstructStats};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactStatePayloadKind {
    FullState,
    RecurrentOnly,
    KvRecurrent,
    /// Attention KV exported from a dense family's resident prefix.
    ///
    /// Distinct from [`Self::KvRecurrent`] on purpose. A dense archive has no
    /// recurrent state at all, and encoding it as a KV-recurrent payload with
    /// an empty second component would let a future caller read a zero-length
    /// recurrent state back as if it were real. The payload kind is the only
    /// thing preventing cross-kind reinterpretation on the disk tier, so it
    /// must describe the payload honestly.
    ResidentKvArchive,
}

#[derive(Debug, Clone)]
pub enum ExactStatePayload {
    FullState {
        bytes: CacheBytes,
    },
    RecurrentOnly {
        recurrent: CacheBytes,
    },
    KvRecurrent {
        kv: CacheBytes,
        recurrent: CacheBytes,
    },
    ResidentKvArchive {
        kv: CacheBytes,
    },
}

impl ExactStatePayload {
    pub fn full_state(bytes: Vec<u8>) -> Self {
        Self::FullState {
            bytes: CacheBytes::inline(bytes),
        }
    }

    pub fn recurrent_only(recurrent: Vec<u8>) -> Self {
        Self::RecurrentOnly {
            recurrent: CacheBytes::inline(recurrent),
        }
    }

    pub fn kv_recurrent(kv: Vec<u8>, recurrent: Vec<u8>) -> Self {
        Self::KvRecurrent {
            kv: CacheBytes::inline(kv),
            recurrent: CacheBytes::inline(recurrent),
        }
    }

    pub fn resident_kv_archive(kv: Vec<u8>) -> Self {
        Self::ResidentKvArchive {
            kv: CacheBytes::inline(kv),
        }
    }

    /// Components in the order they are written to the disk tier.
    ///
    /// Returns an error when any component cannot be borrowed contiguously.
    pub fn disk_components(&self) -> Result<Vec<Cow<'_, [u8]>>> {
        Ok(match self {
            Self::FullState { bytes } => vec![bytes.as_cow()?],
            Self::RecurrentOnly { recurrent } => vec![recurrent.as_cow()?],
            Self::KvRecurrent { kv, recurrent } => vec![kv.as_cow()?, recurrent.as_cow()?],
            Self::ResidentKvArchive { kv } => vec![kv.as_cow()?],
        })
    }

    /// Rebuild a payload from components restored off the disk tier.
    ///
    /// The component count is part of the payload contract, so a mismatch
    /// means the entry is not what the caller believes it is and must be
    /// rejected rather than padded or truncated.
    pub fn from_disk_components(
        kind: ExactStatePayloadKind,
        mut components: Vec<CacheBytes>,
    ) -> Result<Self> {
        let expected = kind.disk_component_count();
        if components.len() != expected {
            return Err(anyhow!(
                "cached {kind} payload expects {expected} components, got {}",
                components.len()
            ));
        }
        Ok(match kind {
            ExactStatePayloadKind::FullState => Self::FullState {
                bytes: components.remove(0),
            },
            ExactStatePayloadKind::ResidentKvArchive => Self::ResidentKvArchive {
                kv: components.remove(0),
            },
            ExactStatePayloadKind::RecurrentOnly => Self::RecurrentOnly {
                recurrent: components.remove(0),
            },
            ExactStatePayloadKind::KvRecurrent => {
                let kv = components.remove(0);
                let recurrent = components.remove(0);
                Self::KvRecurrent { kv, recurrent }
            }
        })
    }

    /// Whether every component is backed by the disk tier.
    pub fn is_mapped(&self) -> bool {
        match self {
            Self::FullState { bytes } => bytes.is_mapped(),
            Self::RecurrentOnly { recurrent } => recurrent.is_mapped(),
            Self::KvRecurrent { kv, recurrent } => kv.is_mapped() && recurrent.is_mapped(),
            Self::ResidentKvArchive { kv } => kv.is_mapped(),
        }
    }

    pub fn kind(&self) -> ExactStatePayloadKind {
        match self {
            Self::FullState { .. } => ExactStatePayloadKind::FullState,
            Self::RecurrentOnly { .. } => ExactStatePayloadKind::RecurrentOnly,
            Self::KvRecurrent { .. } => ExactStatePayloadKind::KvRecurrent,
            Self::ResidentKvArchive { .. } => ExactStatePayloadKind::ResidentKvArchive,
        }
    }

    pub fn byte_len(&self) -> u64 {
        match self {
            Self::FullState { bytes } => bytes.len(),
            Self::RecurrentOnly { recurrent } => recurrent.len(),
            Self::KvRecurrent { kv, recurrent } => kv.len().saturating_add(recurrent.len()),
            Self::ResidentKvArchive { kv } => kv.len(),
        }
    }

    pub fn recurrent_state_bytes(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            Self::RecurrentOnly { recurrent } | Self::KvRecurrent { recurrent, .. } => {
                recurrent.as_cow()
            }
            _ => Err(anyhow!("cache payload has no recurrent component")),
        }
    }

    pub fn recurrent_state_bytes_timed(
        &self,
    ) -> Result<(Cow<'_, [u8]>, CacheBytesReconstructStats)> {
        match self {
            Self::RecurrentOnly { recurrent } | Self::KvRecurrent { recurrent, .. } => {
                recurrent.as_cow_timed()
            }
            _ => Err(anyhow!("cache payload has no recurrent component")),
        }
    }

    pub fn full_state_bytes_timed(&self) -> Result<(Cow<'_, [u8]>, CacheBytesReconstructStats)> {
        match self {
            Self::FullState { bytes } => bytes.as_cow_timed(),
            _ => Err(anyhow!("cache payload is not full-state")),
        }
    }

    pub fn kv_bytes(&self) -> Result<Option<Cow<'_, [u8]>>> {
        match self {
            Self::KvRecurrent { kv, .. } | Self::ResidentKvArchive { kv } => Ok(Some(kv.as_cow()?)),
            _ => Ok(None),
        }
    }

    pub fn kv_bytes_timed(&self) -> Result<Option<(Cow<'_, [u8]>, CacheBytesReconstructStats)>> {
        match self {
            Self::KvRecurrent { kv, .. } | Self::ResidentKvArchive { kv } => {
                Ok(Some(kv.as_cow_timed()?))
            }
            _ => Ok(None),
        }
    }

    pub fn dedupe_into(self, blobs: &mut CacheBlobStore) -> (Self, CacheDedupeStats) {
        match self {
            Self::FullState { bytes } => {
                let (bytes, stats) = blobs.store_bytes(bytes);
                (Self::FullState { bytes }, stats)
            }
            Self::RecurrentOnly { recurrent } => {
                let (recurrent, stats) = blobs.store_bytes(recurrent);
                (Self::RecurrentOnly { recurrent }, stats)
            }
            Self::KvRecurrent { kv, recurrent } => {
                let (kv, kv_stats) = blobs.store_bytes(kv);
                let (recurrent, recurrent_stats) = blobs.store_bytes(recurrent);
                (
                    Self::KvRecurrent { kv, recurrent },
                    kv_stats.saturating_add(recurrent_stats),
                )
            }
            Self::ResidentKvArchive { kv } => {
                let (kv, stats) = blobs.store_bytes(kv);
                (Self::ResidentKvArchive { kv }, stats)
            }
        }
    }

    pub fn release_from(&self, blobs: &mut CacheBlobStore) {
        match self {
            Self::FullState { bytes } => blobs.release_bytes(bytes),
            Self::RecurrentOnly { recurrent } => blobs.release_bytes(recurrent),
            Self::KvRecurrent { kv, recurrent } => {
                blobs.release_bytes(kv);
                blobs.release_bytes(recurrent);
            }
            Self::ResidentKvArchive { kv } => blobs.release_bytes(kv),
        }
    }
}

impl ExactStatePayloadKind {
    /// Stable identifier persisted with disk-tier entries. Changing any of
    /// these strings must come with a disk-tier format-version bump.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullState => "full-state",
            Self::RecurrentOnly => "recurrent-only",
            Self::KvRecurrent => "kv-recurrent",
            Self::ResidentKvArchive => "resident-kv-archive",
        }
    }

    /// How many byte components this payload serializes into.
    pub fn disk_component_count(self) -> usize {
        match self {
            Self::FullState | Self::RecurrentOnly | Self::ResidentKvArchive => 1,
            Self::KvRecurrent => 2,
        }
    }
}

impl fmt::Display for ExactStatePayloadKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
