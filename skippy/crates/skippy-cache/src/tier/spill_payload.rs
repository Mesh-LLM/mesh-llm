//! Borrow cold spill components without constructing a second complete snapshot.
use std::borrow::Cow;

use anyhow::{Context, Result, bail, ensure};

use super::CacheGenKvPayload;
use crate::payload::{ExactStatePayload, ExactStatePayloadKind};

pub(super) struct SpillPayload<'a> {
    kv: Cow<'a, [u8]>,
    recurrent: Cow<'a, [u8]>,
    total_bytes: usize,
    pub(super) decoded_kv_bytes: u64,
}

impl<'a> SpillPayload<'a> {
    pub(super) fn new(
        payload: &'a ExactStatePayload,
        cachegen: Option<&mut CacheGenKvPayload>,
        descriptor: Option<&str>,
    ) -> Result<Self> {
        ensure!(
            payload.byte_len() > 0,
            "refusing to spill an empty exact-state payload: no state component was exported"
        );
        let (kv, recurrent) = match payload.kind() {
            ExactStatePayloadKind::FullState => {
                (payload.full_state_bytes_timed()?.0, Cow::Borrowed(&[][..]))
            }
            ExactStatePayloadKind::RecurrentOnly => {
                (Cow::Borrowed(&[][..]), payload.recurrent_state_bytes()?)
            }
            ExactStatePayloadKind::KvRecurrent => {
                let kv = payload.kv_bytes()?.unwrap_or(Cow::Borrowed(&[]));
                ensure!(
                    !kv.is_empty(),
                    "refusing to spill kv-recurrent state without a KV component"
                );
                (kv, payload.recurrent_state_bytes()?)
            }
        };
        let decoded_kv_bytes = kv.len() as u64;
        let kv = if let Some(cachegen) = cachegen {
            validate_cachegen(payload.kind(), cachegen, decoded_kv_bytes, descriptor)?;
            drop(kv);
            Cow::Owned(std::mem::take(&mut cachegen.archive))
        } else {
            kv
        };
        let total_bytes = kv
            .len()
            .checked_add(recurrent.len())
            .context("spill payload length overflows usize")?;
        Ok(Self {
            kv,
            recurrent,
            total_bytes,
            decoded_kv_bytes,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.total_bytes
    }

    pub(super) fn kv_bytes(&self) -> u64 {
        self.kv.len() as u64
    }

    pub(super) fn recurrent_bytes(&self) -> u64 {
        self.recurrent.len() as u64
    }

    pub(super) fn digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.kv);
        hasher.update(&self.recurrent);
        hasher.finalize().to_hex().to_string()
    }

    /// Ordinary segments borrow one immutable component. Only a cut spanning
    /// their boundary needs scratch storage, bounded by the configured segment.
    pub(super) fn segment(
        &self,
        offset: u64,
        bytes: u64,
        max_scratch_bytes: usize,
    ) -> Result<Cow<'_, [u8]>> {
        let start = usize::try_from(offset).context("segment offset exceeds usize")?;
        let len = usize::try_from(bytes).context("segment length exceeds usize")?;
        let end = start.checked_add(len).context("segment range overflows")?;
        ensure!(end <= self.total_bytes, "segment exceeds spill payload");
        if end <= self.kv.len() {
            return Ok(Cow::Borrowed(&self.kv[start..end]));
        }
        if start >= self.kv.len() {
            return Ok(Cow::Borrowed(
                &self.recurrent[start - self.kv.len()..end - self.kv.len()],
            ));
        }
        ensure!(
            len <= max_scratch_bytes,
            "cross-component segment exceeds the spill scratch budget"
        );
        let mut segment = Vec::with_capacity(len);
        segment.extend_from_slice(&self.kv[start..]);
        segment.extend_from_slice(&self.recurrent[..end - self.kv.len()]);
        Ok(Cow::Owned(segment))
    }
}

fn validate_cachegen(
    kind: ExactStatePayloadKind,
    archive: &CacheGenKvPayload,
    decoded_bytes: u64,
    descriptor: Option<&str>,
) -> Result<()> {
    if kind != ExactStatePayloadKind::KvRecurrent
        || archive.archive.is_empty()
        || archive.decoded_len != decoded_bytes
        || archive.calibration_digest.is_empty()
        || descriptor.is_none_or(str::is_empty)
    {
        bail!("invalid CacheGen spill: archive, descriptor, and decoded KV length are required");
    }
    let decoded_len =
        usize::try_from(archive.decoded_len).context("CacheGen decoded KV length exceeds usize")?;
    crate::cachegen::archive::validate_archive(&archive.archive, decoded_len)
        .context("invalid CacheGen archive supplied for L3 spill")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l3::segment_digest;

    #[test]
    fn component_stream_digest_matches_concatenation_for_every_boundary() {
        let bytes: Vec<u8> = (0..8195).map(|n| (n % 251) as u8).collect();
        for split in [0, 1, 63, 64, 1023, 1024, 4096, 8194, 8195] {
            let view = SpillPayload {
                kv: Cow::Borrowed(&bytes[..split]),
                recurrent: Cow::Borrowed(&bytes[split..]),
                total_bytes: bytes.len(),
                decoded_kv_bytes: split as u64,
            };
            assert_eq!(view.digest(), segment_digest(&bytes));
        }
    }

    #[test]
    fn native_components_and_boundary_aligned_segments_borrow_original_storage() {
        let payload = ExactStatePayload::kv_recurrent(vec![1; 4096], vec![2; 128]);
        let view = SpillPayload::new(&payload, None, None).unwrap();
        assert!(matches!(view.kv, Cow::Borrowed(_)));
        assert!(matches!(view.recurrent, Cow::Borrowed(_)));
        for (offset, bytes) in [(0, 2048), (2048, 2048), (4096, 128)] {
            assert!(matches!(
                view.segment(offset, bytes, 2048).unwrap(),
                Cow::Borrowed(_)
            ));
        }
        let kv = payload.kv_bytes().unwrap().unwrap();
        assert_eq!(view.segment(0, 4096, 2048).unwrap().as_ptr(), kv.as_ptr());
    }

    #[test]
    fn cross_component_cut_materializes_only_the_bounded_segment() {
        let payload = ExactStatePayload::kv_recurrent(vec![1, 2, 3], vec![4, 5, 6]);
        let view = SpillPayload::new(&payload, None, None).unwrap();
        let segment = view.segment(2, 3, 3).unwrap();
        assert!(matches!(segment, Cow::Owned(_)));
        assert_eq!(segment.as_ref(), &[3, 4, 5]);
        assert!(view.segment(2, 3, 2).is_err());
        assert!(view.segment(5, 2, 10).is_err());
        assert!(view.segment(u64::MAX, 2, 10).is_err());
    }

    #[test]
    fn single_component_payloads_and_empty_boundary_segments_do_not_copy() {
        for payload in [
            ExactStatePayload::full_state(vec![7, 8, 9]),
            ExactStatePayload::recurrent_only(vec![7, 8, 9]),
            ExactStatePayload::kv_recurrent(vec![7, 8, 9], Vec::new()),
        ] {
            let view = SpillPayload::new(&payload, None, None).unwrap();
            assert_eq!(view.digest(), segment_digest(&[7, 8, 9]));
            assert_eq!(view.len(), 3);
            assert!(matches!(view.segment(0, 3, 1).unwrap(), Cow::Borrowed(_)));
            assert!(matches!(view.segment(3, 0, 1).unwrap(), Cow::Borrowed(_)));
        }
        assert!(SpillPayload::new(&ExactStatePayload::full_state(Vec::new()), None, None).is_err());
    }
}
