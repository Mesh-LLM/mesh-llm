//! The L3 tier under the radix cache.
//!
//! `UnifiedRadixCache` holds `ExactStatePayload` entries in RAM (L1/L2);
//! this tier gives them a durable floor: spill a payload under its prefix
//! identity when the radix cache evicts it, and fill it back on a radix
//! miss — from local disk, or from a peer via `l3_remote` since both are
//! backends of the same `HandoffSegmentStore` contract. State never crosses
//! a numerical-identity boundary: spills stamp the tier's
//! `exact_state_identity`, and fills refuse manifests stamped differently.

use anyhow::{Context, Result, bail};

use crate::l3::{HandoffManifest, HandoffSegmentRef, HandoffSegmentStore, segment_digest};
use crate::payload::{ExactStatePayload, ExactStatePayloadKind};

/// Key an L3 entry by the radix coordinates that identify it in RAM: the
/// namespace (which already binds the numerical stage identity) and the
/// exact token path.
pub fn l3_prefix_key(namespace: &str, token_ids: &[i32]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"l3-prefix-key-v1");
    hasher.update(namespace.as_bytes());
    for token_id in token_ids {
        hasher.update(&token_id.to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

pub struct L3Tier {
    store: HandoffSegmentStore,
    state_identity: String,
    segment_bytes: usize,
}

impl L3Tier {
    pub fn open(
        root: impl Into<std::path::PathBuf>,
        budget_bytes: u64,
        state_identity: String,
        segment_bytes: usize,
    ) -> Result<Self> {
        Ok(Self {
            store: HandoffSegmentStore::open(root, budget_bytes)?,
            state_identity,
            segment_bytes: segment_bytes.max(1),
        })
    }

    pub fn store(&self) -> &HandoffSegmentStore {
        &self.store
    }

    /// Spill an evicted radix payload under its prefix identity. Returns the
    /// manifest key.
    pub fn spill(
        &self,
        prefix_key: &str,
        token_count: u64,
        payload: &ExactStatePayload,
        kv_desc_json: Option<String>,
    ) -> Result<String> {
        let (kv, recurrent): (Vec<u8>, Vec<u8>) = match payload.kind() {
            ExactStatePayloadKind::FullState => (
                payload
                    .full_state_bytes_timed()
                    .context("failed to reconstruct full state for spill")?
                    .0
                    .into_owned(),
                Vec::new(),
            ),
            ExactStatePayloadKind::RecurrentOnly => (
                Vec::new(),
                payload
                    .recurrent_state_bytes()
                    .context("failed to reconstruct recurrent state for spill")?
                    .into_owned(),
            ),
            ExactStatePayloadKind::KvRecurrent => (
                payload
                    .kv_bytes()
                    .context("failed to reconstruct KV bytes for spill")?
                    .map(|bytes| bytes.into_owned())
                    .unwrap_or_default(),
                payload
                    .recurrent_state_bytes()
                    .context("failed to reconstruct recurrent state for spill")?
                    .into_owned(),
            ),
        };

        let mut wire = Vec::with_capacity(kv.len() + recurrent.len());
        wire.extend_from_slice(&kv);
        wire.extend_from_slice(&recurrent);
        let payload_digest = segment_digest(&wire);

        let mut manifest = HandoffManifest::new(
            self.state_identity.clone(),
            payload.kind().as_str().to_string(),
        );
        manifest.total_bytes = wire.len() as u64;
        manifest.payload_digest = payload_digest.clone();
        manifest.kv_bytes = kv.len() as u64;
        manifest.recurrent_bytes = recurrent.len() as u64;
        manifest.kv_desc_json = kv_desc_json;
        manifest.token_count = token_count;
        for (index, chunk) in wire.chunks(self.segment_bytes).enumerate() {
            let (digest, _) = self.store.put_segment(chunk)?;
            manifest.segments.push(HandoffSegmentRef {
                index: index as u32,
                offset: (index * self.segment_bytes) as u64,
                bytes: chunk.len() as u64,
                digest,
                meta_json: None,
            });
        }
        self.store.commit(&manifest)?;
        self.store.link_prefix(prefix_key, &payload_digest)?;
        Ok(payload_digest)
    }

    /// Fill a radix miss from the tier: the payload, its token count, and
    /// the serialized KV page descriptor when one was spilled, or `None`
    /// when the tier has nothing for this prefix.
    pub fn fill(
        &self,
        prefix_key: &str,
    ) -> Result<Option<(ExactStatePayload, u64, Option<String>)>> {
        let Some(manifest) = self.store.manifest_for_prefix(prefix_key)? else {
            return Ok(None);
        };
        if manifest.state_identity != self.state_identity {
            bail!(
                "L3 entry for this prefix was spilled under state identity {} but the tier serves {}",
                manifest.state_identity,
                self.state_identity
            );
        }
        let wire = self.store.assemble(&manifest)?;
        let kv_bytes = usize::try_from(manifest.kv_bytes).context("kv bytes exceed usize")?;
        let payload = match manifest.payload_kind.as_str() {
            "full-state" => ExactStatePayload::full_state(wire),
            "recurrent-only" => ExactStatePayload::recurrent_only(wire),
            "kv-recurrent" => {
                let recurrent = wire[kv_bytes..].to_vec();
                let mut kv = wire;
                kv.truncate(kv_bytes);
                ExactStatePayload::kv_recurrent(kv, recurrent)
            }
            other => bail!("L3 manifest holds unknown payload kind {other}"),
        };
        Ok(Some((payload, manifest.token_count, manifest.kv_desc_json)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir()
            .join("skippy-l3-tier-tests")
            .join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    fn tier(name: &str, identity: &str) -> L3Tier {
        L3Tier::open(temp_root(name), 0, identity.to_string(), 4096).expect("open tier")
    }

    #[test]
    fn spill_and_fill_roundtrip_all_payload_kinds() {
        let tier = tier("roundtrip", "blake3:identity-a");
        let cases = vec![
            (
                "prefix-full",
                ExactStatePayload::full_state((0..50_000u32).map(|v| v as u8).collect()),
            ),
            (
                "prefix-recurrent",
                ExactStatePayload::recurrent_only(vec![9u8; 10_000]),
            ),
            (
                "prefix-kv",
                ExactStatePayload::kv_recurrent(vec![1u8; 20_000], vec![2u8; 5_000]),
            ),
        ];
        for (prefix, payload) in cases {
            tier.spill(prefix, 512, &payload, Some("{\"desc\":1}".to_string()))
                .expect("spill");
            let (filled, tokens, kv_desc_json) = tier
                .fill(prefix)
                .expect("fill")
                .expect("tier must hold the prefix");
            assert_eq!(tokens, 512);
            assert_eq!(kv_desc_json.as_deref(), Some("{\"desc\":1}"));
            assert_eq!(filled.kind(), payload.kind());
            match (payload.kind(), &filled) {
                (ExactStatePayloadKind::KvRecurrent, _) => {
                    assert_eq!(
                        filled.kv_bytes().unwrap().unwrap().into_owned(),
                        payload.kv_bytes().unwrap().unwrap().into_owned()
                    );
                    assert_eq!(
                        filled.recurrent_state_bytes().unwrap().into_owned(),
                        payload.recurrent_state_bytes().unwrap().into_owned()
                    );
                }
                (ExactStatePayloadKind::RecurrentOnly, _) => assert_eq!(
                    filled.recurrent_state_bytes().unwrap().into_owned(),
                    payload.recurrent_state_bytes().unwrap().into_owned()
                ),
                (ExactStatePayloadKind::FullState, _) => assert_eq!(
                    filled.full_state_bytes_timed().unwrap().0.into_owned(),
                    payload.full_state_bytes_timed().unwrap().0.into_owned()
                ),
            }
        }
    }

    #[test]
    fn unknown_prefix_fills_none() {
        let tier = tier("miss", "blake3:identity-a");
        assert!(tier.fill("never-spilled").expect("fill").is_none());
    }

    #[test]
    fn identity_mismatch_is_refused_not_served() {
        let root = temp_root("identity");
        let writer = L3Tier::open(&root, 0, "blake3:identity-a".to_string(), 4096).unwrap();
        writer
            .spill(
                "prefix",
                128,
                &ExactStatePayload::full_state(vec![5u8; 1024]),
                None,
            )
            .expect("spill");
        let reader = L3Tier::open(&root, 0, "blake3:identity-b".to_string(), 4096).unwrap();
        assert!(reader.fill("prefix").is_err());
    }

    #[test]
    fn respilling_a_prefix_supersedes_the_older_entry() {
        let tier = tier("supersede", "blake3:identity-a");
        tier.spill(
            "prefix",
            100,
            &ExactStatePayload::full_state(vec![1u8; 2048]),
            None,
        )
        .expect("first spill");
        tier.spill(
            "prefix",
            200,
            &ExactStatePayload::full_state(vec![2u8; 2048]),
            None,
        )
        .expect("second spill");
        let (filled, tokens, _) = tier.fill("prefix").expect("fill").expect("present");
        assert_eq!(tokens, 200);
        assert_eq!(
            filled.full_state_bytes_timed().unwrap().0.into_owned(),
            vec![2u8; 2048]
        );
    }
}
