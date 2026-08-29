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

/// The namespace's own index key.
pub fn l3_namespace_key(namespace: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"l3-namespace-key-v1");
    hasher.update(namespace.as_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

/// A successful fill from the tier.
pub struct L3Fill {
    pub payload: ExactStatePayload,
    /// How many of the query's leading tokens the filled state covers — the
    /// length the entry was recorded at, which may be shorter than the
    /// query (longest-recorded-prefix semantics, mirroring the radix).
    pub token_count: u64,
    pub kv_desc_json: Option<String>,
    pub payload_bytes: u64,
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

    /// Spill a radix payload under its (namespace, token-path) coordinates.
    /// Returns the manifest key. Entries at many lengths coexist — each is a
    /// complete state for its own length, which is what longest-prefix fill
    /// leans on.
    pub fn spill(
        &self,
        namespace: &str,
        token_ids: &[i32],
        payload: &ExactStatePayload,
        kv_desc_json: Option<String>,
    ) -> Result<String> {
        let token_count = token_ids.len() as u64;
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
        self.store.link_prefix(
            &l3_namespace_key(namespace),
            token_count,
            &l3_prefix_key(namespace, token_ids),
            &payload_digest,
        )?;
        Ok(payload_digest)
    }

    /// Fill a radix miss with the longest recorded prefix of the query,
    /// mirroring the radix cache's longest-component-prefix semantics: probe
    /// recorded lengths for this namespace from longest to shortest (capped
    /// at `max_probes`), hashing the query's own leading tokens at each
    /// length — so a recorded entry only matches when the query genuinely
    /// starts with the tokens it was recorded for.
    pub fn fill_longest(
        &self,
        namespace: &str,
        token_ids: &[i32],
        max_probes: usize,
    ) -> Result<Option<L3Fill>> {
        let namespace_key = l3_namespace_key(namespace);
        let query_len = token_ids.len() as u64;
        let lengths = self.store.recorded_prefix_lengths(&namespace_key)?;
        for length in lengths
            .into_iter()
            .filter(|length| *length > 0 && *length <= query_len)
            .take(max_probes.max(1))
        {
            let prefix_key = l3_prefix_key(namespace, &token_ids[..length as usize]);
            let Some(manifest) =
                self.store
                    .manifest_for_prefix(&namespace_key, length, &prefix_key)?
            else {
                continue;
            };
            if manifest.state_identity != self.state_identity {
                bail!(
                    "L3 entry for this prefix was spilled under state identity {} but the tier serves {}",
                    manifest.state_identity,
                    self.state_identity
                );
            }
            if manifest.token_count != length {
                bail!(
                    "L3 index length {length} disagrees with manifest token count {}",
                    manifest.token_count
                );
            }
            // Memory bound: `assemble` materializes the payload once
            // (`total_bytes`); the kv/recurrent split below reuses that
            // allocation via `split_off`, so peak extra memory is the
            // payload itself. Full-state fills are whole-blob by nature;
            // kv-recurrent fills could stream per segment later.
            let mut wire = self.store.assemble(&manifest)?;
            let payload_bytes = wire.len() as u64;
            let kv_bytes = usize::try_from(manifest.kv_bytes).context("kv bytes exceed usize")?;
            let payload = match manifest.payload_kind.as_str() {
                "full-state" => ExactStatePayload::full_state(wire),
                "recurrent-only" => ExactStatePayload::recurrent_only(wire),
                "kv-recurrent" => {
                    let recurrent = wire.split_off(kv_bytes);
                    ExactStatePayload::kv_recurrent(wire, recurrent)
                }
                other => bail!("L3 manifest holds unknown payload kind {other}"),
            };
            return Ok(Some(L3Fill {
                payload,
                token_count: manifest.token_count,
                kv_desc_json: manifest.kv_desc_json,
                payload_bytes,
            }));
        }
        Ok(None)
    }

    /// What the tier can restore right now: (manifest count, restorable
    /// token total, segment footprint bytes). Startup visibility so warm
    /// state is never invisible.
    pub fn restorable_summary(&self) -> Result<(usize, u64, u64)> {
        let keys = self.store.list_manifests()?;
        let mut tokens = 0u64;
        let mut count = 0usize;
        for key in &keys {
            if let Ok(manifest) = self.store.load_manifest(key)
                && manifest.state_identity == self.state_identity
            {
                tokens = tokens.saturating_add(manifest.token_count);
                count += 1;
            }
        }
        Ok((count, tokens, self.store.segment_footprint_bytes()?))
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

    fn tokens(len: usize) -> Vec<i32> {
        (0..len as i32).collect()
    }

    #[test]
    fn spill_and_fill_roundtrip_all_payload_kinds() {
        let tier = tier("roundtrip", "blake3:identity-a");
        let cases = vec![
            (
                "namespace-full",
                ExactStatePayload::full_state((0..50_000u32).map(|v| v as u8).collect()),
            ),
            (
                "namespace-recurrent",
                ExactStatePayload::recurrent_only(vec![9u8; 10_000]),
            ),
            (
                "namespace-kv",
                ExactStatePayload::kv_recurrent(vec![1u8; 20_000], vec![2u8; 5_000]),
            ),
        ];
        for (namespace, payload) in cases {
            tier.spill(
                namespace,
                &tokens(512),
                &payload,
                Some("{\"desc\":1}".to_string()),
            )
            .expect("spill");
            let fill = tier
                .fill_longest(namespace, &tokens(512), 64)
                .expect("fill")
                .expect("tier must hold the prefix");
            assert_eq!(fill.token_count, 512);
            assert_eq!(fill.kv_desc_json.as_deref(), Some("{\"desc\":1}"));
            assert_eq!(fill.payload.kind(), payload.kind());
            match payload.kind() {
                ExactStatePayloadKind::KvRecurrent => {
                    assert_eq!(
                        fill.payload.kv_bytes().unwrap().unwrap().into_owned(),
                        payload.kv_bytes().unwrap().unwrap().into_owned()
                    );
                    assert_eq!(
                        fill.payload.recurrent_state_bytes().unwrap().into_owned(),
                        payload.recurrent_state_bytes().unwrap().into_owned()
                    );
                }
                ExactStatePayloadKind::RecurrentOnly => assert_eq!(
                    fill.payload.recurrent_state_bytes().unwrap().into_owned(),
                    payload.recurrent_state_bytes().unwrap().into_owned()
                ),
                ExactStatePayloadKind::FullState => assert_eq!(
                    fill.payload
                        .full_state_bytes_timed()
                        .unwrap()
                        .0
                        .into_owned(),
                    payload.full_state_bytes_timed().unwrap().0.into_owned()
                ),
            }
        }
    }

    /// The sacrament case: a later, longer prompt (multi-turn growth) must
    /// find the longest recorded shorter prefix, not just an exact match.
    #[test]
    fn longer_query_fills_from_longest_recorded_prefix() {
        let tier = tier("longest", "blake3:identity-a");
        tier.spill(
            "ns",
            &tokens(800),
            &ExactStatePayload::full_state(vec![1u8; 2048]),
            None,
        )
        .expect("spill 800");
        tier.spill(
            "ns",
            &tokens(1200),
            &ExactStatePayload::full_state(vec![2u8; 2048]),
            None,
        )
        .expect("spill 1200");

        // Query extends the 1200-token path: the longest entry wins.
        let fill = tier
            .fill_longest("ns", &tokens(1900), 64)
            .expect("fill")
            .expect("hit");
        assert_eq!(fill.token_count, 1200);
        assert_eq!(
            fill.payload
                .full_state_bytes_timed()
                .unwrap()
                .0
                .into_owned(),
            vec![2u8; 2048]
        );

        // Query between the two recorded lengths: the shorter entry wins.
        let fill = tier
            .fill_longest("ns", &tokens(1000), 64)
            .expect("fill")
            .expect("hit");
        assert_eq!(fill.token_count, 800);
    }

    /// A recorded length only matches when the query genuinely starts with
    /// the recorded tokens — a divergent prompt of the same length must
    /// miss, not corrupt.
    #[test]
    fn divergent_tokens_at_a_recorded_length_miss() {
        let tier = tier("divergent", "blake3:identity-a");
        tier.spill(
            "ns",
            &tokens(600),
            &ExactStatePayload::full_state(vec![3u8; 1024]),
            None,
        )
        .expect("spill");
        let mut divergent = tokens(600);
        divergent[100] = 999_999;
        assert!(
            tier.fill_longest("ns", &divergent, 64)
                .expect("fill")
                .is_none()
        );
    }

    #[test]
    fn unknown_namespace_fills_none() {
        let tier = tier("miss", "blake3:identity-a");
        assert!(
            tier.fill_longest("never-spilled", &tokens(64), 64)
                .expect("fill")
                .is_none()
        );
    }

    #[test]
    fn identity_mismatch_is_refused_not_served() {
        let root = temp_root("identity");
        let writer = L3Tier::open(&root, 0, "blake3:identity-a".to_string(), 4096).unwrap();
        writer
            .spill(
                "ns",
                &tokens(128),
                &ExactStatePayload::full_state(vec![5u8; 1024]),
                None,
            )
            .expect("spill");
        let reader = L3Tier::open(&root, 0, "blake3:identity-b".to_string(), 4096).unwrap();
        assert!(reader.fill_longest("ns", &tokens(128), 64).is_err());
    }

    #[test]
    fn respilling_a_length_supersedes_the_older_entry() {
        let tier = tier("supersede", "blake3:identity-a");
        tier.spill(
            "ns",
            &tokens(100),
            &ExactStatePayload::full_state(vec![1u8; 2048]),
            None,
        )
        .expect("first spill");
        tier.spill(
            "ns",
            &tokens(100),
            &ExactStatePayload::full_state(vec![2u8; 2048]),
            None,
        )
        .expect("second spill");
        let fill = tier
            .fill_longest("ns", &tokens(100), 64)
            .expect("fill")
            .expect("present");
        assert_eq!(fill.token_count, 100);
        assert_eq!(
            fill.payload
                .full_state_bytes_timed()
                .unwrap()
                .0
                .into_owned(),
            vec![2u8; 2048]
        );
    }

    #[test]
    fn restorable_summary_counts_matching_identity_only() {
        let root = temp_root("summary");
        let tier_a = L3Tier::open(&root, 0, "blake3:identity-a".to_string(), 4096).unwrap();
        let tier_b = L3Tier::open(&root, 0, "blake3:identity-b".to_string(), 4096).unwrap();
        tier_a
            .spill(
                "ns",
                &tokens(300),
                &ExactStatePayload::full_state(vec![1u8; 512]),
                None,
            )
            .unwrap();
        tier_b
            .spill(
                "ns",
                &tokens(700),
                &ExactStatePayload::full_state(vec![2u8; 512]),
                None,
            )
            .unwrap();
        let (count, restorable_tokens, footprint) = tier_a.restorable_summary().unwrap();
        assert_eq!(count, 1);
        assert_eq!(restorable_tokens, 300);
        assert!(footprint >= 1024);
    }
}
