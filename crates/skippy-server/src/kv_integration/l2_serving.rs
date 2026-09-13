use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use skippy_cache::{L2Hit, L2Origin, L2Stats, L2Tier, l2_cache_key};
use skippy_runtime::RuntimeKvPageDesc;

use crate::runtime_state::RuntimeState;

use super::{
    ExactStateExtra, ExactStateRecordAdmission, ExactStateRestore, KvStageIntegration,
    PendingExactStateRecord, PrefillKvIdentity, StagePrefixCachePayload,
    records::add_reconstruct_stats,
};

const PROMOTION_MAX_BYTES: u64 = 64 * 1024 * 1024;
const RESTORE_WORTH_TOKENS: u64 = 4_096;
const SECOND_HIT_WINDOW: Duration = Duration::from_secs(10 * 60);
const MAX_PROMOTION_OBSERVATIONS: usize = 4_096;

#[derive(Clone, Copy)]
struct PromotionObservation {
    first_seen: Instant,
    hits: u8,
    terminal: bool,
}

/// Stage-scoped view of the host-RAM tier. The identities are fixed when the
/// model loads, so every lookup and promotion uses the same numerical boundary
/// as L3 without exposing model or prompt fingerprints in telemetry.
#[derive(Clone)]
pub(crate) struct StageL2 {
    tier: Arc<L2Tier>,
    model_identity: String,
    state_identity: String,
    promotions: Arc<Mutex<HashMap<String, PromotionObservation>>>,
}

impl StageL2 {
    pub(crate) fn new(budget_bytes: u64, model_identity: String, state_identity: String) -> Self {
        Self {
            tier: Arc::new(L2Tier::new(budget_bytes)),
            model_identity,
            state_identity,
            promotions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub(crate) fn get(
        &self,
        identity: &PrefillKvIdentity,
        location: &skippy_cache::L3Location,
    ) -> Option<(String, L2Hit)> {
        let token_count = usize::try_from(location.token_count).ok()?;
        let tokens = identity.token_ids.get(..token_count)?;
        let key = self.cache_key(&identity.namespace, tokens);
        let hit = self.tier.get(&key)?;
        if hit.token_count != location.token_count || hit.payload_digest != location.manifest_key {
            self.remove(&key);
            return None;
        }
        Some((key, hit))
    }

    pub(crate) fn remove(&self, key: &str) {
        let _ = self.tier.remove(key);
    }

    pub(crate) fn invalidate_digest(&self, payload_digest: &str) {
        let _ = self.tier.remove_by_digest(payload_digest);
    }

    pub(crate) fn consider_l3_fill(
        &self,
        manifest_key: &str,
        token_count: u64,
        payload_bytes: u64,
    ) -> bool {
        self.should_promote(manifest_key, token_count, payload_bytes)
    }

    pub(crate) fn promote(
        &self,
        namespace: &str,
        tokens: &[i32],
        expected_payload_digest: &str,
        payload: &skippy_cache::ExactStatePayload,
        extra: &ExactStateExtra,
    ) -> bool {
        let key = self.cache_key(namespace, tokens);
        let kv_desc_json = extra
            .kv_desc
            .as_ref()
            .and_then(|desc| serde_json::to_string(desc).ok());
        let admitted = self
            .tier
            .admit_payload(
                key,
                tokens.len() as u64,
                expected_payload_digest,
                payload,
                kv_desc_json,
                L2Origin::FromL3,
            )
            .is_ok();
        if let Some(observation) = self
            .promotions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_mut(expected_payload_digest)
        {
            observation.terminal = true;
        }
        admitted
    }

    pub(crate) fn stats(&self) -> L2Stats {
        self.tier.stats()
    }

    fn cache_key(&self, namespace: &str, tokens: &[i32]) -> String {
        l2_cache_key(
            &self.model_identity,
            &self.state_identity,
            namespace,
            tokens,
        )
    }

    fn should_promote(&self, manifest_key: &str, token_count: u64, payload_bytes: u64) -> bool {
        let now = Instant::now();
        let mut observations = self
            .promotions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        observations.retain(|_, observation| {
            now.saturating_duration_since(observation.first_seen) <= SECOND_HIT_WINDOW
        });
        if observations.len() >= MAX_PROMOTION_OBSERVATIONS
            && !observations.contains_key(manifest_key)
            && let Some(oldest) = observations
                .iter()
                .min_by(|left, right| {
                    left.1
                        .first_seen
                        .cmp(&right.1.first_seen)
                        .then_with(|| left.0.cmp(right.0))
                })
                .map(|(key, _)| key.clone())
        {
            observations.remove(&oldest);
        }
        let observation =
            observations
                .entry(manifest_key.to_string())
                .or_insert(PromotionObservation {
                    first_seen: now,
                    hits: 0,
                    terminal: false,
                });
        if observation.terminal {
            return false;
        }
        observation.hits = observation.hits.saturating_add(1);
        if payload_bytes > PROMOTION_MAX_BYTES {
            observation.terminal = true;
            return false;
        }
        // Keep the observation eligible until the worker has attempted the
        // promotion. If the bounded worker queue drops this fill, the next L3
        // restore may retry instead of suppressing the entry for ten minutes.
        token_count >= RESTORE_WORTH_TOKENS || observation.hits >= 2
    }
}

impl KvStageIntegration {
    pub(super) fn restore_from_l2(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
        location: &skippy_cache::L3Location,
        lookup_started: Instant,
    ) -> Result<Option<ExactStateRestore>> {
        let Some(l2) = &self.l2 else {
            return Ok(None);
        };
        let Some((key, hit)) = l2.get(identity, location) else {
            return Ok(None);
        };
        let token_count = hit.token_count;
        if token_count == 0 || token_count != location.token_count {
            l2.remove(&key);
            return Ok(None);
        }
        let payload = hit.to_payload();
        let kv_desc = match hit.payload.kv_desc_json() {
            Some(json) => match serde_json::from_str::<RuntimeKvPageDesc>(json) {
                Ok(desc) => Some(desc),
                Err(_) => {
                    l2.remove(&key);
                    return Ok(None);
                }
            },
            None => None,
        };
        let fill_ms = lookup_started.elapsed().as_secs_f64() * 1000.0;
        let mut reconstruct_ms = 0.0;
        let mut reconstruct_bytes = 0u64;
        let mut reconstruct_blocks = 0usize;
        let mut kv_import_ms = 0.0;
        let mut recurrent_import_ms = 0.0;
        let mut deterministic_failure = false;
        let restore = (|| -> Result<bool> {
            match payload.kind().into() {
                StagePrefixCachePayload::FullState => {
                    let (bytes, stats) = payload
                        .full_state_bytes_timed()
                        .context("reconstruct L2 full-state payload")
                        .map_err(|error| mark_failure(&mut deterministic_failure, error))?;
                    if bytes.is_empty() {
                        deterministic_failure = true;
                        anyhow::bail!("L2 full-state payload is empty");
                    }
                    add_reconstruct_stats(
                        &mut reconstruct_ms,
                        &mut reconstruct_bytes,
                        &mut reconstruct_blocks,
                        stats,
                    );
                    let started = Instant::now();
                    runtime.import_full_state_for_token_count(
                        session_id,
                        bytes.as_ref(),
                        token_count,
                    )?;
                    kv_import_ms = started.elapsed().as_secs_f64() * 1000.0;
                }
                StagePrefixCachePayload::KvRecurrent => {
                    if let Some((kv, stats)) = payload
                        .kv_bytes_timed()
                        .context("reconstruct L2 KV payload")
                        .map_err(|error| mark_failure(&mut deterministic_failure, error))?
                    {
                        add_reconstruct_stats(
                            &mut reconstruct_ms,
                            &mut reconstruct_bytes,
                            &mut reconstruct_blocks,
                            stats,
                        );
                        match kv_desc.as_ref() {
                            Some(desc) => {
                                desc.validate_payload(kv.len()).map_err(|error| {
                                    mark_failure(&mut deterministic_failure, error)
                                })?;
                                if desc.token_start != 0 || desc.token_count != token_count {
                                    deterministic_failure = true;
                                    anyhow::bail!("L2 KV page token range mismatch");
                                }
                                let started = Instant::now();
                                runtime.import_kv_page(session_id, desc, kv.as_ref())?;
                                kv_import_ms = started.elapsed().as_secs_f64() * 1000.0;
                            }
                            None if !kv.is_empty() => {
                                deterministic_failure = true;
                                anyhow::bail!("L2 KV payload is missing its descriptor");
                            }
                            None => {}
                        }
                    }
                    let (recurrent, stats) = payload
                        .recurrent_state_bytes_timed()
                        .context("reconstruct L2 recurrent payload")
                        .map_err(|error| mark_failure(&mut deterministic_failure, error))?;
                    if recurrent.is_empty() && !self.dense_without_recurrent {
                        deterministic_failure = true;
                        anyhow::bail!("L2 recurrent-state payload is empty");
                    }
                    add_reconstruct_stats(
                        &mut reconstruct_ms,
                        &mut reconstruct_bytes,
                        &mut reconstruct_blocks,
                        stats,
                    );
                    let started = Instant::now();
                    if recurrent.is_empty() {
                        runtime.set_session_position(session_id, token_count)?;
                    } else {
                        runtime.import_recurrent_state_for_token_count(
                            session_id,
                            recurrent.as_ref(),
                            token_count,
                        )?;
                    }
                    recurrent_import_ms = started.elapsed().as_secs_f64() * 1000.0;
                }
                StagePrefixCachePayload::Disabled | StagePrefixCachePayload::ResidentKv => {
                    return Ok(false);
                }
            }
            Ok(true)
        })();
        let restored = match restore {
            Ok(restored) => restored,
            Err(error) => {
                if deterministic_failure {
                    l2.remove(&key);
                }
                return Err(error);
            }
        };
        if !restored {
            return Ok(None);
        }
        let logical_bytes = payload.byte_len();
        let payload_kind = payload.kind();
        let rewarm_enqueued = self.try_begin_record(&identity.page_id)
            && matches!(
                self.enqueue_exact_state_record(PendingExactStateRecord {
                    page_id: identity.page_id.clone(),
                    payload,
                    extra: ExactStateExtra { kv_desc },
                    namespace: identity.namespace.clone(),
                    token_ids: identity.token_ids[..token_count as usize].to_vec(),
                    l3_fill_claim: None,
                    write_through_l3: false,
                    l2_promotion_digest: None,
                    l3_cost: None,
                }),
                ExactStateRecordAdmission::Queued
            );
        Ok(Some(ExactStateRestore {
            page_id: identity.page_id.clone(),
            token_count: token_count as usize,
            payload_kind,
            logical_bytes,
            entries: l2.stats().entries as usize,
            reconstruct_ms,
            reconstruct_bytes,
            reconstruct_blocks,
            lookup_ms: fill_ms,
            kv_import_ms,
            recurrent_import_ms,
            source: "l2",
            fill_ms,
            rewarm_enqueued,
        }))
    }
}

fn mark_failure(deterministic: &mut bool, error: anyhow::Error) -> anyhow::Error {
    *deterministic = true;
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(tokens: Vec<i32>) -> PrefillKvIdentity {
        PrefillKvIdentity {
            identity: crate::kv_proto::PageIdentity::default(),
            page_id: "page".to_string(),
            namespace: "namespace".to_string(),
            token_ids: tokens,
        }
    }

    fn location(token_count: u64, manifest_key: String) -> skippy_cache::L3Location {
        skippy_cache::L3Location {
            namespace_key: "namespace-key".to_string(),
            prefix_key: "prefix-key".to_string(),
            token_count,
            manifest_key,
            kv_desc_json: None,
            kv_bytes: 0,
            native_kv_passthrough: false,
        }
    }

    #[test]
    fn second_l3_fill_promotes_and_serves_the_located_prefix() {
        let l2 = StageL2::new(1 << 20, "model".to_string(), "state".to_string());
        let bytes = vec![7; 128];
        let digest = skippy_cache::segment_digest(&bytes);
        let payload = skippy_cache::ExactStatePayload::full_state(bytes);
        let identity = identity(vec![1, 2, 3, 4]);
        let location = location(3, digest.clone());

        assert!(!l2.consider_l3_fill(&digest, 3, payload.byte_len()));
        assert!(l2.consider_l3_fill(&digest, 3, payload.byte_len()));
        assert!(l2.promote(
            &identity.namespace,
            &identity.token_ids[..3],
            &digest,
            &payload,
            &ExactStateExtra { kv_desc: None },
        ));
        assert!(!l2.consider_l3_fill(&digest, 3, payload.byte_len()));

        let (_, hit) = l2
            .get(&identity, &location)
            .expect("longer query should hit the located L3 prefix mirror");
        assert_eq!(hit.token_count, 3);
        assert_eq!(hit.payload_digest, digest);
    }

    #[test]
    fn restart_class_prefix_promotes_on_first_fill_but_oversized_payload_does_not() {
        let l2 = StageL2::new(1 << 20, "model".to_string(), "state".to_string());
        assert!(l2.consider_l3_fill("restart", RESTORE_WORTH_TOKENS, 1 << 20));
        assert!(!l2.consider_l3_fill("oversized", RESTORE_WORTH_TOKENS, PROMOTION_MAX_BYTES + 1,));
        assert!(!l2.consider_l3_fill("oversized", RESTORE_WORTH_TOKENS, PROMOTION_MAX_BYTES,));
    }

    #[test]
    fn a_changed_durable_digest_invalidates_the_mirror() {
        let l2 = StageL2::new(1 << 20, "model".to_string(), "state".to_string());
        let bytes = vec![7; 128];
        let digest = skippy_cache::segment_digest(&bytes);
        let payload = skippy_cache::ExactStatePayload::full_state(bytes);
        let identity = identity(vec![1, 2, 3]);
        assert!(l2.promote(
            &identity.namespace,
            &identity.token_ids,
            &digest,
            &payload,
            &ExactStateExtra { kv_desc: None },
        ));

        let changed = location(3, skippy_cache::segment_digest(b"changed"));
        assert!(l2.get(&identity, &changed).is_none());
        assert_eq!(l2.stats().entries, 0);
    }
}
