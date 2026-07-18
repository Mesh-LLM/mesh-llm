use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::{
    kv_integration::{KvStageIntegration, PrefillKvIdentity},
    runtime_state::RuntimeState,
    telemetry::{Telemetry, lifecycle_attrs},
};
use anyhow::Result;
use serde_json::{Value, json};
use skippy_metrics::attr;
use skippy_protocol::{
    StageConfig,
    binary::{StageReplyStats, StageWireMessage, WireMessageKind},
};
use skippy_runtime::ActivationFrame;

use super::kv_eviction::BinaryProactiveEviction;
use super::stage_execution::{
    binary_message_base, binary_message_request_id, elapsed_ms, stage_mask,
};

#[derive(Default)]
pub(in crate::binary_transport) struct BinaryKvLookupResult {
    pub(in crate::binary_transport) restored_tokens: usize,
    pub(in crate::binary_transport) activation: Option<ActivationFrame>,
    pub(in crate::binary_transport) stats: StageReplyStats,
}

#[derive(Default)]
pub(in crate::binary_transport) struct BinaryKvRecordResult {
    pub(in crate::binary_transport) recorded_pages: usize,
    pub(in crate::binary_transport) recorded_tokens: u64,
    pub(in crate::binary_transport) evicted_entries: usize,
    pub(in crate::binary_transport) evicted_tokens: u64,
    pub(in crate::binary_transport) recorded_activations: usize,
    pub(in crate::binary_transport) recorded_activation_bytes: u64,
    pub(in crate::binary_transport) evicted_activation_entries: usize,
    pub(in crate::binary_transport) evicted_activation_bytes: u64,
}

#[derive(Default)]
pub(in crate::binary_transport) struct BinaryPrefixCacheControlResult {
    pub(in crate::binary_transport) hit: bool,
    pub(in crate::binary_transport) stats: StageReplyStats,
}

pub(in crate::binary_transport) struct BinaryRestoredPrefix {
    page_id: String,
    token_count: usize,
    entries: usize,
    resident_seq_id: Option<i32>,
    resident_borrowed: Option<bool>,
    exact: bool,
}

impl BinaryRestoredPrefix {
    fn exact(page_id: String, token_count: usize, entries: usize) -> Self {
        Self {
            page_id,
            token_count,
            entries,
            resident_seq_id: None,
            resident_borrowed: None,
            exact: true,
        }
    }

    fn resident(
        page_id: String,
        token_count: usize,
        seq_id: i32,
        entries: usize,
        borrowed: bool,
    ) -> Self {
        Self {
            page_id,
            token_count,
            entries,
            resident_seq_id: Some(seq_id),
            resident_borrowed: Some(borrowed),
            exact: false,
        }
    }

    fn insert_hit_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        if self.exact {
            attrs.insert(
                "skippy.exact_cache.hit_page_id".to_string(),
                json!(self.page_id),
            );
            attrs.insert(
                "skippy.exact_cache.entries".to_string(),
                json!(self.entries),
            );
        } else {
            attrs.insert("skippy.kv.hit_page_id".to_string(), json!(self.page_id));
            attrs.insert(
                "skippy.kv.resident_entries".to_string(),
                json!(self.entries),
            );
            if let Some(seq_id) = self.resident_seq_id {
                attrs.insert("skippy.kv.resident_seq_id".to_string(), json!(seq_id));
            }
            if let Some(borrowed) = self.resident_borrowed {
                attrs.insert("skippy.kv.resident_lane_hit".to_string(), json!(borrowed));
            }
        }
    }
}
fn binary_kv_attrs(config: &StageConfig, kv: &KvStageIntegration) -> BTreeMap<String, Value> {
    let mut attrs = lifecycle_attrs(config);
    for (key, value) in kv.attrs() {
        attrs.insert(key.to_string(), value);
    }
    attrs
}

fn binary_message_kv_attrs(
    config: &StageConfig,
    kv: &KvStageIntegration,
    session_id: &str,
    message: &StageWireMessage,
    token_count: usize,
) -> BTreeMap<String, Value> {
    let mut attrs = binary_kv_attrs(config, kv);
    attrs.insert(attr::SESSION_ID.to_string(), json!(session_id));
    attrs.insert(
        attr::REQUEST_ID.to_string(),
        json!(binary_message_request_id(message)),
    );
    attrs.insert(
        "skippy.message_kind".to_string(),
        json!(format!("{:?}", message.kind)),
    );
    attrs.insert(
        "skippy.kv.token_start".to_string(),
        json!(message.pos_start.max(0)),
    );
    attrs.insert("skippy.kv.token_count".to_string(), json!(token_count));
    attrs
}

pub(in crate::binary_transport) fn maybe_prefix_cache_control(
    config: &StageConfig,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
) -> BinaryPrefixCacheControlResult {
    let mut result = BinaryPrefixCacheControlResult::default();
    let Some(kv) = kv else {
        return result;
    };
    if !kv.should_lookup() || token_ids.is_empty() {
        return result;
    }
    let token_start = if message.kind == WireMessageKind::TryRestorePrefillDecode {
        0
    } else {
        message.pos_start.max(0) as u64
    };
    let base = binary_message_base(config, session_id, message);
    let identity = kv.prefill_identity(config, &base, token_start, token_ids);
    let mut attrs = binary_message_kv_attrs(config, kv, session_id, message, token_ids.len());
    attrs.insert("skippy.kv.lookup_candidates".to_string(), json!(1));
    let started = Instant::now();
    if token_start != 0 {
        result.stats.kv_lookup_misses += 1;
        attrs.insert(
            "skippy.kv.lookup_ms".to_string(),
            json!(elapsed_ms(started)),
        );
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("nonzero_token_start_unsupported"),
        );
        telemetry.emit("stage.binary_kv_lookup_decision", attrs);
        return result;
    }
    match message.kind {
        WireMessageKind::ProbePrefill => {
            if let Some(probed) = kv.probe_resident_prefix(&identity) {
                result.hit = probed.token_count >= token_ids.len();
                if result.hit {
                    result.stats.kv_lookup_hits += 1;
                    result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
                    attrs.insert("skippy.kv.decision".to_string(), json!("probe_hit"));
                    attrs.insert("skippy.kv.hit_page_id".to_string(), json!(probed.page_id));
                    attrs.insert(
                        "skippy.kv.restored_tokens".to_string(),
                        json!(probed.token_count),
                    );
                    attrs.insert(
                        "skippy.kv.resident_seq_id".to_string(),
                        json!(probed.seq_id),
                    );
                    attrs.insert(
                        "skippy.kv.resident_entries".to_string(),
                        json!(probed.entries),
                    );
                } else {
                    result.stats.kv_lookup_misses += 1;
                    attrs.insert("skippy.kv.decision".to_string(), json!("probe_short"));
                    attrs.insert(
                        "skippy.kv.restored_tokens".to_string(),
                        json!(probed.token_count),
                    );
                }
            } else {
                result.stats.kv_lookup_misses += 1;
                attrs.insert("skippy.kv.decision".to_string(), json!("probe_miss"));
            }
        }
        WireMessageKind::RestorePrefill
        | WireMessageKind::TryRestorePrefill
        | WireMessageKind::TryRestorePrefillDecode => {
            let restore = {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                restore_binary_prefix(
                    kv,
                    &mut runtime,
                    session_id,
                    std::slice::from_ref(&identity),
                    token_ids,
                )
            };
            match restore {
                Ok(Some(restored)) if restored.token_count >= token_ids.len() => {
                    result.hit = true;
                    result.stats.kv_lookup_hits += 1;
                    result.stats.kv_imported_tokens += restored.token_count as i64;
                    result.stats.kv_imported_pages += 1;
                    result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
                    let decision = match message.kind {
                        WireMessageKind::TryRestorePrefill => "try_restore_hit",
                        WireMessageKind::TryRestorePrefillDecode => "try_restore_decode_hit",
                        _ => "restore_hit",
                    };
                    attrs.insert("skippy.kv.decision".to_string(), json!(decision));
                    restored.insert_hit_attrs(&mut attrs);
                    attrs.insert(
                        "skippy.kv.restored_tokens".to_string(),
                        json!(restored.token_count),
                    );
                }
                Ok(Some(restored)) => {
                    result.stats.kv_lookup_misses += 1;
                    let decision = match message.kind {
                        WireMessageKind::TryRestorePrefill => "try_restore_short",
                        WireMessageKind::TryRestorePrefillDecode => "try_restore_decode_short",
                        _ => "restore_short",
                    };
                    attrs.insert("skippy.kv.decision".to_string(), json!(decision));
                    attrs.insert(
                        "skippy.kv.restored_tokens".to_string(),
                        json!(restored.token_count),
                    );
                }
                Ok(None) => {
                    result.stats.kv_lookup_misses += 1;
                    let decision = match message.kind {
                        WireMessageKind::TryRestorePrefill => "try_restore_miss",
                        WireMessageKind::TryRestorePrefillDecode => "try_restore_decode_miss",
                        _ => "restore_miss",
                    };
                    attrs.insert("skippy.kv.decision".to_string(), json!(decision));
                }
                Err(error) => {
                    result.stats.kv_lookup_errors += 1;
                    let decision = match message.kind {
                        WireMessageKind::TryRestorePrefill => "try_restore_error",
                        WireMessageKind::TryRestorePrefillDecode => "try_restore_decode_error",
                        _ => "restore_error",
                    };
                    attrs.insert("skippy.kv.decision".to_string(), json!(decision));
                    attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                }
            }
        }
        _ => return result,
    }
    attrs.insert(
        "skippy.kv.lookup_ms".to_string(),
        json!(elapsed_ms(started)),
    );
    telemetry.emit("stage.binary_kv_lookup_decision", attrs);
    result
}

fn restore_binary_prefix(
    kv: &KvStageIntegration,
    runtime: &mut RuntimeState,
    session_id: &str,
    identities: &[PrefillKvIdentity],
    token_ids: &[i32],
) -> Result<Option<BinaryRestoredPrefix>> {
    match kv.restore_exact_state(runtime, session_id, identities)? {
        Some(restored) => Ok(Some(BinaryRestoredPrefix::exact(
            restored.page_id,
            restored.token_count,
            restored.entries,
        ))),
        None => kv
            .restore_resident_prefix(runtime, session_id, identities, token_ids)
            .map(|restored| {
                restored.map(|restored| {
                    BinaryRestoredPrefix::resident(
                        restored.page_id,
                        restored.token_count,
                        restored.seq_id,
                        restored.entries,
                        restored.borrowed,
                    )
                })
            }),
    }
}
pub(in crate::binary_transport) fn emit_binary_proactive_eviction(
    telemetry: &Telemetry,
    eviction: &BinaryProactiveEviction,
) {
    if eviction.should_emit_summary() {
        telemetry.emit("stage.binary_kv_record_decision", eviction.attrs());
    } else {
        telemetry.emit_debug("stage.binary_kv_record_decision", eviction.attrs());
    }
}
#[allow(clippy::too_many_arguments)]
pub(in crate::binary_transport) fn maybe_lookup_binary_prefill(
    config: &StageConfig,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
    activation_width: i32,
) -> BinaryKvLookupResult {
    let mut result = BinaryKvLookupResult::default();
    let Some(kv) = kv else {
        return result;
    };
    if !message.kind.is_prefill()
        || message.kind.requires_predicted_reply()
        || !kv.should_lookup()
        || token_ids.is_empty()
    {
        return result;
    }
    let token_start = message.pos_start.max(0) as u64;
    let base = binary_message_base(config, session_id, message);
    let identities = kv.lookup_identities(config, &base, token_start, token_ids);
    let mut attrs = binary_message_kv_attrs(config, kv, session_id, message, token_ids.len());
    attrs.insert(
        "skippy.kv.lookup_candidates".to_string(),
        json!(identities.len()),
    );
    let started = Instant::now();
    if token_start != 0 {
        result.stats.kv_lookup_misses += 1;
        attrs.insert(
            "skippy.kv.lookup_ms".to_string(),
            json!(elapsed_ms(started)),
        );
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("nonzero_token_start_unsupported"),
        );
        telemetry.emit("stage.binary_kv_lookup_decision", attrs);
        return result;
    }
    if config.downstream.is_some() {
        let Some(activation) =
            kv.restore_resident_activation(config, &base, token_start, token_ids, activation_width)
        else {
            result.stats.kv_lookup_misses += 1;
            attrs.insert(
                "skippy.kv.lookup_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("activation_resident_miss"),
            );
            telemetry.emit("stage.binary_kv_lookup_decision", attrs);
            return result;
        };
        let prefix_restore = {
            let mut runtime = runtime.lock().expect("runtime lock poisoned");
            restore_binary_prefix(
                kv,
                &mut runtime,
                session_id,
                std::slice::from_ref(&activation.identity),
                token_ids,
            )
        };
        match prefix_restore {
            Ok(Some(restored)) if restored.token_count >= token_ids.len() => {
                result.restored_tokens = restored.token_count;
                result.activation = Some(activation.frame);
                result.stats.kv_lookup_hits += 1;
                result.stats.kv_imported_tokens += restored.token_count as i64;
                result.stats.kv_imported_pages += 1;
                result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
                attrs.insert(
                    "skippy.kv.lookup_ms".to_string(),
                    json!(elapsed_ms(started)),
                );
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!("activation_resident_hit"),
                );
                restored.insert_hit_attrs(&mut attrs);
                attrs.insert(
                    "skippy.activation_cache.hit_page_id".to_string(),
                    json!(activation.page_id),
                );
                attrs.insert(
                    "skippy.kv.restored_tokens".to_string(),
                    json!(restored.token_count),
                );
                attrs.insert(
                    "skippy.activation_cache.payload_bytes".to_string(),
                    json!(activation.payload_bytes),
                );
                attrs.insert(
                    "skippy.activation_cache.entries".to_string(),
                    json!(activation.entries),
                );
                telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                return result;
            }
            Ok(Some(restored)) => {
                result.stats.kv_lookup_misses += 1;
                attrs.insert(
                    "skippy.kv.lookup_ms".to_string(),
                    json!(elapsed_ms(started)),
                );
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!("activation_hit_prefix_short"),
                );
                attrs.insert(
                    "skippy.kv.restored_tokens".to_string(),
                    json!(restored.token_count),
                );
                telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                return result;
            }
            Ok(None) => {
                result.stats.kv_lookup_misses += 1;
                attrs.insert(
                    "skippy.kv.lookup_ms".to_string(),
                    json!(elapsed_ms(started)),
                );
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!("activation_hit_kv_miss"),
                );
                attrs.insert(
                    "skippy.activation_cache.hit_page_id".to_string(),
                    json!(activation.page_id),
                );
                telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                return result;
            }
            Err(error) => {
                result.stats.kv_lookup_errors += 1;
                attrs.insert(
                    "skippy.kv.lookup_ms".to_string(),
                    json!(elapsed_ms(started)),
                );
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!("activation_hit_kv_error"),
                );
                attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                return result;
            }
        }
    }
    let prefix_restore = {
        let mut runtime = runtime.lock().expect("runtime lock poisoned");
        restore_binary_prefix(kv, &mut runtime, session_id, &identities, token_ids)
    };
    match prefix_restore {
        Ok(Some(restored)) => {
            result.restored_tokens = restored.token_count;
            result.stats.kv_lookup_hits += 1;
            result.stats.kv_imported_tokens += restored.token_count as i64;
            result.stats.kv_imported_pages += 1;
            result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
            attrs.insert(
                "skippy.kv.lookup_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert("skippy.kv.decision".to_string(), json!("resident_hit"));
            restored.insert_hit_attrs(&mut attrs);
            attrs.insert(
                "skippy.kv.restored_tokens".to_string(),
                json!(restored.token_count),
            );
            attrs.insert(
                "skippy.kv.suffix_prefill_tokens".to_string(),
                json!(token_ids.len().saturating_sub(restored.token_count)),
            );
            telemetry.emit("stage.binary_kv_lookup_decision", attrs);
            return result;
        }
        Ok(None) => {}
        Err(error) => {
            result.stats.kv_lookup_errors += 1;
            attrs.insert(
                "skippy.kv.lookup_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert("skippy.kv.decision".to_string(), json!("resident_error"));
            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
            telemetry.emit("stage.binary_kv_lookup_decision", attrs);
            return result;
        }
    }
    result.stats.kv_lookup_misses += 1;
    attrs.insert(
        "skippy.kv.lookup_ms".to_string(),
        json!(elapsed_ms(started)),
    );
    attrs.insert("skippy.kv.decision".to_string(), json!("resident_miss"));
    telemetry.emit("stage.binary_kv_lookup_decision", attrs);
    result
}

#[allow(clippy::too_many_arguments)]
pub(in crate::binary_transport) fn maybe_record_binary_prefill(
    config: &StageConfig,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
    min_record_tokens: u64,
    activation_width: i32,
    output: Option<&ActivationFrame>,
) -> BinaryKvRecordResult {
    let mut result = BinaryKvRecordResult::default();
    let Some(kv) = kv else {
        return result;
    };
    if !message.kind.is_prefill()
        || message.kind.requires_predicted_reply()
        || !kv.should_record()
        || token_ids.is_empty()
    {
        return result;
    }
    let token_start = message.pos_start.max(0) as u64;
    let base = binary_message_base(config, session_id, message);
    let identities = kv.record_identities(config, &base, token_start, token_ids);
    let mut attrs = binary_message_kv_attrs(config, kv, session_id, message, token_ids.len());
    attrs.insert(
        "skippy.kv.record_candidates".to_string(),
        json!(identities.len()),
    );
    let started = Instant::now();
    if token_start != 0 {
        attrs.insert(
            "skippy.kv.record_ms".to_string(),
            json!(elapsed_ms(started)),
        );
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("nonzero_token_start_unsupported"),
        );
        telemetry.emit("stage.binary_kv_record_decision", attrs);
        return result;
    }
    {
        let mut runtime = runtime.lock().expect("runtime lock poisoned");
        for identity in identities {
            let token_count = identity.identity.token_count;
            if token_count <= min_record_tokens {
                continue;
            }
            let token_count_usize = usize::try_from(token_count)
                .unwrap_or(usize::MAX)
                .min(token_ids.len());
            if token_count_usize == token_ids.len() {
                match kv.record_exact_state(&mut runtime, session_id, &identity) {
                    Ok(Some(record)) => {
                        result.recorded_pages = result.recorded_pages.saturating_add(1);
                        result.recorded_tokens = result
                            .recorded_tokens
                            .saturating_add(record.token_count as u64);
                        result.evicted_entries = result
                            .evicted_entries
                            .saturating_add(record.evicted_entries);
                        result.evicted_tokens = result
                            .evicted_tokens
                            .saturating_add(record.evicted_logical_bytes);
                        attrs.insert(
                            "skippy.exact_cache.recorded_page_id".to_string(),
                            json!(record.page_id),
                        );
                        attrs.insert(
                            "skippy.exact_cache.payload_kind".to_string(),
                            json!(record.payload_kind.to_string()),
                        );
                        attrs.insert(
                            "skippy.exact_cache.logical_bytes".to_string(),
                            json!(record.logical_bytes),
                        );
                        attrs.insert(
                            "skippy.exact_cache.physical_bytes".to_string(),
                            json!(record.physical_bytes),
                        );
                        attrs.insert(
                            "skippy.exact_cache.entries".to_string(),
                            json!(record.entries),
                        );
                        attrs.insert(
                            "skippy.exact_cache.dedupe_reused_block_count".to_string(),
                            json!(record.dedupe.reused_block_count),
                        );
                        continue;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        attrs.insert(
                            "skippy.exact_cache.record_error".to_string(),
                            json!(error.to_string()),
                        );
                    }
                }
            }
            match kv.record_resident_prefix(
                &mut runtime,
                session_id,
                &identity,
                &token_ids[..token_count_usize],
            ) {
                Ok(Some(record)) => {
                    result.recorded_pages = result.recorded_pages.saturating_add(1);
                    result.recorded_tokens = result
                        .recorded_tokens
                        .saturating_add(record.token_count as u64);
                    result.evicted_entries = result
                        .evicted_entries
                        .saturating_add(record.evicted_entries);
                    result.evicted_tokens =
                        result.evicted_tokens.saturating_add(record.evicted_tokens);
                }
                Ok(None) => {}
                Err(error) => {
                    attrs.insert(
                        "skippy.kv.record_error".to_string(),
                        json!(error.to_string()),
                    );
                    break;
                }
            }
        }
    }
    if config.downstream.is_some()
        && let Some(output) = output
        && let Some(record) = kv.record_resident_activation(
            config,
            &base,
            token_start,
            token_ids,
            activation_width,
            output,
        )
    {
        result.recorded_activations = result.recorded_activations.saturating_add(1);
        result.recorded_activation_bytes = result
            .recorded_activation_bytes
            .saturating_add(record.payload_bytes as u64);
        result.evicted_activation_entries = result
            .evicted_activation_entries
            .saturating_add(record.evicted_entries);
        result.evicted_activation_bytes = result
            .evicted_activation_bytes
            .saturating_add(record.evicted_bytes);
        attrs.insert(
            "skippy.activation_cache.recorded_page_id".to_string(),
            json!(record.page_id),
        );
        attrs.insert(
            "skippy.activation_cache.entries".to_string(),
            json!(record.entries),
        );
        attrs.insert(
            "skippy.activation_cache.resident_bytes".to_string(),
            json!(record.resident_bytes),
        );
    }
    attrs.insert(
        "skippy.kv.record_ms".to_string(),
        json!(elapsed_ms(started)),
    );
    attrs.insert(
        "skippy.kv.recorded_pages".to_string(),
        json!(result.recorded_pages),
    );
    attrs.insert(
        "skippy.kv.recorded_tokens".to_string(),
        json!(result.recorded_tokens),
    );
    attrs.insert(
        "skippy.kv.evicted_entries".to_string(),
        json!(result.evicted_entries),
    );
    attrs.insert(
        "skippy.kv.evicted_tokens".to_string(),
        json!(result.evicted_tokens),
    );
    attrs.insert(
        "skippy.activation_cache.recorded_frames".to_string(),
        json!(result.recorded_activations),
    );
    attrs.insert(
        "skippy.activation_cache.recorded_bytes".to_string(),
        json!(result.recorded_activation_bytes),
    );
    attrs.insert(
        "skippy.activation_cache.evicted_entries".to_string(),
        json!(result.evicted_activation_entries),
    );
    attrs.insert(
        "skippy.activation_cache.evicted_bytes".to_string(),
        json!(result.evicted_activation_bytes),
    );
    telemetry.emit("stage.binary_kv_record_decision", attrs);
    result
}

pub(in crate::binary_transport) fn accumulate_prefill_tokens(
    accumulated: &mut BTreeMap<String, Vec<i32>>,
    session_id: &str,
    token_start: usize,
    token_ids: &[i32],
) {
    if token_ids.is_empty() {
        return;
    }
    let tokens = accumulated.entry(session_id.to_string()).or_default();
    if token_start == 0 {
        tokens.clear();
    }
    if token_start == tokens.len() {
        tokens.extend_from_slice(token_ids);
    }
}

pub(in crate::binary_transport) fn maybe_record_binary_full_prefill(
    config: &StageConfig,
    runtime: &mut RuntimeState,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
) -> BinaryKvRecordResult {
    let mut result = BinaryKvRecordResult::default();
    let Some(kv) = kv else {
        return result;
    };
    if !kv.should_record() || token_ids.is_empty() {
        return result;
    }
    let identities =
        binary_full_prefill_record_identities(kv, config, session_id, message, token_ids);
    let mut attrs = binary_message_kv_attrs(config, kv, session_id, message, token_ids.len());
    attrs.insert(
        "skippy.kv.record_candidates".to_string(),
        json!(identities.len()),
    );
    attrs.insert(
        "skippy.kv.decision".to_string(),
        json!("full_prefill_record"),
    );
    let started = Instant::now();
    for identity in identities {
        let token_count_usize = usize::try_from(identity.identity.token_count)
            .unwrap_or(usize::MAX)
            .min(token_ids.len());
        if token_count_usize == token_ids.len() {
            match kv.record_exact_state(runtime, session_id, &identity) {
                Ok(Some(record)) => {
                    result.recorded_pages = result.recorded_pages.saturating_add(1);
                    result.recorded_tokens = result
                        .recorded_tokens
                        .saturating_add(record.token_count as u64);
                    result.evicted_entries = result
                        .evicted_entries
                        .saturating_add(record.evicted_entries);
                    result.evicted_tokens = result
                        .evicted_tokens
                        .saturating_add(record.evicted_logical_bytes);
                    attrs.insert(
                        "skippy.exact_cache.recorded_page_id".to_string(),
                        json!(record.page_id),
                    );
                    attrs.insert(
                        "skippy.exact_cache.payload_kind".to_string(),
                        json!(record.payload_kind.to_string()),
                    );
                    attrs.insert(
                        "skippy.exact_cache.logical_bytes".to_string(),
                        json!(record.logical_bytes),
                    );
                    attrs.insert(
                        "skippy.exact_cache.physical_bytes".to_string(),
                        json!(record.physical_bytes),
                    );
                    attrs.insert(
                        "skippy.exact_cache.entries".to_string(),
                        json!(record.entries),
                    );
                    continue;
                }
                Ok(None) => {}
                Err(error) => {
                    attrs.insert(
                        "skippy.exact_cache.record_error".to_string(),
                        json!(error.to_string()),
                    );
                }
            }
        }
        match kv.record_resident_prefix(
            runtime,
            session_id,
            &identity,
            &token_ids[..token_count_usize],
        ) {
            Ok(Some(record)) => {
                result.recorded_pages = result.recorded_pages.saturating_add(1);
                result.recorded_tokens = result
                    .recorded_tokens
                    .saturating_add(record.token_count as u64);
                result.evicted_entries = result
                    .evicted_entries
                    .saturating_add(record.evicted_entries);
                result.evicted_tokens = result.evicted_tokens.saturating_add(record.evicted_tokens);
                attrs.insert(
                    "skippy.kv.recorded_page_id".to_string(),
                    json!(record.page_id),
                );
                attrs.insert(
                    "skippy.kv.resident_seq_id".to_string(),
                    json!(record.seq_id),
                );
            }
            Ok(None) => {}
            Err(error) => {
                attrs.insert(
                    "skippy.kv.record_error".to_string(),
                    json!(error.to_string()),
                );
                break;
            }
        }
    }
    attrs.insert(
        "skippy.kv.record_ms".to_string(),
        json!(elapsed_ms(started)),
    );
    attrs.insert(
        "skippy.kv.recorded_pages".to_string(),
        json!(result.recorded_pages),
    );
    attrs.insert(
        "skippy.kv.recorded_tokens".to_string(),
        json!(result.recorded_tokens),
    );
    telemetry.emit("stage.binary_kv_record_decision", attrs);
    result
}

pub(in crate::binary_transport) fn binary_full_prefill_record_identities(
    kv: &KvStageIntegration,
    config: &StageConfig,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
) -> Vec<PrefillKvIdentity> {
    let base = binary_message_base(config, session_id, message);
    kv.record_identities(config, &base, 0, token_ids)
}
pub(in crate::binary_transport) fn add_binary_record_stats(
    reply_stats: &mut StageReplyStats,
    config: &StageConfig,
    record: &BinaryKvRecordResult,
) {
    if record.recorded_pages > 0 {
        reply_stats.kv_recorded_pages += record.recorded_pages as i64;
        reply_stats.kv_record_stage_mask |= stage_mask(config.stage_index);
    }
    if record.recorded_activations > 0 {
        reply_stats.kv_recorded_bytes = reply_stats
            .kv_recorded_bytes
            .saturating_add(record.recorded_activation_bytes as i64);
    }
}

#[cfg(test)]
mod tests {
    use super::binary_full_prefill_record_identities;
    use crate::binary_transport::stage_execution::prefix_cache_test_config;
    use crate::kv_integration::KvStageIntegration;
    use skippy_protocol::binary::{
        StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
    };
    fn prefill_message() -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32),
            request_id: 11,
            session_id: 13,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }
    #[test]
    fn binary_full_prefill_record_plan_includes_shared_prefix_candidate() {
        let config = prefix_cache_test_config();
        let kv = KvStageIntegration::from_config(&config)
            .unwrap()
            .expect("resident prefix cache enabled");
        let message = prefill_message();
        let recorded_tokens = (0..2214).collect::<Vec<_>>();
        let mut lookup_tokens = recorded_tokens.clone();
        lookup_tokens.extend(100_000..100_017);

        let record_plan = binary_full_prefill_record_identities(
            &kv,
            &config,
            "session",
            &message,
            &recorded_tokens,
        );
        let base = super::binary_message_base(&config, "session", &message);
        let lookup_plan = kv.lookup_identities(&config, &base, 0, &lookup_tokens);

        let record_counts = record_plan
            .iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();
        let lookup_counts = lookup_plan
            .iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();

        assert_eq!(record_counts, vec![2214, 2176]);
        assert!(lookup_counts.contains(&2176));

        let recorded_shared = record_plan
            .iter()
            .find(|identity| identity.identity.token_count == 2176)
            .expect("binary full-prefill record plan should include shared grid prefix");
        let lookup_shared = lookup_plan
            .iter()
            .find(|identity| identity.identity.token_count == 2176)
            .expect("lookup plan should probe shared grid prefix");
        let recorded_exact = record_plan
            .iter()
            .find(|identity| identity.identity.token_count == 2214)
            .expect("binary full-prefill record plan should keep exact first prompt");
        let lookup_exact = lookup_plan
            .iter()
            .find(|identity| identity.identity.token_count == 2231)
            .expect("lookup plan should probe exact second prompt");

        assert_eq!(recorded_shared.page_id, lookup_shared.page_id);
        assert_ne!(recorded_exact.page_id, lookup_exact.page_id);
    }
}
