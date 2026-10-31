//! Materialize a native prefix only when releasing it to the persistent tier.
use super::*;
use crate::runtime_state::RuntimeState;
use config::durable_spill::PendingDurableSpill;

#[derive(Debug)]
pub(super) struct NativeOffloadClaim {
    claims: Arc<Mutex<BTreeSet<String>>>,
    key: String,
}

impl NativeOffloadClaim {
    fn acquire(
        claims: &Arc<Mutex<BTreeSet<String>>>,
        namespace: &str,
        tokens: &[i32],
    ) -> Option<Self> {
        let key = format!(
            "native-offload:{}",
            skippy_cache::l3_prefix_key(namespace, tokens)
        );
        if !claims
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key.clone())
        {
            return None;
        }
        Some(Self {
            claims: Arc::clone(claims),
            key,
        })
    }
}

impl Drop for NativeOffloadClaim {
    fn drop(&mut self) {
        self.claims
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.key);
    }
}

impl KvStageIntegration {
    pub(super) fn native_payload_is_complete(
        &self,
        kind: skippy_cache::ExactStatePayloadKind,
    ) -> bool {
        if !self.records_resident_prefixes() {
            return kind != skippy_cache::ExactStatePayloadKind::RecurrentOnly;
        }
        self.loaded_memory_cache.is_some_and(|caps| {
            if caps.resident_attention {
                kind == skippy_cache::ExactStatePayloadKind::KvRecurrent
            } else {
                caps.resident_checkpoint
                    && kind == skippy_cache::ExactStatePayloadKind::RecurrentOnly
            }
        })
    }

    pub(super) fn native_components_are_complete(
        &self,
        kv_bytes: u64,
        recurrent_bytes: u64,
        has_descriptor: bool,
    ) -> bool {
        self.loaded_memory_cache.is_none_or(|caps| {
            !caps.resident
                || ((!caps.resident_attention || (kv_bytes > 0 && has_descriptor))
                    && (!caps.resident_checkpoint || recurrent_bytes > 0))
        })
    }

    pub(super) fn release_native_prefix(
        &self,
        runtime: &mut RuntimeState,
        session: &str,
        victim: &skippy_cache::RadixEvictionCandidate<RadixResidentEntry>,
    ) -> Result<()> {
        // Disk admission/export is optional. Failure must not prevent reclaiming
        // native cells needed by an active request.
        if let Err(error) = self.offload_native_prefix(runtime, session, victim) {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::AcqRel) {
                let _ = skippy_events::diagnostics::emit(
                    skippy_events::diagnostics::ServingDiagnostic::Warning {
                        message:
                            "Native prefix cold offload failed; continuing with cache eviction"
                                .into(),
                        context: Some(format!("reason={error:#}")),
                    },
                );
            }
        }
        runtime.drop_resident_prefix_sequence(session, victim.value.seq_id)
    }

    fn offload_native_prefix(
        &self,
        runtime: &mut RuntimeState,
        session: &str,
        victim: &skippy_cache::RadixEvictionCandidate<RadixResidentEntry>,
    ) -> Result<bool> {
        let sender = self
            .exact_state_record_worker
            .durable_sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let Some(sender) = sender else {
            return Ok(false);
        };
        let Some(capabilities) = self.loaded_memory_cache else {
            return Ok(false);
        };
        let Some(l3) = self.l3.as_ref() else {
            return Ok(false);
        };
        if l3
            .locate_longest(&victim.namespace, &victim.tokens, 1)?
            .is_some_and(|entry| entry.token_count == victim.tokens.len() as u64)
        {
            return Ok(false);
        }
        let Some(claim) =
            NativeOffloadClaim::acquire(&self.inflight_records, &victim.namespace, &victim.tokens)
        else {
            return Ok(false);
        };
        let sequence = victim.value.seq_id;
        let tokens = victim.tokens.len() as u64;
        let desc = if capabilities.resident_attention {
            Some(runtime.resident_kv_page_desc(session, sequence, tokens)?)
        } else {
            None
        };
        let recurrent_bytes = if capabilities.resident_checkpoint {
            runtime.resident_recurrent_state_bytes(session, sequence)?
        } else {
            0
        };
        let kv_bytes = desc.as_ref().map_or(0, |desc| desc.payload_bytes);
        let bytes = kv_bytes
            .checked_add(recurrent_bytes as u64)
            .ok_or_else(|| anyhow::anyhow!("native offload size overflow"))?;
        if bytes == 0 {
            return Ok(false);
        }
        let Some(reservation) = sender.try_reserve(bytes) else {
            return Ok(false);
        };
        let kv = if let Some(desc) = &desc {
            runtime
                .export_resident_kv_page(
                    session,
                    sequence,
                    tokens,
                    usize::try_from(desc.payload_bytes)?,
                )?
                .payload
        } else {
            Vec::new()
        };
        let recurrent = if recurrent_bytes > 0 {
            runtime.export_resident_recurrent_state(session, sequence, recurrent_bytes)?
        } else {
            Vec::new()
        };
        let payload = if capabilities.resident_attention {
            ExactStatePayload::kv_recurrent(kv, recurrent)
        } else {
            anyhow::ensure!(
                capabilities.resident_checkpoint && !recurrent.is_empty(),
                "resident prefix has no complete export component"
            );
            ExactStatePayload::recurrent_only(recurrent)
        };
        Ok(sender.send_reserved(
            PendingDurableSpill {
                page_id: victim.value.page_id.clone(),
                payload,
                extra: ExactStateExtra { kv_desc: desc },
                namespace: victim.namespace.clone(),
                token_ids: victim.tokens.clone(),
                l3_cost: None,
                native_raw: true,
            },
            reservation,
            claim,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_offload_claim_is_token_keyed_and_released_on_every_drop() {
        let claims = Arc::new(Mutex::new(BTreeSet::new()));
        let claim = NativeOffloadClaim::acquire(&claims, "model", &[1, 2]).unwrap();
        assert!(NativeOffloadClaim::acquire(&claims, "model", &[1, 2]).is_none());
        assert!(NativeOffloadClaim::acquire(&claims, "model", &[1, 3]).is_some());
        assert!(NativeOffloadClaim::acquire(&claims, "other", &[1, 2]).is_some());
        drop(claim);
        assert!(NativeOffloadClaim::acquire(&claims, "model", &[1, 2]).is_some());
        assert!(claims.lock().unwrap().is_empty());
    }
}
