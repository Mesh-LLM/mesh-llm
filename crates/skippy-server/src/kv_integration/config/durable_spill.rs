use super::*;

#[derive(Clone, Copy)]
pub(super) struct DurableRecordTarget<'a> {
    pub(super) l3: Option<&'a L3Tier>,
    pub(super) cachegen_enabled: bool,
    #[cfg(test)]
    pub(super) before_l3_spill: Option<&'a dyn Fn()>,
}

/// Exact-state payload handed from the serving worker to durable storage.
#[derive(Debug)]
pub(super) struct PendingDurableSpill {
    pub(super) page_id: String,
    pub(super) payload: skippy_cache::ExactStatePayload,
    pub(super) extra: super::super::ExactStateExtra,
    pub(super) namespace: String,
    pub(super) token_ids: Vec<i32>,
    pub(super) l3_cost: Option<skippy_cache::policy::CostSample>,
}

type DurableSpillWorker = (
    Option<std::sync::mpsc::SyncSender<PendingDurableSpill>>,
    Option<std::thread::JoinHandle<()>>,
);

/// Starts the optional bounded worker that isolates durable I/O from L1 publication.
pub(super) fn start_durable_spill_worker(
    l3: Option<Arc<L3Tier>>,
    cachegen_enabled: bool,
    stage_id: &str,
    #[cfg(test)] received: Arc<std::sync::atomic::AtomicUsize>,
    #[cfg(test)] pause: Arc<std::sync::atomic::AtomicBool>,
) -> Result<DurableSpillWorker> {
    let Some(l3) = l3 else {
        return Ok((None, None));
    };
    let (tx, rx) =
        std::sync::mpsc::sync_channel::<PendingDurableSpill>(EXACT_STATE_RECORD_CAPACITY);
    let task = std::thread::Builder::new()
        .name(format!("skippy-l3-spill-{stage_id}"))
        .spawn(move || {
            while let Ok(pending) = rx.recv() {
                #[cfg(test)]
                received.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                #[cfg(test)]
                while pause.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                spill_exact_record_to_l3(&l3, cachegen_enabled, pending);
            }
        })?;
    Ok((Some(tx), Some(task)))
}

/// Persists one exact-state payload without affecting its already-published L1 entry.
pub(super) fn spill_exact_record_to_l3(
    l3: &L3Tier,
    cachegen_enabled: bool,
    pending: PendingDurableSpill,
) {
    let PendingDurableSpill {
        page_id,
        payload,
        extra,
        namespace,
        token_ids,
        l3_cost,
    } = pending;
    let kv_desc_json = extra
        .kv_desc
        .as_ref()
        .and_then(|desc| serde_json::to_string(desc).ok());
    let geometry = extra
        .kv_desc
        .as_ref()
        .and_then(|desc| kv_page_geometry(desc, payload.byte_len()));
    let cachegen_spill = if cachegen_enabled {
        match (extra.kv_desc.as_ref(), payload.kv_bytes().ok().flatten()) {
            (Some(desc), Some(kv)) if !kv.is_empty() && cachegen_descriptor_is_qualified(desc) => {
                match skippy_runtime::encode_cachegen_kv_page(desc, kv.as_ref()) {
                    Ok(archive) if archive.bytes.len() < kv.len() => {
                        let calibration_digest = skippy_cache::segment_digest(&archive.bytes);
                        Some(l3.spill_cachegen_with_cost(
                            &namespace,
                            &token_ids,
                            &payload,
                            kv_desc_json.clone().unwrap_or_default(),
                            skippy_cache::CacheGenKvPayload {
                                archive: archive.bytes,
                                decoded_len: desc.payload_bytes,
                                calibration_digest,
                            },
                            l3_cost,
                        ))
                    }
                    Ok(_) => None,
                    Err(error) => {
                        static WARNED_CACHEGEN: std::sync::atomic::AtomicBool =
                            std::sync::atomic::AtomicBool::new(false);
                        if !WARNED_CACHEGEN.swap(true, std::sync::atomic::Ordering::AcqRel) {
                            let _ = mesh_llm_events::emit_event(OutputEvent::Warning {
                                message: "CacheGen encode declined; storing native KV page"
                                    .to_string(),
                                context: Some(format!("page_id={} reason={error:#}", page_id)),
                            });
                        }
                        None
                    }
                }
            }
            _ => None,
        }
    } else {
        None
    };
    let spill = cachegen_spill.unwrap_or_else(|| {
        l3.spill_with_cost(
            &namespace,
            &token_ids,
            &payload,
            kv_desc_json,
            geometry.as_ref(),
            l3_cost,
        )
    });
    emit_l3_state_transitions(l3);
    if let Err(error) = spill {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::AcqRel) {
            let _ = mesh_llm_events::emit_event(OutputEvent::Warning {
                message: "Skippy L3 disk cache write refused; see kv-cache status".to_string(),
                context: Some(format!("page_id={page_id} reason={error:#}")),
            });
        }
    }
}
