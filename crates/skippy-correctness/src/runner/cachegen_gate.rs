use std::{
    fs::{self, File},
    hint::black_box,
    io::Write,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use skippy_cache::cachegen::archive::{
    CacheGenArchive, ComponentLayout, PageLayout, decode_f16_page, encode_f16_page,
};
use skippy_runtime::{
    GGML_TYPE_F16, KV_PAGE_FLAG_V_TRANSPOSED, RuntimeKvPageDesc, StageModel, StageSession,
    TokenSignal,
};

use crate::report::CacheGenGateReport;

use super::{
    stage_execution::{BinaryStateHandoffConfig, elapsed_ms},
    state_handoff::LocalStatePayload,
};

struct PersistedPayloads {
    native_kv: Vec<u8>,
    native_recurrent: Vec<u8>,
    cachegen_archive: Vec<u8>,
    cachegen_recurrent: Vec<u8>,
    native_write_ms: f64,
    cachegen_write_ms: f64,
    native_read_ms: f64,
    cachegen_read_ms: f64,
}

pub(in crate::runner) fn run_cachegen_gate(
    model: &StageModel,
    args: &BinaryStateHandoffConfig,
    payload: &LocalStatePayload,
    prefix: &[i32],
    continuation: i32,
) -> Result<CacheGenGateReport> {
    let LocalStatePayload::KvRecurrent {
        kv_desc: Some(kv_desc),
        kv,
        recurrent,
    } = payload
    else {
        bail!("CacheGen gate requires an exported KV page descriptor and bytes");
    };
    kv_desc.validate_payload(kv.len())?;

    let encode_started = Instant::now();
    let archive = encode_kv_archive(kv_desc, kv)?;
    let encode_ms = elapsed_ms(encode_started);
    let native_storage_bytes = kv.len().saturating_add(recurrent.len());
    let cachegen_storage_bytes = archive.bytes.len().saturating_add(recurrent.len());

    let persisted = persist_and_read_payloads(kv, recurrent, &archive.bytes)?;
    // Keep the scalar decoder in the gate as an independently timed oracle,
    // but never feed its multi-gigabyte output into the measured restore. The
    // accelerated path must consume the persisted archive directly or fail.
    let scalar_oracle_decode_started = Instant::now();
    let scalar_oracle = decode_kv_archive(kv_desc, &persisted.cachegen_archive)?;
    let scalar_oracle_decode_ms = elapsed_ms(scalar_oracle_decode_started);
    black_box(&scalar_oracle);
    drop(scalar_oracle);

    let native_payload = LocalStatePayload::KvRecurrent {
        kv_desc: Some(kv_desc.clone()),
        kv: persisted.native_kv,
        recurrent: persisted.native_recurrent,
    };
    let native_import_started = Instant::now();
    let mut native = import_session(model, &native_payload, prefix)?;
    let native_import_ms = elapsed_ms(native_import_started);
    let native_continuation =
        run_native_continuation(&mut native, continuation, args.cachegen_continuation_steps)?;
    drop(native);

    let cachegen_import_started = Instant::now();
    let mut cachegen = import_cachegen_session(
        model,
        kv_desc,
        &persisted.cachegen_archive,
        &persisted.cachegen_recurrent,
        prefix,
    )?;
    let cachegen_import_ms = elapsed_ms(cachegen_import_started);
    let continuation =
        compare_cachegen_continuation(&mut cachegen, continuation, native_continuation)?;
    let native_p99_decode_ms = percentile_99(&continuation.native_decode_ms);
    let cachegen_p99_decode_ms = percentile_99(&continuation.cachegen_decode_ms);
    let p99_decode_regression = relative_regression(cachegen_p99_decode_ms, native_p99_decode_ms);
    let native_ttft_ms = restore_to_first_token_ms(
        persisted.native_read_ms,
        native_import_ms,
        &continuation.native_decode_ms,
    );
    let cachegen_ttft_ms = restore_to_first_token_ms(
        persisted.cachegen_read_ms,
        cachegen_import_ms,
        &continuation.cachegen_decode_ms,
    );
    let token_agreement =
        continuation.matching_tokens as f64 / args.cachegen_continuation_steps as f64;
    let compression_ratio = cachegen_storage_bytes as f64 / native_storage_bytes.max(1) as f64;

    let mut failure_reasons = Vec::new();
    if cachegen_storage_bytes >= native_storage_bytes {
        failure_reasons.push("encoded payload is not smaller than native".to_string());
    }
    if cachegen_ttft_ms >= native_ttft_ms {
        failure_reasons.push(format!(
            "restore-to-first-token did not beat native ({cachegen_ttft_ms:.3} ms >= {native_ttft_ms:.3} ms)"
        ));
    }
    if token_agreement < args.cachegen_min_token_agreement {
        failure_reasons.push(format!(
            "token agreement {token_agreement:.4} is below {:.4}",
            args.cachegen_min_token_agreement
        ));
    }
    if p99_decode_regression > args.cachegen_max_p99_decode_regression {
        failure_reasons.push(format!(
            "p99 decode regression {p99_decode_regression:.4} exceeds {:.4}",
            args.cachegen_max_p99_decode_regression
        ));
    }
    if let Some(limit) = args.cachegen_max_peak_working_bytes
        && archive.estimated_peak_codec_working_bytes > limit
    {
        failure_reasons.push(format!(
            "estimated codec working set {} bytes exceeds {limit} bytes",
            archive.estimated_peak_codec_working_bytes
        ));
    }

    Ok(CacheGenGateReport {
        passed: failure_reasons.is_empty(),
        failure_reasons,
        restore_path: "native-device",
        continuation_steps: args.cachegen_continuation_steps,
        native_storage_bytes,
        cachegen_storage_bytes,
        compression_ratio,
        tile_count: archive.tile_count,
        encode_ms,
        scalar_oracle_decode_ms,
        native_write_ms: persisted.native_write_ms,
        cachegen_write_ms: persisted.cachegen_write_ms,
        native_persist_ms: persisted.native_write_ms,
        cachegen_persist_ms: encode_ms + persisted.cachegen_write_ms,
        native_read_ms: persisted.native_read_ms,
        cachegen_read_ms: persisted.cachegen_read_ms,
        native_import_ms,
        cachegen_import_ms,
        native_ttft_ms,
        cachegen_ttft_ms,
        native_decode_tokens_per_second: tokens_per_second(&continuation.native_decode_ms),
        cachegen_decode_tokens_per_second: tokens_per_second(&continuation.cachegen_decode_ms),
        native_p99_decode_ms,
        cachegen_p99_decode_ms,
        p99_decode_regression,
        matching_tokens: continuation.matching_tokens,
        token_agreement,
        first_token_mismatch_step: continuation.first_token_mismatch_step,
        mean_entropy_abs_drift: mean(&continuation.entropy_abs_drift),
        max_entropy_abs_drift: max_or_zero(&continuation.entropy_abs_drift),
        mean_top_logprob_abs_drift: mean(&continuation.top_logprob_abs_drift),
        max_top_logprob_abs_drift: max_or_zero(&continuation.top_logprob_abs_drift),
        estimated_peak_codec_working_bytes: archive.estimated_peak_codec_working_bytes,
        min_token_agreement: args.cachegen_min_token_agreement,
        max_p99_decode_regression: args.cachegen_max_p99_decode_regression,
        max_peak_codec_working_bytes: args.cachegen_max_peak_working_bytes,
    })
}

fn import_cachegen_session(
    model: &StageModel,
    kv_desc: &RuntimeKvPageDesc,
    archive: &[u8],
    recurrent: &[u8],
    prefix: &[i32],
) -> Result<StageSession> {
    let mut session = model
        .create_session()
        .context("create CacheGen gate session")?;
    session
        .import_cachegen_kv_page(kv_desc, archive)
        .context("import CacheGen archive directly into resident KV")?;
    session
        .import_recurrent_state_for_token_count(recurrent, prefix.len() as u64)
        .context("import CacheGen gate recurrent state")?;
    Ok(session)
}

fn import_session(
    model: &StageModel,
    payload: &LocalStatePayload,
    prefix: &[i32],
) -> Result<StageSession> {
    let LocalStatePayload::KvRecurrent {
        kv_desc: Some(kv_desc),
        kv,
        recurrent,
    } = payload
    else {
        bail!("CacheGen gate internal payload is not KV-recurrent");
    };
    let mut session = model.create_session().context("create gate session")?;
    session
        .import_kv_page(kv_desc, kv)
        .context("import gate KV page")?;
    session
        .import_recurrent_state_for_token_count(recurrent, prefix.len() as u64)
        .context("import gate recurrent state")?;
    Ok(session)
}

struct ContinuationComparison {
    native_decode_ms: Vec<f64>,
    cachegen_decode_ms: Vec<f64>,
    matching_tokens: usize,
    first_token_mismatch_step: Option<usize>,
    entropy_abs_drift: Vec<f64>,
    top_logprob_abs_drift: Vec<f64>,
}

struct NativeContinuation {
    predicted_tokens: Vec<i32>,
    signals: Vec<TokenSignal>,
    decode_ms: Vec<f64>,
}

fn run_native_continuation(
    native: &mut StageSession,
    mut token: i32,
    steps: usize,
) -> Result<NativeContinuation> {
    let mut predicted_tokens = Vec::with_capacity(steps);
    let mut signals = Vec::with_capacity(steps);
    let mut decode_ms = Vec::with_capacity(steps);
    for _ in 0..steps {
        let started = Instant::now();
        let prediction = native.decode_step(token).context("native gate decode")?;
        decode_ms.push(elapsed_ms(started));
        signals.push(
            native
                .last_token_signal()
                .context("native gate token signal")?,
        );
        predicted_tokens.push(prediction);
        token = prediction;
    }
    Ok(NativeContinuation {
        predicted_tokens,
        signals,
        decode_ms,
    })
}

fn compare_cachegen_continuation(
    cachegen: &mut StageSession,
    first_token: i32,
    native: NativeContinuation,
) -> Result<ContinuationComparison> {
    let steps = native.predicted_tokens.len();
    let mut cachegen_decode_ms = Vec::with_capacity(steps);
    let mut matching_tokens = 0usize;
    let mut first_token_mismatch_step = None;
    let mut entropy_abs_drift = Vec::with_capacity(steps);
    let mut top_logprob_abs_drift = Vec::with_capacity(steps);
    for step in 0..steps {
        let token = if step == 0 {
            first_token
        } else {
            native.predicted_tokens[step - 1]
        };
        let started = Instant::now();
        let cachegen_prediction = cachegen
            .decode_step(token)
            .context("CacheGen gate decode")?;
        cachegen_decode_ms.push(elapsed_ms(started));
        let cachegen_signal = cachegen
            .last_token_signal()
            .context("CacheGen gate token signal")?;
        let native_prediction = native.predicted_tokens[step];
        let native_signal = native.signals[step];
        if native_prediction == cachegen_prediction {
            matching_tokens += 1;
        } else if first_token_mismatch_step.is_none() {
            first_token_mismatch_step = Some(step);
        }
        entropy_abs_drift.push(f64::from(
            (native_signal.entropy - cachegen_signal.entropy).abs(),
        ));
        top_logprob_abs_drift.push(f64::from(
            (native_signal.top_logprob - cachegen_signal.top_logprob).abs(),
        ));
    }
    Ok(ContinuationComparison {
        native_decode_ms: native.decode_ms,
        cachegen_decode_ms,
        matching_tokens,
        first_token_mismatch_step,
        entropy_abs_drift,
        top_logprob_abs_drift,
    })
}

fn encode_kv_archive(desc: &RuntimeKvPageDesc, raw: &[u8]) -> Result<CacheGenArchive> {
    desc.validate_payload(raw.len())?;
    encode_f16_page(&page_layout(desc)?, raw)
}

fn decode_kv_archive(desc: &RuntimeKvPageDesc, archive: &[u8]) -> Result<Vec<u8>> {
    let raw_len = usize::try_from(desc.payload_bytes).context("descriptor length exceeds usize")?;
    desc.validate_payload(raw_len)?;
    decode_f16_page(archive, raw_len)
}

fn page_layout(desc: &RuntimeKvPageDesc) -> Result<PageLayout> {
    let components = if desc.component_count == 0 {
        vec![component_layout(
            desc.token_count,
            desc.layer_count,
            desc.k_type,
            desc.v_type,
            desc.k_row_bytes,
            desc.v_row_bytes,
            desc.v_element_bytes,
            desc.k_idx_row_bytes,
            0,
            desc.payload_bytes,
            desc.flags,
        )?]
    } else {
        desc.components
            .iter()
            .take(desc.component_count as usize)
            .map(|component| {
                component_layout(
                    component.token_count,
                    component.layer_count,
                    component.k_type,
                    component.v_type,
                    component.k_row_bytes,
                    component.v_row_bytes,
                    component.v_element_bytes,
                    component.k_idx_row_bytes,
                    component.payload_offset,
                    component.payload_bytes,
                    component.flags,
                )
            })
            .collect::<Result<Vec<_>>>()?
    };
    Ok(PageLayout {
        payload_bytes: desc.payload_bytes,
        components,
    })
}

#[allow(clippy::too_many_arguments)]
fn component_layout(
    token_count: u64,
    layer_count: u32,
    k_type: u32,
    v_type: u32,
    k_row_bytes: u32,
    v_row_bytes: u32,
    v_element_bytes: u32,
    k_idx_row_bytes: u32,
    payload_offset: u64,
    payload_bytes: u64,
    flags: u64,
) -> Result<ComponentLayout> {
    if k_type != GGML_TYPE_F16 || v_type != GGML_TYPE_F16 {
        bail!("CacheGen gate only accepts runtime F16 K/V pages");
    }
    Ok(ComponentLayout {
        token_count,
        layer_count,
        k_row_bytes,
        v_row_bytes,
        v_element_bytes,
        k_idx_row_bytes,
        payload_offset,
        payload_bytes,
        v_transposed: flags & KV_PAGE_FLAG_V_TRANSPOSED != 0,
    })
}

fn persist_and_read_payloads(
    native_kv: &[u8],
    recurrent: &[u8],
    cachegen_archive: &[u8],
) -> Result<PersistedPayloads> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock precedes Unix epoch")?
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "skippy-cachegen-gate-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&root).with_context(|| format!("create {}", root.display()))?;
    let native_path = root.join("native.bin");
    let cachegen_path = root.join("cachegen.bin");
    let result = (|| {
        let native_write_ms = write_payload(&native_path, native_kv, recurrent)?;
        let cachegen_write_ms = write_payload(&cachegen_path, cachegen_archive, recurrent)?;
        let (native_bytes, native_read_ms) = read_payload(&native_path)?;
        let (cachegen_bytes, cachegen_read_ms) = read_payload(&cachegen_path)?;
        let native_split = native_kv.len();
        let cachegen_split = cachegen_archive.len();
        if native_bytes.len() != native_split + recurrent.len()
            || cachegen_bytes.len() != cachegen_split + recurrent.len()
        {
            bail!("persisted gate payload length mismatch");
        }
        Ok(PersistedPayloads {
            native_kv: native_bytes[..native_split].to_vec(),
            native_recurrent: native_bytes[native_split..].to_vec(),
            cachegen_archive: cachegen_bytes[..cachegen_split].to_vec(),
            cachegen_recurrent: cachegen_bytes[cachegen_split..].to_vec(),
            native_write_ms,
            cachegen_write_ms,
            native_read_ms,
            cachegen_read_ms,
        })
    })();
    let _ = fs::remove_dir_all(root);
    result
}

fn write_payload(path: &Path, first: &[u8], second: &[u8]) -> Result<f64> {
    let started = Instant::now();
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    file.write_all(first)?;
    file.write_all(second)?;
    file.sync_all()?;
    Ok(elapsed_ms(started))
}

fn read_payload(path: &PathBuf) -> Result<(Vec<u8>, f64)> {
    let started = Instant::now();
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    black_box(&bytes);
    Ok((bytes, elapsed_ms(started)))
}

fn percentile_99(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len().saturating_sub(1));
    sorted.get(index).copied().unwrap_or(0.0)
}

fn relative_regression(candidate: f64, baseline: f64) -> f64 {
    if baseline <= f64::EPSILON {
        if candidate <= baseline {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (candidate - baseline) / baseline
    }
}

fn tokens_per_second(samples: &[f64]) -> f64 {
    let total_ms: f64 = samples.iter().sum();
    if total_ms <= f64::EPSILON {
        0.0
    } else {
        samples.len() as f64 * 1000.0 / total_ms
    }
}

fn restore_to_first_token_ms(read_ms: f64, import_ms: f64, decode_ms: &[f64]) -> f64 {
    read_ms + import_ms + decode_ms.first().copied().unwrap_or(0.0)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn max_or_zero(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_cache::cachegen::lmcache::MAX_TOKENS_PER_CHUNK;

    fn f16_bytes(values: usize) -> Vec<u8> {
        (0..values)
            .flat_map(|index| {
                skippy_protocol::binary::f32_to_f16_bits(index as f32 / 17.0).to_le_bytes()
            })
            .collect()
    }

    fn descriptor(flags: u64) -> RuntimeKvPageDesc {
        descriptor_with_tokens(flags, 4)
    }

    fn descriptor_with_tokens(flags: u64, token_count: u64) -> RuntimeKvPageDesc {
        let payload_bytes = token_count * 2 * 2 * 6;
        RuntimeKvPageDesc {
            version: 1,
            layer_start: 0,
            layer_end: 2,
            token_start: 0,
            token_count,
            layer_count: 2,
            k_type: GGML_TYPE_F16,
            v_type: GGML_TYPE_F16,
            k_row_bytes: 6,
            v_row_bytes: 6,
            v_element_bytes: 2,
            k_idx_row_bytes: 0,
            payload_bytes,
            flags,
            codec: 1,
            component_count: 0,
            components: Default::default(),
        }
    }

    #[test]
    fn archive_roundtrip_preserves_geometry_and_length() {
        let desc = descriptor(0);
        let raw = f16_bytes(48);
        let archive = encode_kv_archive(&desc, &raw).expect("encode");
        let decoded = decode_kv_archive(&desc, &archive.bytes).expect("decode");
        assert_eq!(decoded.len(), raw.len());
        assert_eq!(archive.tile_count, 4);
        assert_ne!(decoded, raw, "fixture must exercise lossy quantization");
    }

    #[test]
    fn transposed_v_layout_is_restored_before_native_import() {
        let desc = descriptor(KV_PAGE_FLAG_V_TRANSPOSED);
        let raw = f16_bytes(48);
        let archive = encode_kv_archive(&desc, &raw).expect("encode");
        let decoded = decode_kv_archive(&desc, &archive.bytes).expect("decode");
        assert_eq!(decoded.len(), raw.len());
        assert_eq!(archive.tile_count, 4);
    }

    #[test]
    fn archive_rejects_uncovered_descriptor_bytes() {
        let mut desc = descriptor(0);
        desc.payload_bytes += 2;
        let raw = f16_bytes(49);
        assert!(encode_kv_archive(&desc, &raw).is_err());
    }

    #[test]
    fn archive_chunks_long_transposed_pages_at_the_reference_tile_size() {
        let token_count = MAX_TOKENS_PER_CHUNK as u64 + 4;
        let desc = descriptor_with_tokens(KV_PAGE_FLAG_V_TRANSPOSED, token_count);
        let raw = f16_bytes(desc.payload_bytes as usize / 2);
        let archive = encode_kv_archive(&desc, &raw).expect("encode");
        let decoded = decode_kv_archive(&desc, &archive.bytes).expect("decode");
        assert_eq!(decoded.len(), raw.len());
        assert_eq!(archive.tile_count, 8);
    }

    #[test]
    fn restore_timing_contains_only_persisted_read_native_import_and_first_decode() {
        let ttft_ms = restore_to_first_token_ms(12.0, 34.0, &[5.0, 999.0]);
        assert_eq!(ttft_ms, 51.0);
    }
}
