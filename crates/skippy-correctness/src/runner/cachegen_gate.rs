use std::{
    collections::BTreeMap,
    fs::{self, File},
    hint::black_box,
    io::Write,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_cache::cachegen::lmcache::{
    MAX_TOKENS_PER_CHUNK, bins_for_layer, decode_f16_segment, encode_f16_segment,
};
use skippy_runtime::{
    GGML_TYPE_F16, KV_PAGE_FLAG_V_TRANSPOSED, RuntimeKvPageDesc, StageModel, StageSession,
};

use crate::report::CacheGenGateReport;

use super::{
    stage_execution::{BinaryStateHandoffConfig, elapsed_ms},
    state_handoff::LocalStatePayload,
};

const ARCHIVE_MAGIC: [u8; 4] = *b"CKG1";
const ARCHIVE_HEADER_BYTES: usize = 16;
const RECORD_HEADER_BYTES: usize = 52;
const CACHEGEN_ROWS_PER_TILE: usize = MAX_TOKENS_PER_CHUNK;
const RECORD_CACHEGEN: u8 = 0;
const RECORD_EXACT: u8 = 1;
const RECORD_CACHEGEN_TRANSPOSED: u8 = 2;

type ByteRange = (usize, usize);
type TransposedRegion = (usize, usize, usize, usize);
type TransposedCoverage = BTreeMap<TransposedRegion, Vec<ByteRange>>;

struct CacheGenArchive {
    bytes: Vec<u8>,
    tile_count: usize,
    estimated_peak_codec_working_bytes: usize,
}

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
    let decode_started = Instant::now();
    let decoded_kv = decode_kv_archive(kv_desc, &persisted.cachegen_archive)?;
    let decode_ms = elapsed_ms(decode_started);

    let native_payload = LocalStatePayload::KvRecurrent {
        kv_desc: Some(kv_desc.clone()),
        kv: persisted.native_kv,
        recurrent: persisted.native_recurrent,
    };
    let cachegen_payload = LocalStatePayload::KvRecurrent {
        kv_desc: Some(kv_desc.clone()),
        kv: decoded_kv,
        recurrent: persisted.cachegen_recurrent,
    };

    let native_import_started = Instant::now();
    let mut native = import_session(model, &native_payload, prefix)?;
    let native_import_ms = elapsed_ms(native_import_started);
    let cachegen_import_started = Instant::now();
    let mut cachegen = import_session(model, &cachegen_payload, prefix)?;
    let cachegen_import_ms = elapsed_ms(cachegen_import_started);

    let continuation = compare_continuation(
        &mut native,
        &mut cachegen,
        continuation,
        args.cachegen_continuation_steps,
    )?;
    let native_p99_decode_ms = percentile_99(&continuation.native_decode_ms);
    let cachegen_p99_decode_ms = percentile_99(&continuation.cachegen_decode_ms);
    let p99_decode_regression = relative_regression(cachegen_p99_decode_ms, native_p99_decode_ms);
    let native_ttft_ms = persisted.native_read_ms
        + native_import_ms
        + continuation
            .native_decode_ms
            .first()
            .copied()
            .unwrap_or(0.0);
    let cachegen_ttft_ms = persisted.cachegen_read_ms
        + decode_ms
        + cachegen_import_ms
        + continuation
            .cachegen_decode_ms
            .first()
            .copied()
            .unwrap_or(0.0);
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
        continuation_steps: args.cachegen_continuation_steps,
        native_storage_bytes,
        cachegen_storage_bytes,
        compression_ratio,
        tile_count: archive.tile_count,
        encode_ms,
        decode_ms,
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

fn compare_continuation(
    native: &mut StageSession,
    cachegen: &mut StageSession,
    mut token: i32,
    steps: usize,
) -> Result<ContinuationComparison> {
    let mut native_decode_ms = Vec::with_capacity(steps);
    let mut cachegen_decode_ms = Vec::with_capacity(steps);
    let mut matching_tokens = 0usize;
    let mut first_token_mismatch_step = None;
    let mut entropy_abs_drift = Vec::with_capacity(steps);
    let mut top_logprob_abs_drift = Vec::with_capacity(steps);
    for step in 0..steps {
        let started = Instant::now();
        let native_prediction = native.decode_step(token).context("native gate decode")?;
        native_decode_ms.push(elapsed_ms(started));
        let native_signal = native
            .last_token_signal()
            .context("native gate token signal")?;

        let started = Instant::now();
        let cachegen_prediction = cachegen
            .decode_step(token)
            .context("CacheGen gate decode")?;
        cachegen_decode_ms.push(elapsed_ms(started));
        let cachegen_signal = cachegen
            .last_token_signal()
            .context("CacheGen gate token signal")?;
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
        token = native_prediction;
    }
    Ok(ContinuationComparison {
        native_decode_ms,
        cachegen_decode_ms,
        matching_tokens,
        first_token_mismatch_step,
        entropy_abs_drift,
        top_logprob_abs_drift,
    })
}

fn encode_kv_archive(desc: &RuntimeKvPageDesc, raw: &[u8]) -> Result<CacheGenArchive> {
    desc.validate_payload(raw.len())?;
    let mut records = Vec::new();
    if desc.component_count == 0 {
        encode_component(
            ComponentLayout {
                token_count: desc.token_count,
                layer_count: desc.layer_count,
                k_type: desc.k_type,
                v_type: desc.v_type,
                k_row_bytes: desc.k_row_bytes,
                v_row_bytes: desc.v_row_bytes,
                v_element_bytes: desc.v_element_bytes,
                k_idx_row_bytes: desc.k_idx_row_bytes,
                payload_offset: 0,
                payload_bytes: desc.payload_bytes,
                flags: desc.flags,
            },
            raw,
            &mut records,
        )?;
    } else {
        for component in desc.components.iter().take(desc.component_count as usize) {
            encode_component(
                ComponentLayout {
                    token_count: component.token_count,
                    layer_count: component.layer_count,
                    k_type: component.k_type,
                    v_type: component.v_type,
                    k_row_bytes: component.k_row_bytes,
                    v_row_bytes: component.v_row_bytes,
                    v_element_bytes: component.v_element_bytes,
                    k_idx_row_bytes: component.k_idx_row_bytes,
                    payload_offset: component.payload_offset,
                    payload_bytes: component.payload_bytes,
                    flags: component.flags,
                },
                raw,
                &mut records,
            )?;
        }
    }
    records.sort_by_key(|record| (record.output_offset, record.token_start));
    validate_record_coverage(&records, raw.len())?;

    let record_count = u32::try_from(records.len()).context("too many CacheGen archive records")?;
    let payload_bytes = records
        .iter()
        .try_fold(ARCHIVE_HEADER_BYTES, |total, record| {
            total
                .checked_add(RECORD_HEADER_BYTES)
                .and_then(|value| value.checked_add(record.payload.len()))
                .ok_or_else(|| anyhow!("CacheGen archive length overflow"))
        })?;
    let mut bytes = Vec::with_capacity(payload_bytes);
    bytes.extend_from_slice(&ARCHIVE_MAGIC);
    bytes.extend_from_slice(&(raw.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&record_count.to_le_bytes());
    let mut largest_working_set = 0usize;
    let mut tile_count = 0usize;
    for record in records {
        bytes.push(record.kind);
        bytes.push(record.element_bytes);
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&(record.output_offset as u64).to_le_bytes());
        bytes.extend_from_slice(&(record.decoded_len as u64).to_le_bytes());
        bytes.extend_from_slice(&record.token_count.to_le_bytes());
        bytes.extend_from_slice(&record.token_start.to_le_bytes());
        bytes.extend_from_slice(&record.total_tokens.to_le_bytes());
        bytes.extend_from_slice(&(record.payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&record.payload);
        if record.kind != RECORD_EXACT {
            tile_count += 1;
            let values = record.decoded_len / 2;
            let codec_working = record
                .decoded_len
                .saturating_add(values.saturating_mul(5))
                .saturating_add(record.payload.len());
            largest_working_set = largest_working_set.max(codec_working);
        }
    }
    Ok(CacheGenArchive {
        estimated_peak_codec_working_bytes: bytes
            .len()
            .saturating_add(raw.len())
            .saturating_add(largest_working_set),
        bytes,
        tile_count,
    })
}

#[derive(Clone, Copy)]
struct ComponentLayout {
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
}

struct ArchiveRecord {
    kind: u8,
    element_bytes: u8,
    output_offset: usize,
    decoded_len: usize,
    token_count: u64,
    token_start: u64,
    total_tokens: u64,
    payload: Vec<u8>,
}

fn encode_component(
    component: ComponentLayout,
    raw: &[u8],
    records: &mut Vec<ArchiveRecord>,
) -> Result<()> {
    if component.k_type != GGML_TYPE_F16 || component.v_type != GGML_TYPE_F16 {
        bail!("CacheGen gate only accepts runtime F16 K/V pages");
    }
    let token_count =
        usize::try_from(component.token_count).context("token count exceeds usize")?;
    let layer_count = component.layer_count as usize;
    let k_row = component.k_row_bytes as usize;
    let v_row = component.v_row_bytes as usize;
    if token_count == 0 || layer_count == 0 || k_row == 0 || v_row == 0 {
        bail!("CacheGen gate received empty KV geometry");
    }
    if !k_row.is_multiple_of(2) || !v_row.is_multiple_of(2) {
        bail!("CacheGen gate requires whole F16 rows");
    }
    let base = usize::try_from(component.payload_offset).context("payload offset exceeds usize")?;
    let component_len =
        usize::try_from(component.payload_bytes).context("payload size exceeds usize")?;
    let k_layer_bytes = token_count
        .checked_mul(k_row)
        .context("K layer size overflow")?;
    let v_layer_bytes = token_count
        .checked_mul(v_row)
        .context("V layer size overflow")?;
    let k_bytes = layer_count
        .checked_mul(k_layer_bytes)
        .context("K size overflow")?;
    let v_bytes = layer_count
        .checked_mul(v_layer_bytes)
        .context("V size overflow")?;
    let k_idx_bytes = layer_count
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(component.k_idx_row_bytes as usize))
        .context("K-index size overflow")?;
    if k_bytes
        .checked_add(v_bytes)
        .and_then(|value| value.checked_add(k_idx_bytes))
        != Some(component_len)
    {
        bail!("CacheGen gate descriptor geometry does not cover its component payload");
    }
    let end = base
        .checked_add(component_len)
        .context("component range overflow")?;
    let component_raw = raw
        .get(base..end)
        .ok_or_else(|| anyhow!("component range exceeds KV payload"))?;

    for layer in 0..layer_count {
        let layer_offset = layer * k_layer_bytes;
        for row_start in (0..token_count).step_by(CACHEGEN_ROWS_PER_TILE) {
            let rows = (token_count - row_start).min(CACHEGEN_ROWS_PER_TILE);
            let local_offset = layer_offset + row_start * k_row;
            let tile = &component_raw[local_offset..local_offset + rows * k_row];
            records.push(ArchiveRecord {
                kind: RECORD_CACHEGEN,
                element_bytes: 2,
                output_offset: base + local_offset,
                decoded_len: tile.len(),
                token_count: rows as u64,
                token_start: 0,
                total_tokens: 0,
                payload: encode_f16_segment(
                    tile,
                    k_row / 2,
                    bins_for_layer(layer, layer_count, true),
                )?,
            });
        }
    }

    let v_base = k_bytes;
    let v_transposed = component.flags & KV_PAGE_FLAG_V_TRANSPOSED != 0;
    for layer in 0..layer_count {
        let layer_offset = v_base + layer * v_layer_bytes;
        let layer_tile = &component_raw[layer_offset..layer_offset + v_layer_bytes];
        for row_start in (0..token_count).step_by(CACHEGEN_ROWS_PER_TILE) {
            let rows = (token_count - row_start).min(CACHEGEN_ROWS_PER_TILE);
            let (kind, output_offset, encoded_input, token_start, total_tokens) = if v_transposed {
                if component.v_element_bytes != 2 {
                    bail!("transposed F16 V page must declare two-byte elements");
                }
                (
                    RECORD_CACHEGEN_TRANSPOSED,
                    base + layer_offset,
                    transpose_range_to_token_major(
                        layer_tile,
                        token_count,
                        row_start,
                        rows,
                        v_row / 2,
                        2,
                    )?,
                    row_start as u64,
                    token_count as u64,
                )
            } else {
                let local_offset = layer_offset + row_start * v_row;
                (
                    RECORD_CACHEGEN,
                    base + local_offset,
                    component_raw[local_offset..local_offset + rows * v_row].to_vec(),
                    0,
                    0,
                )
            };
            records.push(ArchiveRecord {
                kind,
                element_bytes: 2,
                output_offset,
                decoded_len: encoded_input.len(),
                token_count: rows as u64,
                token_start,
                total_tokens,
                payload: encode_f16_segment(
                    &encoded_input,
                    v_row / 2,
                    bins_for_layer(layer, layer_count, false),
                )?,
            });
        }
    }
    if k_idx_bytes > 0 {
        let local_offset = k_bytes + v_bytes;
        records.push(ArchiveRecord {
            kind: RECORD_EXACT,
            element_bytes: 1,
            output_offset: base + local_offset,
            decoded_len: k_idx_bytes,
            token_count: component.token_count,
            token_start: 0,
            total_tokens: 0,
            payload: component_raw[local_offset..local_offset + k_idx_bytes].to_vec(),
        });
    }
    Ok(())
}

fn decode_kv_archive(desc: &RuntimeKvPageDesc, archive: &[u8]) -> Result<Vec<u8>> {
    if archive.len() < ARCHIVE_HEADER_BYTES || archive[..4] != ARCHIVE_MAGIC {
        bail!("invalid CacheGen gate archive header");
    }
    let raw_len =
        usize::try_from(read_u64(&archive[4..12])?).context("raw length exceeds usize")?;
    if raw_len != usize::try_from(desc.payload_bytes).context("descriptor length exceeds usize")? {
        bail!("CacheGen archive raw length disagrees with KV descriptor");
    }
    let record_count = read_u32(&archive[12..16])? as usize;
    let mut cursor = ARCHIVE_HEADER_BYTES;
    let mut decoded = vec![0u8; raw_len];
    let mut coverage = Vec::with_capacity(record_count);
    let mut transposed_coverage = TransposedCoverage::new();
    let mut decoded_total = 0usize;
    for _ in 0..record_count {
        let header = archive
            .get(cursor..cursor + RECORD_HEADER_BYTES)
            .ok_or_else(|| anyhow!("truncated CacheGen archive record header"))?;
        cursor += RECORD_HEADER_BYTES;
        let kind = header[0];
        let element_bytes = header[1] as usize;
        if header[2..4] != [0, 0] {
            bail!("CacheGen archive record reserved bytes are non-zero");
        }
        let output_offset = usize::try_from(read_u64(&header[4..12])?)
            .context("record output offset exceeds usize")?;
        let decoded_len = usize::try_from(read_u64(&header[12..20])?)
            .context("record decoded length exceeds usize")?;
        let token_count = usize::try_from(read_u64(&header[20..28])?)
            .context("record token count exceeds usize")?;
        let token_start = usize::try_from(read_u64(&header[28..36])?)
            .context("record token start exceeds usize")?;
        let total_tokens = usize::try_from(read_u64(&header[36..44])?)
            .context("record total token count exceeds usize")?;
        let payload_len = usize::try_from(read_u64(&header[44..52])?)
            .context("record payload length exceeds usize")?;
        let payload = archive
            .get(cursor..cursor + payload_len)
            .ok_or_else(|| anyhow!("truncated CacheGen archive record payload"))?;
        cursor += payload_len;
        decoded_total = decoded_total
            .checked_add(decoded_len)
            .context("decoded archive size overflow")?;
        match kind {
            RECORD_CACHEGEN => {
                let output_end = output_offset
                    .checked_add(decoded_len)
                    .context("record output range overflow")?;
                let output = decoded
                    .get_mut(output_offset..output_end)
                    .ok_or_else(|| anyhow!("record output range exceeds KV payload"))?;
                let tile = decode_f16_segment(payload)?;
                if tile.len() != decoded_len {
                    bail!("decoded CacheGen tile length disagrees with archive record");
                }
                output.copy_from_slice(&tile);
                coverage.push((output_offset, output_end));
            }
            RECORD_EXACT => {
                let output_end = output_offset
                    .checked_add(decoded_len)
                    .context("record output range overflow")?;
                let output = decoded
                    .get_mut(output_offset..output_end)
                    .ok_or_else(|| anyhow!("record output range exceeds KV payload"))?;
                if payload.len() != decoded_len {
                    bail!("exact archive record length mismatch");
                }
                output.copy_from_slice(payload);
                coverage.push((output_offset, output_end));
            }
            RECORD_CACHEGEN_TRANSPOSED => {
                let token_major = decode_f16_segment(payload)?;
                if token_major.len() != decoded_len
                    || element_bytes == 0
                    || token_count == 0
                    || total_tokens == 0
                    || token_start
                        .checked_add(token_count)
                        .is_none_or(|end| end > total_tokens)
                {
                    bail!("invalid transposed CacheGen archive record geometry");
                }
                let dims = decoded_len
                    .checked_div(token_count)
                    .and_then(|value| value.checked_div(element_bytes))
                    .ok_or_else(|| anyhow!("transposed record geometry overflow"))?;
                let layer_bytes = total_tokens
                    .checked_mul(dims)
                    .and_then(|value| value.checked_mul(element_bytes))
                    .context("transposed output size overflow")?;
                let output_end = output_offset
                    .checked_add(layer_bytes)
                    .context("transposed output range overflow")?;
                let output = decoded
                    .get_mut(output_offset..output_end)
                    .ok_or_else(|| anyhow!("transposed output range exceeds KV payload"))?;
                transpose_range_from_token_major(
                    &token_major,
                    output,
                    total_tokens,
                    token_start,
                    token_count,
                    dims,
                    element_bytes,
                )?;
                transposed_coverage
                    .entry((output_offset, total_tokens, dims, element_bytes))
                    .or_default()
                    .push((token_start, token_start + token_count));
            }
            _ => bail!("unknown CacheGen archive record kind {kind}"),
        }
    }
    if cursor != archive.len() {
        bail!("CacheGen archive has trailing bytes");
    }
    validate_decoded_coverage(
        &mut coverage,
        &mut transposed_coverage,
        raw_len,
        decoded_total,
    )?;
    desc.validate_payload(decoded.len())?;
    Ok(decoded)
}

fn validate_record_coverage(records: &[ArchiveRecord], raw_len: usize) -> Result<()> {
    let decoded_total = records.iter().try_fold(0usize, |total, record| {
        total
            .checked_add(record.decoded_len)
            .ok_or_else(|| anyhow!("archive decoded size overflow"))
    })?;
    if decoded_total != raw_len {
        bail!("CacheGen archive records do not account for the KV payload");
    }
    Ok(())
}

fn validate_decoded_coverage(
    coverage: &mut Vec<ByteRange>,
    transposed: &mut TransposedCoverage,
    raw_len: usize,
    decoded_total: usize,
) -> Result<()> {
    if decoded_total != raw_len {
        bail!("CacheGen archive decoded bytes do not account for the KV payload");
    }
    for (&(output_offset, total_tokens, dims, element_bytes), ranges) in transposed.iter_mut() {
        ranges.sort_unstable();
        let mut next_token = 0usize;
        for &(start, end) in ranges.iter() {
            if start != next_token || end < start {
                bail!("transposed CacheGen records overlap or leave a token gap");
            }
            next_token = end;
        }
        if next_token != total_tokens {
            bail!("transposed CacheGen records leave a token gap");
        }
        let end = output_offset
            .checked_add(
                total_tokens
                    .checked_mul(dims)
                    .and_then(|value| value.checked_mul(element_bytes))
                    .context("transposed coverage size overflow")?,
            )
            .context("transposed coverage range overflow")?;
        coverage.push((output_offset, end));
    }
    coverage.sort_unstable();
    let mut next = 0usize;
    for &(start, end) in coverage.iter() {
        if start != next || end < start {
            bail!("CacheGen archive records do not exactly cover the KV payload");
        }
        next = end;
    }
    if next != raw_len {
        bail!("CacheGen archive records leave a gap in the KV payload");
    }
    Ok(())
}

fn transpose_range_to_token_major(
    source: &[u8],
    total_tokens: usize,
    token_start: usize,
    token_count: usize,
    dims: usize,
    element_bytes: usize,
) -> Result<Vec<u8>> {
    let expected = total_tokens
        .checked_mul(dims)
        .and_then(|value| value.checked_mul(element_bytes))
        .context("transpose size overflow")?;
    if source.len() != expected {
        bail!("transposed V tile length does not match its geometry");
    }
    let output_len = token_count
        .checked_mul(dims)
        .and_then(|value| value.checked_mul(element_bytes))
        .context("transpose output size overflow")?;
    if token_start
        .checked_add(token_count)
        .is_none_or(|end| end > total_tokens)
    {
        bail!("transpose token range exceeds source geometry");
    }
    let mut output = vec![0u8; output_len];
    for local_token in 0..token_count {
        let token = token_start + local_token;
        for dim in 0..dims {
            let source_offset = (dim * total_tokens + token) * element_bytes;
            let output_offset = (local_token * dims + dim) * element_bytes;
            output[output_offset..output_offset + element_bytes]
                .copy_from_slice(&source[source_offset..source_offset + element_bytes]);
        }
    }
    Ok(output)
}

fn transpose_range_from_token_major(
    source: &[u8],
    output: &mut [u8],
    total_tokens: usize,
    token_start: usize,
    token_count: usize,
    dims: usize,
    element_bytes: usize,
) -> Result<()> {
    let expected_source = token_count
        .checked_mul(dims)
        .and_then(|value| value.checked_mul(element_bytes))
        .context("transpose source size overflow")?;
    let expected_output = total_tokens
        .checked_mul(dims)
        .and_then(|value| value.checked_mul(element_bytes))
        .context("transpose output size overflow")?;
    if source.len() != expected_source || output.len() != expected_output {
        bail!("transposed CacheGen decode length mismatch");
    }
    for local_token in 0..token_count {
        let token = token_start + local_token;
        for dim in 0..dims {
            let source_offset = (local_token * dims + dim) * element_bytes;
            let output_offset = (dim * total_tokens + token) * element_bytes;
            output[output_offset..output_offset + element_bytes]
                .copy_from_slice(&source[source_offset..source_offset + element_bytes]);
        }
    }
    Ok(())
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

fn read_u64(bytes: &[u8]) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes.try_into().context("u64 field length")?,
    ))
}

fn read_u32(bytes: &[u8]) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes.try_into().context("u32 field length")?,
    ))
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
        let token_count = CACHEGEN_ROWS_PER_TILE as u64 + 4;
        let desc = descriptor_with_tokens(KV_PAGE_FLAG_V_TRANSPOSED, token_count);
        let raw = f16_bytes(desc.payload_bytes as usize / 2);
        let archive = encode_kv_archive(&desc, &raw).expect("encode");
        let decoded = decode_kv_archive(&desc, &archive.bytes).expect("decode");
        assert_eq!(decoded.len(), raw.len());
        assert_eq!(archive.tile_count, 8);
    }
}
