//! Bounded CacheGen archives for complete native KV pages.
//!
//! A page archive covers every byte in the decoded runtime payload exactly
//! once. K/V records carry LMCache-compatible segments while auxiliary
//! indexer state remains exact. The archive is portable; native backends may
//! decode its validated records directly into resident KV tensors.

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};

use super::lmcache::{
    MAX_TOKENS_PER_CHUNK, bins_for_layer, decode_f16_segment, encode_f16_segment,
    validate_f16_segment,
};

pub const ARCHIVE_MAGIC: [u8; 4] = *b"CKG1";
pub const ARCHIVE_HEADER_BYTES: usize = 16;
pub const RECORD_HEADER_BYTES: usize = 52;

type ByteRange = (usize, usize);
type TransposedRegion = (usize, usize, usize, usize);
type TransposedCoverage = BTreeMap<TransposedRegion, Vec<ByteRange>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RecordKind {
    CacheGen = 0,
    Exact = 1,
    CacheGenTransposed = 2,
}

impl TryFrom<u8> for RecordKind {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::CacheGen),
            1 => Ok(Self::Exact),
            2 => Ok(Self::CacheGenTransposed),
            _ => bail!("unknown CacheGen archive record kind {value}"),
        }
    }
}

/// Native-page geometry for one independent base or sliding-window cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentLayout {
    pub token_count: u64,
    pub layer_count: u32,
    pub k_row_bytes: u32,
    pub v_row_bytes: u32,
    pub v_element_bytes: u32,
    pub k_idx_row_bytes: u32,
    pub payload_offset: u64,
    pub payload_bytes: u64,
    pub v_transposed: bool,
}

/// Complete native-page geometry. Components must cover `payload_bytes`
/// exactly and appear in output order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageLayout {
    pub payload_bytes: u64,
    pub components: Vec<ComponentLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheGenArchive {
    pub bytes: Vec<u8>,
    pub tile_count: usize,
    pub estimated_peak_codec_working_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    pub kind: RecordKind,
    pub element_bytes: usize,
    pub output_offset: usize,
    pub decoded_len: usize,
    pub token_count: usize,
    pub token_start: usize,
    pub total_tokens: usize,
    pub payload: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedArchive<'a> {
    pub raw_len: usize,
    pub records: Vec<Record<'a>>,
}

struct OwnedRecord {
    kind: RecordKind,
    element_bytes: u8,
    output_offset: usize,
    decoded_len: usize,
    token_count: u64,
    token_start: u64,
    total_tokens: u64,
    payload: Vec<u8>,
}

pub fn encode_f16_page(layout: &PageLayout, raw: &[u8]) -> Result<CacheGenArchive> {
    let raw_len = usize::try_from(layout.payload_bytes).context("page size exceeds usize")?;
    if raw.len() != raw_len {
        bail!("CacheGen page length disagrees with its layout");
    }
    validate_components(layout, raw_len)?;

    let mut records = Vec::new();
    for component in &layout.components {
        encode_component(*component, raw, &mut records)?;
    }
    records.sort_by_key(|record| (record.output_offset, record.token_start));
    validate_owned_record_coverage(&records, raw_len)?;

    let record_count = u32::try_from(records.len()).context("too many CacheGen archive records")?;
    let payload_bytes =
        records
            .iter()
            .try_fold(ARCHIVE_HEADER_BYTES, |total, record| -> Result<usize> {
                total
                    .checked_add(RECORD_HEADER_BYTES)
                    .and_then(|value| value.checked_add(record.payload.len()))
                    .ok_or_else(|| anyhow!("CacheGen archive length overflow"))
            })?;
    let mut bytes = Vec::with_capacity(payload_bytes);
    bytes.extend_from_slice(&ARCHIVE_MAGIC);
    bytes.extend_from_slice(&layout.payload_bytes.to_le_bytes());
    bytes.extend_from_slice(&record_count.to_le_bytes());

    let mut largest_working_set = 0usize;
    let mut tile_count = 0usize;
    for record in records {
        bytes.push(record.kind as u8);
        bytes.push(record.element_bytes);
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&(record.output_offset as u64).to_le_bytes());
        bytes.extend_from_slice(&(record.decoded_len as u64).to_le_bytes());
        bytes.extend_from_slice(&record.token_count.to_le_bytes());
        bytes.extend_from_slice(&record.token_start.to_le_bytes());
        bytes.extend_from_slice(&record.total_tokens.to_le_bytes());
        bytes.extend_from_slice(&(record.payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&record.payload);
        if record.kind != RecordKind::Exact {
            tile_count += 1;
            let values = record.decoded_len / 2;
            let codec_working = record
                .decoded_len
                .saturating_add(values.saturating_mul(5))
                .saturating_add(record.payload.len());
            largest_working_set = largest_working_set.max(codec_working);
        }
    }

    validate_archive(&bytes, raw_len).context("validate encoded CacheGen page")?;

    Ok(CacheGenArchive {
        estimated_peak_codec_working_bytes: bytes
            .len()
            .saturating_add(raw.len())
            .saturating_add(largest_working_set),
        bytes,
        tile_count,
    })
}

pub fn validate_archive(archive: &[u8], expected_raw_len: usize) -> Result<ValidatedArchive<'_>> {
    if archive.len() < ARCHIVE_HEADER_BYTES || archive[..4] != ARCHIVE_MAGIC {
        bail!("invalid CacheGen archive header");
    }
    let raw_len = usize::try_from(read_u64(&archive[4..12])?)
        .context("CacheGen archive raw length exceeds usize")?;
    if raw_len != expected_raw_len {
        bail!("CacheGen archive raw length disagrees with KV descriptor");
    }
    let record_count = read_u32(&archive[12..16])? as usize;
    let minimum_headers = record_count
        .checked_mul(RECORD_HEADER_BYTES)
        .and_then(|value| value.checked_add(ARCHIVE_HEADER_BYTES))
        .ok_or_else(|| anyhow!("CacheGen archive record table overflows"))?;
    if minimum_headers > archive.len() {
        bail!("truncated CacheGen archive record table");
    }

    let mut cursor = ARCHIVE_HEADER_BYTES;
    let mut records = Vec::with_capacity(record_count);
    let mut coverage = Vec::with_capacity(record_count);
    let mut transposed_coverage = TransposedCoverage::new();
    let mut decoded_total = 0usize;
    for _ in 0..record_count {
        let header = archive
            .get(cursor..cursor + RECORD_HEADER_BYTES)
            .ok_or_else(|| anyhow!("truncated CacheGen archive record header"))?;
        cursor += RECORD_HEADER_BYTES;
        let kind = RecordKind::try_from(header[0])?;
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
        let payload_end = cursor
            .checked_add(payload_len)
            .ok_or_else(|| anyhow!("record payload range overflow"))?;
        let payload = archive
            .get(cursor..payload_end)
            .ok_or_else(|| anyhow!("truncated CacheGen archive record payload"))?;
        cursor = payload_end;
        decoded_total = decoded_total
            .checked_add(decoded_len)
            .context("decoded archive size overflow")?;

        validate_record_geometry(Record {
            kind,
            element_bytes,
            output_offset,
            decoded_len,
            token_count,
            token_start,
            total_tokens,
            payload,
        })?;
        record_coverage(
            kind,
            output_offset,
            decoded_len,
            token_start,
            token_count,
            total_tokens,
            element_bytes,
            &mut coverage,
            &mut transposed_coverage,
        )?;
        records.push(Record {
            kind,
            element_bytes,
            output_offset,
            decoded_len,
            token_count,
            token_start,
            total_tokens,
            payload,
        });
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
    Ok(ValidatedArchive { raw_len, records })
}

pub fn decode_f16_page(archive: &[u8], expected_raw_len: usize) -> Result<Vec<u8>> {
    let validated = validate_archive(archive, expected_raw_len)?;
    let mut decoded = vec![0u8; validated.raw_len];
    for record in validated.records {
        match record.kind {
            RecordKind::CacheGen => {
                let output_end = record
                    .output_offset
                    .checked_add(record.decoded_len)
                    .context("record output range overflow")?;
                let output = decoded
                    .get_mut(record.output_offset..output_end)
                    .ok_or_else(|| anyhow!("record output range exceeds KV payload"))?;
                let tile = decode_f16_segment(record.payload)?;
                output.copy_from_slice(&tile);
            }
            RecordKind::Exact => {
                let output_end = record
                    .output_offset
                    .checked_add(record.decoded_len)
                    .context("record output range overflow")?;
                decoded[record.output_offset..output_end].copy_from_slice(record.payload);
            }
            RecordKind::CacheGenTransposed => {
                let token_major = decode_f16_segment(record.payload)?;
                let dims = record.decoded_len / record.token_count / record.element_bytes;
                let layer_bytes = record.total_tokens * dims * record.element_bytes;
                let output_end = record.output_offset + layer_bytes;
                transpose_range_from_token_major(
                    &token_major,
                    &mut decoded[record.output_offset..output_end],
                    record.total_tokens,
                    record.token_start,
                    record.token_count,
                    dims,
                    record.element_bytes,
                )?;
            }
        }
    }
    Ok(decoded)
}

fn validate_components(layout: &PageLayout, raw_len: usize) -> Result<()> {
    if layout.components.is_empty() {
        bail!("CacheGen page has no components");
    }
    let mut next = 0usize;
    for component in &layout.components {
        let offset =
            usize::try_from(component.payload_offset).context("component offset exceeds usize")?;
        let bytes =
            usize::try_from(component.payload_bytes).context("component size exceeds usize")?;
        if offset != next || component.token_count == 0 || component.layer_count == 0 {
            bail!("CacheGen components are empty, reordered, or leave a gap");
        }
        next = next
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("component range overflow"))?;
    }
    if next != raw_len {
        bail!("CacheGen components do not cover the native page");
    }
    Ok(())
}

fn encode_component(
    component: ComponentLayout,
    raw: &[u8],
    records: &mut Vec<OwnedRecord>,
) -> Result<()> {
    let token_count =
        usize::try_from(component.token_count).context("token count exceeds usize")?;
    let layer_count = component.layer_count as usize;
    let k_row = component.k_row_bytes as usize;
    if k_row == 0 || !k_row.is_multiple_of(2) {
        bail!("CacheGen requires non-empty whole-F16 K rows");
    }
    let base = usize::try_from(component.payload_offset).context("payload offset exceeds usize")?;
    let component_len =
        usize::try_from(component.payload_bytes).context("payload size exceeds usize")?;
    let k_layer_bytes = token_count
        .checked_mul(k_row)
        .context("K layer size overflow")?;
    let k_bytes = layer_count
        .checked_mul(k_layer_bytes)
        .context("K size overflow")?;
    let k_idx_bytes = layer_count
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(component.k_idx_row_bytes as usize))
        .context("K-index size overflow")?;
    let v_bytes = component_len
        .checked_sub(k_bytes)
        .and_then(|value| value.checked_sub(k_idx_bytes))
        .ok_or_else(|| anyhow!("component payload is shorter than K and indexer state"))?;
    if !v_bytes.is_multiple_of(layer_count) {
        bail!("V payload is not uniform across layers");
    }
    let v_layer_bytes = v_bytes / layer_count;
    if !v_layer_bytes.is_multiple_of(token_count) {
        bail!("V payload is not uniform across tokens");
    }
    let v_row = v_layer_bytes / token_count;
    if v_row == 0 || !v_row.is_multiple_of(2) {
        bail!("CacheGen requires non-empty whole-F16 V rows");
    }
    if component.v_transposed {
        if component.v_element_bytes != 2 {
            bail!("transposed F16 V page must declare two-byte elements");
        }
    } else if component.v_row_bytes as usize != v_row {
        bail!("non-transposed V payload disagrees with its row size");
    }

    let end = base
        .checked_add(component_len)
        .context("component range overflow")?;
    let component_raw = raw
        .get(base..end)
        .ok_or_else(|| anyhow!("component range exceeds KV payload"))?;

    for layer in 0..layer_count {
        let layer_offset = layer * k_layer_bytes;
        for row_start in (0..token_count).step_by(MAX_TOKENS_PER_CHUNK) {
            let rows = (token_count - row_start).min(MAX_TOKENS_PER_CHUNK);
            let local_offset = layer_offset + row_start * k_row;
            let tile = &component_raw[local_offset..local_offset + rows * k_row];
            records.push(OwnedRecord {
                kind: RecordKind::CacheGen,
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
    for layer in 0..layer_count {
        let layer_offset = v_base + layer * v_layer_bytes;
        let layer_tile = &component_raw[layer_offset..layer_offset + v_layer_bytes];
        for row_start in (0..token_count).step_by(MAX_TOKENS_PER_CHUNK) {
            let rows = (token_count - row_start).min(MAX_TOKENS_PER_CHUNK);
            let (kind, output_offset, encoded_input, token_start, total_tokens) =
                if component.v_transposed {
                    (
                        RecordKind::CacheGenTransposed,
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
                        RecordKind::CacheGen,
                        base + local_offset,
                        component_raw[local_offset..local_offset + rows * v_row].to_vec(),
                        0,
                        0,
                    )
                };
            records.push(OwnedRecord {
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
        records.push(OwnedRecord {
            kind: RecordKind::Exact,
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

fn validate_record_geometry(record: Record<'_>) -> Result<()> {
    if record.decoded_len == 0 || record.token_count == 0 {
        bail!("CacheGen archive record has empty geometry");
    }
    match record.kind {
        RecordKind::CacheGen | RecordKind::CacheGenTransposed => {
            if record.element_bytes != 2 {
                bail!("CacheGen F16 record has an invalid element size");
            }
            let segment = validate_f16_segment(record.payload)?;
            let expected = segment
                .rows
                .checked_mul(segment.channels)
                .and_then(|value| value.checked_mul(2))
                .ok_or_else(|| anyhow!("CacheGen segment decoded length overflow"))?;
            if segment.rows != record.token_count || expected != record.decoded_len {
                bail!("CacheGen segment geometry disagrees with its archive record");
            }
            if record.kind == RecordKind::CacheGenTransposed {
                if record.total_tokens == 0
                    || record
                        .token_start
                        .checked_add(record.token_count)
                        .is_none_or(|end| end > record.total_tokens)
                {
                    bail!("invalid transposed CacheGen archive record geometry");
                }
            } else if record.token_start != 0 || record.total_tokens != 0 {
                bail!("non-transposed CacheGen record carries transpose geometry");
            }
        }
        RecordKind::Exact => {
            if record.element_bytes != 1
                || record.payload.len() != record.decoded_len
                || record.token_start != 0
                || record.total_tokens != 0
            {
                bail!("exact CacheGen archive record has invalid geometry");
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_coverage(
    kind: RecordKind,
    output_offset: usize,
    decoded_len: usize,
    token_start: usize,
    token_count: usize,
    total_tokens: usize,
    element_bytes: usize,
    coverage: &mut Vec<ByteRange>,
    transposed: &mut TransposedCoverage,
) -> Result<()> {
    if kind == RecordKind::CacheGenTransposed {
        let dims = decoded_len
            .checked_div(token_count)
            .and_then(|value| value.checked_div(element_bytes))
            .filter(|dims| *dims > 0)
            .ok_or_else(|| anyhow!("transposed record geometry overflow"))?;
        transposed
            .entry((output_offset, total_tokens, dims, element_bytes))
            .or_default()
            .push((token_start, token_start + token_count));
    } else {
        let end = output_offset
            .checked_add(decoded_len)
            .ok_or_else(|| anyhow!("record output range overflow"))?;
        coverage.push((output_offset, end));
    }
    Ok(())
}

fn validate_owned_record_coverage(records: &[OwnedRecord], raw_len: usize) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::f32_to_f16_bits;

    fn f16_bytes(values: usize) -> Vec<u8> {
        (0..values)
            .flat_map(|index| f32_to_f16_bits(index as f32 / 17.0).to_le_bytes())
            .collect()
    }

    fn layout(token_count: u64, v_transposed: bool) -> PageLayout {
        let payload_bytes = token_count * 2 * 2 * 6;
        PageLayout {
            payload_bytes,
            components: vec![ComponentLayout {
                token_count,
                layer_count: 2,
                k_row_bytes: 6,
                v_row_bytes: if v_transposed { 0 } else { 6 },
                v_element_bytes: if v_transposed { 2 } else { 0 },
                k_idx_row_bytes: 0,
                payload_offset: 0,
                payload_bytes,
                v_transposed,
            }],
        }
    }

    #[test]
    fn page_roundtrip_preserves_geometry_and_length() {
        let layout = layout(4, false);
        let raw = f16_bytes(48);
        let archive = encode_f16_page(&layout, &raw).expect("encode");
        let decoded = decode_f16_page(&archive.bytes, raw.len()).expect("decode");
        assert_eq!(decoded.len(), raw.len());
        assert_eq!(archive.tile_count, 4);
        assert_ne!(decoded, raw, "fixture must exercise lossy quantization");
    }

    #[test]
    fn transposed_v_layout_uses_inferred_row_width() {
        let layout = layout(4, true);
        let raw = f16_bytes(48);
        let archive = encode_f16_page(&layout, &raw).expect("encode");
        let decoded = decode_f16_page(&archive.bytes, raw.len()).expect("decode");
        assert_eq!(decoded.len(), raw.len());
        assert_eq!(archive.tile_count, 4);
    }

    #[test]
    fn validation_rejects_a_corrupt_nested_segment_before_decode() {
        let layout = layout(4, false);
        let raw = f16_bytes(48);
        let mut archive = encode_f16_page(&layout, &raw).expect("encode").bytes;
        archive[ARCHIVE_HEADER_BYTES + RECORD_HEADER_BYTES] = b'X';
        assert!(validate_archive(&archive, raw.len()).is_err());
    }

    #[test]
    fn long_transposed_pages_are_chunked_at_the_reference_limit() {
        let token_count = MAX_TOKENS_PER_CHUNK as u64 + 4;
        let layout = layout(token_count, true);
        let raw = f16_bytes(layout.payload_bytes as usize / 2);
        let archive = encode_f16_page(&layout, &raw).expect("encode");
        let decoded = decode_f16_page(&archive.bytes, raw.len()).expect("decode");
        assert_eq!(decoded.len(), raw.len());
        assert_eq!(archive.tile_count, 8);
    }

    #[test]
    fn archive_rejects_trailing_and_uncovered_bytes() {
        let layout = layout(4, false);
        let raw = f16_bytes(48);
        let mut archive = encode_f16_page(&layout, &raw).expect("encode").bytes;
        archive.push(0);
        assert!(validate_archive(&archive, raw.len()).is_err());

        let mut uncovered = layout;
        uncovered.payload_bytes += 2;
        assert!(encode_f16_page(&uncovered, &f16_bytes(49)).is_err());
    }
}
