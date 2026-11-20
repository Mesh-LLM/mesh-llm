use crate::native_timing::native_call;
use std::ptr;

use anyhow::{Result, bail, ensure};
use skippy_cache::cachegen::archive::{
    CacheGenArchive, ComponentLayout, PageLayout, RecordKind, ValidatedArchive, ValueType,
    decode_page, encode_page, validate_archive,
};
use skippy_ffi::{CacheGenRecordV1, KvPageDesc as RawKvPageDesc};

use crate::error::{ensure_ok, free_error};
use crate::session::StageSession;
use crate::{
    CheckpointShape, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
    KV_PAGE_FLAG_V_TRANSPOSED,
};

/// One stage checkpoint, built to the stage's [`CheckpointShape`]: a hybrid
/// checkpoint cannot be built without both its KV page and its recurrent state.
#[derive(Debug, Clone, Copy)]
pub enum CheckpointParts<'a> {
    Attention {
        desc: &'a RuntimeKvPageDesc,
        page: &'a [u8],
    },
    Recurrent {
        state: &'a [u8],
    },
    AttentionAndRecurrent {
        desc: &'a RuntimeKvPageDesc,
        page: &'a [u8],
        state: &'a [u8],
    },
}

impl<'a> CheckpointParts<'a> {
    /// Fit exported components to a shape. A part the shape needs and the
    /// export lacks, or one it does not need, is an error, not a quieter restore.
    pub fn fit(
        shape: CheckpointShape,
        page: Option<(&'a RuntimeKvPageDesc, &'a [u8])>,
        state: &'a [u8],
    ) -> Result<Self> {
        Ok(match (shape, page, state.is_empty()) {
            (CheckpointShape::Attention, Some((desc, page)), true) => {
                Self::Attention { desc, page }
            }
            (CheckpointShape::Recurrent, None, false) => Self::Recurrent { state },
            (CheckpointShape::AttentionAndRecurrent, Some((desc, page)), false) => {
                Self::AttentionAndRecurrent { desc, page, state }
            }
            (shape, page, no_state) => bail!(
                "a {shape:?} checkpoint cannot be built from {} and {}",
                if page.is_some() {
                    "a KV page"
                } else {
                    "no KV page"
                },
                if no_state {
                    "no recurrent state"
                } else {
                    "recurrent state"
                },
            ),
        })
    }

    fn page(&self) -> Option<(&'a RuntimeKvPageDesc, &'a [u8])> {
        match *self {
            Self::Attention { desc, page } | Self::AttentionAndRecurrent { desc, page, .. } => {
                Some((desc, page))
            }
            Self::Recurrent { .. } => None,
        }
    }

    fn state(&self) -> &'a [u8] {
        match *self {
            Self::Recurrent { state } | Self::AttentionAndRecurrent { state, .. } => state,
            Self::Attention { .. } => &[],
        }
    }
}

/// Encode one complete runtime KV page into the portable CacheGen archive
/// consumed directly by the native device import ABI.
pub fn encode_cachegen_kv_page(desc: &RuntimeKvPageDesc, raw: &[u8]) -> Result<CacheGenArchive> {
    desc.validate_payload(raw.len())?;
    encode_page(&cachegen_page_layout(desc)?, raw)
}

/// Scalar oracle used by correctness tooling. Serving restores call the
/// native device importer and never materialize this decoded allocation.
pub fn decode_cachegen_kv_page(desc: &RuntimeKvPageDesc, archive: &[u8]) -> Result<Vec<u8>> {
    let raw_len = usize::try_from(desc.payload_bytes)?;
    desc.validate_payload(raw_len)?;
    decode_page(archive, raw_len)
}

fn cachegen_page_layout(desc: &RuntimeKvPageDesc) -> Result<PageLayout> {
    let components = if desc.component_count == 0 {
        vec![cachegen_component_layout(
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
                cachegen_component_layout(
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
fn cachegen_component_layout(
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
    Ok(ComponentLayout {
        token_count,
        layer_count,
        k_type: cachegen_value_type(k_type)?,
        v_type: cachegen_value_type(v_type)?,
        k_row_bytes,
        v_row_bytes,
        v_element_bytes,
        k_idx_row_bytes,
        payload_offset,
        payload_bytes,
        v_transposed: flags & KV_PAGE_FLAG_V_TRANSPOSED != 0,
    })
}

fn cachegen_value_type(value: u32) -> Result<ValueType> {
    match value {
        GGML_TYPE_F32 => Ok(ValueType::F32),
        GGML_TYPE_F16 => Ok(ValueType::F16),
        GGML_TYPE_Q8_0 => Ok(ValueType::Q8_0),
        GGML_TYPE_Q4_0 => Ok(ValueType::Q4_0),
        _ => anyhow::bail!("CacheGen does not support runtime K/V type {value}"),
    }
}

fn cachegen_records(validated: &ValidatedArchive<'_>) -> Result<Vec<CacheGenRecordV1>> {
    validated
        .records
        .iter()
        .map(|record| {
            Ok(CacheGenRecordV1 {
                abi_version: skippy_ffi::CACHEGEN_RECORD_V1_ABI_VERSION,
                kind: match record.kind {
                    RecordKind::CacheGen => skippy_ffi::CACHEGEN_RECORD_F16,
                    RecordKind::Exact => skippy_ffi::CACHEGEN_RECORD_EXACT,
                    RecordKind::CacheGenTransposed => skippy_ffi::CACHEGEN_RECORD_F16_TRANSPOSED,
                    RecordKind::CacheGenF32 => skippy_ffi::CACHEGEN_RECORD_F32,
                    RecordKind::CacheGenF32Transposed => skippy_ffi::CACHEGEN_RECORD_F32_TRANSPOSED,
                    RecordKind::CacheGenQ8_0 => skippy_ffi::CACHEGEN_RECORD_Q8_0,
                    RecordKind::CacheGenQ4_0 => skippy_ffi::CACHEGEN_RECORD_Q4_0,
                },
                element_bytes: record.element_bytes as u32,
                reserved0: 0,
                output_offset: record.output_offset as u64,
                decoded_bytes: record.decoded_len as u64,
                token_count: record.token_count as u64,
                token_start: record.token_start as u64,
                total_tokens: record.total_tokens as u64,
                payload: record.payload.as_ptr().cast(),
                payload_bytes: record.payload.len(),
            })
        })
        .collect()
}
use crate::{RuntimeKvPage, RuntimeKvPageDesc, Status};

/// A size query passes no output buffer, so the native call reports the
/// required byte count with either `Ok` or `BufferTooSmall`.
fn size_query_result(status: Status, error: *mut skippy_ffi::Error, bytes: usize) -> Result<u64> {
    if status != Status::BufferTooSmall && status != Status::Ok {
        ensure_ok(status, error)?;
    } else {
        free_error(error);
    }
    Ok(bytes as u64)
}

impl StageSession {
    /// Bytes [`Self::export_full_state`] would allocate, without copying state.
    pub fn full_state_size(&mut self, layer_start: i32, layer_end: i32) -> Result<u64> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_full_state(
                self.raw,
                layer_start,
                layer_end,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        size_query_result(status, error, bytes)
    }

    /// Bytes [`Self::export_recurrent_state`] would allocate, without copying state.
    pub fn recurrent_state_size(&mut self) -> Result<u64> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_recurrent_state(
                self.raw,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        size_query_result(status, error, bytes)
    }

    /// Bytes [`Self::export_kv_page`] would allocate, without copying state.
    pub fn kv_page_size(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        token_start: u64,
        token_count: u64,
    ) -> Result<u64> {
        self.kv_page_size_query(layer_start, layer_end, token_start, token_count)
            .map(|(bytes, _)| bytes)
    }

    /// The layout [`Self::export_kv_page`] would describe, without copying
    /// state. A memory with a sliding-window cache reports
    /// [`skippy_ffi::KV_PAGE_CODEC_ISWA_COMPOSITE_V1`].
    pub fn kv_page_layout(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        token_start: u64,
        token_count: u64,
    ) -> Result<RuntimeKvPageDesc> {
        self.kv_page_size_query(layer_start, layer_end, token_start, token_count)
            .map(|(_, desc)| desc.into())
    }

    fn kv_page_size_query(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        token_start: u64,
        token_count: u64,
    ) -> Result<(u64, RawKvPageDesc)> {
        let mut desc = RawKvPageDesc::default();
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_kv_page(
                self.raw,
                layer_start,
                layer_end,
                token_start,
                token_count,
                &mut desc,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        size_query_result(status, error, bytes).map(|bytes| (bytes, desc))
    }

    pub fn export_state(&mut self, layer_start: i32, layer_end: i32) -> Result<Vec<u8>> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_export_state(
                self.raw,
                layer_start,
                layer_end,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            ))
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_export_state(
                self.raw,
                layer_start,
                layer_end,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(payload)
    }

    pub fn import_state(&mut self, layer_start: i32, layer_end: i32, input: &[u8]) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_import_state(
                self.raw,
                layer_start,
                layer_end,
                input.as_ptr().cast(),
                input.len(),
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        // Imported target state has no matching draft features.
        self.dflash_after_replace();
        Ok(())
    }

    pub fn import_state_for_token_count(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        input: &[u8],
        token_count: u64,
    ) -> Result<()> {
        self.import_state(layer_start, layer_end, input)?;
        self.set_position(token_count)
    }

    pub fn export_full_state(&mut self, layer_start: i32, layer_end: i32) -> Result<Vec<u8>> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_export_full_state(
                self.raw,
                layer_start,
                layer_end,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            ))
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_export_full_state(
                self.raw,
                layer_start,
                layer_end,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(payload)
    }

    pub fn import_full_state(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        input: &[u8],
    ) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_import_full_state(
                self.raw,
                layer_start,
                layer_end,
                input.as_ptr().cast(),
                input.len(),
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        // Imported target state has no matching draft features.
        self.dflash_after_replace();
        Ok(())
    }

    pub fn import_full_state_for_token_count(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        input: &[u8],
        token_count: u64,
    ) -> Result<()> {
        self.import_full_state(layer_start, layer_end, input)?;
        let native_position = self.native_position()?;
        self.token_count = native_position;
        validate_imported_full_state_position(token_count, native_position)?;
        Ok(())
    }

    pub fn export_kv_page(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        token_start: u64,
        token_count: u64,
    ) -> Result<RuntimeKvPage> {
        let mut desc = RawKvPageDesc::default();
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_export_kv_page(
                self.raw,
                layer_start,
                layer_end,
                token_start,
                token_count,
                &mut desc,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            ))
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_export_kv_page(
                self.raw,
                layer_start,
                layer_end,
                token_start,
                token_count,
                &mut desc,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(RuntimeKvPage {
            desc: desc.into(),
            payload,
        })
    }

    pub fn export_kv_page_into(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        token_start: u64,
        token_count: u64,
        output: &mut [u8],
    ) -> Result<RuntimeKvPageDesc> {
        let mut desc = RawKvPageDesc::default();
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_export_kv_page(
                self.raw,
                layer_start,
                layer_end,
                token_start,
                token_count,
                &mut desc,
                output.as_mut_ptr().cast(),
                output.len(),
                &mut written,
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        if written != output.len() {
            anyhow::bail!(
                "KV page export wrote {written} bytes into {} byte output buffer",
                output.len()
            );
        }
        Ok(desc.into())
    }

    pub fn import_kv_page(&mut self, desc: &RuntimeKvPageDesc, payload: &[u8]) -> Result<()> {
        desc.validate_payload(payload.len())?;
        if desc.codec == skippy_ffi::KV_PAGE_CODEC_ISWA_COMPOSITE_V1 && self.token_count != 0 {
            anyhow::bail!("composite ISWA KV page import requires a fresh session");
        }
        let raw = desc.as_raw();
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_import_kv_page(
                self.raw,
                &raw,
                payload.as_ptr().cast(),
                payload.len(),
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .max(desc.token_start.saturating_add(desc.token_count));
        // Imported target state has no matching draft features.
        self.dflash_after_replace();
        Ok(())
    }

    pub fn import_cachegen_kv_page(
        &mut self,
        desc: &RuntimeKvPageDesc,
        archive: &[u8],
    ) -> Result<()> {
        let raw_len = usize::try_from(desc.payload_bytes)?;
        desc.validate_payload(raw_len)?;
        if desc.codec == skippy_ffi::KV_PAGE_CODEC_ISWA_COMPOSITE_V1 && self.token_count != 0 {
            anyhow::bail!("composite ISWA CacheGen page import requires a fresh session");
        }
        let validated = validate_archive(archive, raw_len)?;
        let records = cachegen_records(&validated)?;
        let raw = desc.as_raw();
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_import_cachegen_kv_page_v1(
                self.raw,
                &raw,
                records.as_ptr(),
                records.len(),
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .max(desc.token_start.saturating_add(desc.token_count));
        // Imported target state has no matching draft features.
        self.dflash_after_replace();
        Ok(())
    }

    pub fn export_recurrent_state(&mut self) -> Result<Vec<u8>> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_export_recurrent_state(
                self.raw,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            ))
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_export_recurrent_state(
                self.raw,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(payload)
    }

    pub fn import_recurrent_state(&mut self, input: &[u8]) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_import_recurrent_state(
                self.raw,
                input.as_ptr().cast(),
                input.len(),
                &mut error,
            ))
        };
        ensure_ok(status, error)?;
        // Imported target state has no matching draft features.
        self.dflash_after_replace();
        Ok(())
    }

    /// Import a whole checkpoint into this fresh session. The native side
    /// checks the parts against the stage again, imports the page, the
    /// recurrent state, the position and the token history in that order, and
    /// leaves the session empty if any of them fails.
    pub fn import_checkpoint(
        &mut self,
        parts: &CheckpointParts<'_>,
        token_count: u64,
        tokens: &[i32],
    ) -> Result<()> {
        ensure!(
            tokens.is_empty() || tokens.len() as u64 == token_count,
            "checkpoint token history has {} tokens for position {token_count}",
            tokens.len()
        );
        let page = parts.page();
        if let Some((desc, payload)) = page {
            desc.validate_payload(payload.len())?;
        }
        let raw_desc = page.map(|(desc, _)| desc.as_raw());
        let state = parts.state();
        let view = skippy_ffi::CheckpointView {
            page: raw_desc.as_ref().map_or(ptr::null(), ptr::from_ref),
            page_payload: page.map_or(ptr::null(), |(_, payload)| payload.as_ptr().cast()),
            page_payload_bytes: page.map_or(0, |(_, payload)| payload.len()),
            recurrent: if state.is_empty() {
                ptr::null()
            } else {
                state.as_ptr().cast()
            },
            recurrent_bytes: state.len(),
            token_count,
            tokens: if tokens.is_empty() {
                ptr::null()
            } else {
                tokens.as_ptr()
            },
        };
        let mut error = ptr::null_mut();
        // SAFETY: every pointer in the view borrows from `parts`, `raw_desc`
        // or `tokens`, all of which outlive the call; native code copies them.
        let status = unsafe { native_call!(skippy_import_checkpoint(self.raw, &view, &mut error)) };
        if let Err(error) = ensure_ok(status, error) {
            self.token_count = 0;
            return Err(error);
        }
        self.token_count = token_count;
        // Imported target state has no matching draft features.
        self.dflash_after_replace();
        Ok(())
    }

    pub fn import_recurrent_state_for_token_count(
        &mut self,
        input: &[u8],
        token_count: u64,
    ) -> Result<()> {
        self.import_recurrent_state(input)?;
        self.set_position(token_count)
    }
}

fn validate_imported_full_state_position(expected: u64, actual: u64) -> Result<()> {
    ensure!(
        actual == expected,
        "full-state import restored native position {actual}, expected {expected}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use skippy_cache::cachegen::archive::{Record, RecordKind, ValidatedArchive};

    use super::{CheckpointParts, cachegen_records, validate_imported_full_state_position};
    use crate::{CheckpointShape, RuntimeKvPageDesc};

    #[test]
    fn a_checkpoint_is_built_only_from_exactly_its_shapes_parts() {
        let desc = RuntimeKvPageDesc {
            version: 1,
            layer_start: 0,
            layer_end: 1,
            token_start: 0,
            token_count: 1,
            layer_count: 1,
            k_type: crate::GGML_TYPE_F16,
            v_type: crate::GGML_TYPE_F16,
            k_row_bytes: 1,
            v_row_bytes: 1,
            v_element_bytes: 1,
            k_idx_row_bytes: 0,
            payload_bytes: 1,
            flags: 0,
            codec: 0,
            component_count: 0,
            components: Box::new([Default::default(); 2]),
        };
        let page = Some((&desc, &[1_u8][..]));
        let state = [2_u8];
        let hybrid = CheckpointShape::AttentionAndRecurrent;
        assert!(CheckpointParts::fit(hybrid, page, &state).is_ok());
        assert!(CheckpointParts::fit(hybrid, None, &state).is_err());
        assert!(CheckpointParts::fit(hybrid, page, &[]).is_err());
        assert!(CheckpointParts::fit(CheckpointShape::Recurrent, None, &state).is_ok());
        assert!(CheckpointParts::fit(CheckpointShape::Recurrent, page, &state).is_err());
        assert!(CheckpointParts::fit(CheckpointShape::Attention, page, &[]).is_ok());
        assert!(CheckpointParts::fit(CheckpointShape::Attention, page, &state).is_err());
    }

    #[test]
    fn maps_validated_cachegen_records_to_the_native_abi() {
        let payload = [1_u8, 2, 3, 4];
        let validated = ValidatedArchive {
            raw_len: 16,
            records: vec![Record {
                kind: RecordKind::CacheGenTransposed,
                element_bytes: 2,
                output_offset: 8,
                decoded_len: 8,
                token_count: 2,
                token_start: 3,
                total_tokens: 5,
                payload: &payload,
            }],
        };

        let records = cachegen_records(&validated).expect("F16 records map to the native ABI");
        assert_eq!(records.len(), 1);
        let record = records[0];
        assert_eq!(record.abi_version, 1);
        assert_eq!(record.kind, skippy_ffi::CACHEGEN_RECORD_F16_TRANSPOSED);
        assert_eq!(record.element_bytes, 2);
        assert_eq!(record.reserved0, 0);
        assert_eq!(record.output_offset, 8);
        assert_eq!(record.decoded_bytes, 8);
        assert_eq!(record.token_count, 2);
        assert_eq!(record.token_start, 3);
        assert_eq!(record.total_tokens, 5);
        assert_eq!(record.payload, payload.as_ptr().cast());
        assert_eq!(record.payload_bytes, payload.len());
    }

    #[test]
    fn maps_quantized_records_to_the_native_abi() {
        let payload = [1_u8, 2, 3, 4];
        let validated = ValidatedArchive {
            raw_len: 16,
            records: vec![Record {
                kind: RecordKind::CacheGenQ8_0,
                element_bytes: 34,
                output_offset: 0,
                decoded_len: 16,
                token_count: 1,
                token_start: 0,
                total_tokens: 0,
                payload: &payload,
            }],
        };

        let records = cachegen_records(&validated).expect("Q8_0 records map to the native ABI");
        assert_eq!(records[0].kind, skippy_ffi::CACHEGEN_RECORD_Q8_0);
        assert_eq!(records[0].element_bytes, 34);

        let validated = ValidatedArchive {
            raw_len: 16,
            records: vec![Record {
                kind: RecordKind::CacheGenQ4_0,
                element_bytes: 18,
                output_offset: 0,
                decoded_len: 16,
                token_count: 1,
                token_start: 0,
                total_tokens: 0,
                payload: &payload,
            }],
        };
        let records = cachegen_records(&validated).expect("Q4_0 records map to the native ABI");
        assert_eq!(records[0].kind, skippy_ffi::CACHEGEN_RECORD_Q4_0);
        assert_eq!(records[0].element_bytes, 18);
    }

    #[test]
    fn maps_f32_records_to_the_native_abi() {
        let payload = [1_u8, 2, 3, 4];
        let validated = ValidatedArchive {
            raw_len: 16,
            records: vec![Record {
                kind: RecordKind::CacheGenF32Transposed,
                element_bytes: 4,
                output_offset: 8,
                decoded_len: 16,
                token_count: 2,
                token_start: 3,
                total_tokens: 5,
                payload: &payload,
            }],
        };

        let records = cachegen_records(&validated).expect("F32 records map to the native ABI");
        assert_eq!(records[0].kind, skippy_ffi::CACHEGEN_RECORD_F32_TRANSPOSED);
        assert_eq!(records[0].element_bytes, 4);
    }

    #[test]
    fn full_state_import_accepts_the_position_carried_by_native_state() {
        validate_imported_full_state_position(17, 17).unwrap();
    }

    #[test]
    fn full_state_import_rejects_lost_native_position() {
        let error = validate_imported_full_state_position(17, 0).unwrap_err();
        assert_eq!(
            error.to_string(),
            "full-state import restored native position 0, expected 17"
        );
    }
}
