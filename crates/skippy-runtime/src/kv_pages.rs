use std::ptr;

use anyhow::{Result, bail, ensure};
use skippy_cache::cachegen::archive::{RecordKind, ValidatedArchive, validate_archive};
use skippy_ffi::{CacheGenRecordV1, KvPageDesc as RawKvPageDesc};

use crate::error::{ensure_ok, free_error};
use crate::session::StageSession;

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
                    RecordKind::CacheGenF32
                    | RecordKind::CacheGenF32Transposed
                    | RecordKind::CacheGenQ8_0
                    | RecordKind::CacheGenQ4_0 => {
                        bail!("resident KV backend does not provide this typed CacheGen adapter")
                    }
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

impl StageSession {
    pub fn export_state(&mut self, layer_start: i32, layer_end: i32) -> Result<Vec<u8>> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_state(
                self.raw,
                layer_start,
                layer_end,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
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
            skippy_ffi::skippy_export_state(
                self.raw,
                layer_start,
                layer_end,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(payload)
    }

    pub fn import_state(&mut self, layer_start: i32, layer_end: i32, input: &[u8]) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_import_state(
                self.raw,
                layer_start,
                layer_end,
                input.as_ptr().cast(),
                input.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)
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
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_full_state(
                self.raw,
                layer_start,
                layer_end,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            )
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
            skippy_ffi::skippy_import_full_state(
                self.raw,
                layer_start,
                layer_end,
                input.as_ptr().cast(),
                input.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)
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
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_kv_page(
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
            )
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
            skippy_ffi::skippy_export_kv_page(
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
            )
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
            skippy_ffi::skippy_import_kv_page(
                self.raw,
                &raw,
                payload.as_ptr().cast(),
                payload.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .max(desc.token_start.saturating_add(desc.token_count));
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
            skippy_ffi::skippy_import_cachegen_kv_page_v1(
                self.raw,
                &raw,
                records.as_ptr(),
                records.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .max(desc.token_start.saturating_add(desc.token_count));
        Ok(())
    }

    pub fn export_recurrent_state(&mut self) -> Result<Vec<u8>> {
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
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_recurrent_state(
                self.raw,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(payload)
    }

    pub fn import_recurrent_state(&mut self, input: &[u8]) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_import_recurrent_state(
                self.raw,
                input.as_ptr().cast(),
                input.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)
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

    use super::{cachegen_records, validate_imported_full_state_position};

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
    fn typed_records_fail_before_entering_an_unsupported_native_backend() {
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

        assert!(cachegen_records(&validated).is_err());
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
