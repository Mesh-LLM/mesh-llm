//! Export cache-owned prefixes only at explicit cold/offload boundaries.
use crate::error::{ensure_ok, free_error};
use crate::{RuntimeKvPage, RuntimeKvPageDesc, Status, session::StageSession};
use anyhow::{Result, ensure};
use std::ptr;

impl StageSession {
    pub fn resident_kv_page_desc(
        &mut self,
        sequence: i32,
        layers: (i32, i32),
        tokens: u64,
    ) -> Result<RuntimeKvPageDesc> {
        let mut desc = skippy_ffi::KvPageDesc::default();
        let mut bytes = 0;
        let mut error = ptr::null_mut();
        // SAFETY: The live session and all output pointers remain valid for the call.
        let status = unsafe {
            skippy_ffi::skippy_session_export_resident_kv_page(
                self.raw,
                sequence,
                layers.0,
                layers.1,
                0,
                tokens,
                &mut desc,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        check_export_size(status, error)?;
        ensure!(
            desc.payload_bytes == bytes as u64,
            "resident KV export size mismatch"
        );
        Ok(desc.into())
    }

    pub fn export_resident_kv_page(
        &mut self,
        sequence: i32,
        layers: (i32, i32),
        tokens: u64,
        bytes: usize,
    ) -> Result<RuntimeKvPage> {
        let mut desc = skippy_ffi::KvPageDesc::default();
        let mut payload = vec![0; bytes];
        let mut written = 0;
        let mut error = ptr::null_mut();
        // SAFETY: The session is live and the output allocation has the advertised capacity.
        let status = unsafe {
            skippy_ffi::skippy_session_export_resident_kv_page(
                self.raw,
                sequence,
                layers.0,
                layers.1,
                0,
                tokens,
                &mut desc,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        ensure!(
            written <= payload.len(),
            "resident KV export exceeded reserved bytes"
        );
        payload.truncate(written);
        Ok(RuntimeKvPage {
            desc: desc.into(),
            payload,
        })
    }

    pub fn resident_recurrent_state_bytes(&mut self, sequence: i32) -> Result<usize> {
        let mut bytes = 0;
        let mut error = ptr::null_mut();
        // SAFETY: The live session and byte-count/error outputs are valid.
        let status = unsafe {
            skippy_ffi::skippy_session_export_resident_recurrent_state(
                self.raw,
                sequence,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        check_export_size(status, error)?;
        Ok(bytes)
    }

    pub fn export_resident_recurrent_state(
        &mut self,
        sequence: i32,
        bytes: usize,
    ) -> Result<Vec<u8>> {
        let mut payload = vec![0; bytes];
        let mut written = 0;
        let mut error = ptr::null_mut();
        // SAFETY: The session is live and output storage matches the reserved capacity.
        let status = unsafe {
            skippy_ffi::skippy_session_export_resident_recurrent_state(
                self.raw,
                sequence,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        ensure!(
            written <= payload.len(),
            "resident recurrent export exceeded reserved bytes"
        );
        payload.truncate(written);
        Ok(payload)
    }
}

fn check_export_size(status: Status, error: *mut skippy_ffi::Error) -> Result<()> {
    if status == Status::BufferTooSmall || status == Status::Ok {
        free_error(error);
        Ok(())
    } else {
        ensure_ok(status, error)
    }
}
