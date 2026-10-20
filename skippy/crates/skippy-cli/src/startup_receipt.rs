//! Opt-in, bounded startup evidence; never parses decode-time diagnostics.
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

const MAX_LINE_BYTES: usize = 16 * 1024;
const MAX_KV_BUFFERS: usize = 64;

#[derive(Debug, Serialize)]
pub(crate) struct KvBuffer {
    backend_device: String,
    logged_mib: f64,
    allocated_bytes_rounded: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct Receipt {
    schema_version: u32,
    source: &'static str,
    allocation_precision: &'static str,
    kv_buffers: Vec<KvBuffer>,
    offloaded_layers: u32,
    total_layers: u32,
}

pub(crate) struct StartupCapture {
    active: AtomicBool,
    state: Mutex<CaptureState>,
}

impl StartupCapture {
    pub(crate) fn new() -> Self {
        Self {
            active: AtomicBool::new(true),
            state: Mutex::new(CaptureState::default()),
        }
    }

    pub(crate) fn record(&self, text: &str) {
        if self.active.load(Ordering::Acquire) {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if self.active.load(Ordering::Acquire)
                && let Err(error) = state.record(text)
            {
                state.error = Some(error.to_string());
            }
        }
    }

    pub(crate) fn finish(&self) -> Result<Receipt> {
        ensure!(
            self.active.swap(false, Ordering::AcqRel),
            "startup receipt already completed"
        );
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.finish()
    }
}

#[derive(Default)]
struct CaptureState {
    pending: String,
    kv_buffers: Vec<KvBuffer>,
    offload: Option<(u32, u32)>,
    error: Option<String>,
}

impl CaptureState {
    fn record(&mut self, text: &str) -> Result<()> {
        if self.error.is_some() {
            return Ok(());
        }
        for fragment in text.split_inclusive('\n') {
            ensure!(
                self.pending.len() + fragment.len() <= MAX_LINE_BYTES,
                "native startup line exceeded bounded receipt capacity"
            );
            self.pending.push_str(fragment);
            if self.pending.ends_with('\n') {
                let line = std::mem::take(&mut self.pending);
                self.line(line.trim())?;
            }
        }
        Ok(())
    }

    fn line(&mut self, line: &str) -> Result<()> {
        if let Some((prefix, value)) = line.split_once(" KV buffer size") {
            let backend_device = prefix
                .split_whitespace()
                .last()
                .context("KV device missing")?;
            ensure!(
                backend_device.len() <= 128
                    && backend_device
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_'),
                "invalid KV device label"
            );
            let logged = value
                .trim()
                .strip_prefix('=')
                .context("KV size missing")?
                .trim()
                .strip_suffix("MiB")
                .context("KV unit missing")?
                .trim();
            let logged_mib: f64 = logged.parse().context("invalid KV allocation")?;
            let bytes = logged_mib * 1_048_576.0;
            ensure!(
                logged_mib.is_finite() && logged_mib > 0.0 && bytes < u64::MAX as f64,
                "invalid KV allocation range"
            );
            ensure!(
                self.kv_buffers.len() < MAX_KV_BUFFERS,
                "too many KV buffers"
            );
            self.kv_buffers.push(KvBuffer {
                backend_device: backend_device.to_owned(),
                logged_mib,
                allocated_bytes_rounded: bytes.round() as u64,
            });
        }
        if let Some((_, value)) = line.split_once("offloaded ") {
            let counts = value
                .strip_suffix(" layers to GPU")
                .context("invalid layer offload line")?;
            let (loaded, total) = counts.split_once('/').context("missing layer total")?;
            let loaded: u32 = loaded.parse().context("invalid offloaded layers")?;
            let total: u32 = total.parse().context("invalid total layers")?;
            ensure!(total > 0 && loaded <= total, "invalid offload counts");
            ensure!(self.offload.is_none(), "multiple model offload summaries");
            self.offload = Some((loaded, total));
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Receipt> {
        ensure!(
            self.error.is_none(),
            "invalid startup receipt: {:?}",
            self.error
        );
        ensure!(self.pending.is_empty(), "partial native startup line");
        ensure!(
            !self.kv_buffers.is_empty(),
            "no native KV allocation observed"
        );
        let (offloaded_layers, total_layers) = self.offload.context("no layer offload summary")?;
        Ok(Receipt {
            schema_version: 1,
            source: "native_startup_log",
            allocation_precision: "0.01 MiB",
            kv_buffers: std::mem::take(&mut self.kv_buffers),
            offloaded_layers,
            total_layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "load_tensors: offloaded 29/29 layers to GPU\n\
        llama_kv_cache: MTL0 KV buffer size = 224.00 MiB\n";

    #[test]
    fn fragments_form_one_receipt_and_capture_stops() -> Result<()> {
        let capture = StartupCapture::new();
        for part in VALID.as_bytes().chunks(7) {
            capture.record(std::str::from_utf8(part)?);
        }
        let receipt = capture.finish()?;
        assert_eq!(receipt.offloaded_layers, 29);
        assert_eq!(receipt.total_layers, 29);
        assert_eq!(receipt.kv_buffers[0].backend_device, "MTL0");
        assert_eq!(receipt.kv_buffers[0].allocated_bytes_rounded, 234_881_024);
        capture.record("malformed decode-time text without newline");
        assert!(capture.finish().is_err());
        assert!(capture.state.lock().unwrap().pending.is_empty());
        Ok(())
    }

    #[test]
    fn missing_partial_conflicting_and_overflow_evidence_fail() {
        for text in [
            "",
            "load_tensors: offloaded 29/29 layers to GPU\n",
            "llama_kv_cache: MTL0 KV buffer size = 224.00 MiB\n",
            &format!("{VALID}partial"),
            &format!("{VALID}load_tensors: offloaded 28/29 layers to GPU\n"),
            &format!("{VALID}{}", "x".repeat(MAX_LINE_BYTES + 1)),
            &VALID.replace("224.00", "NaN"),
        ] {
            let capture = StartupCapture::new();
            capture.record(text);
            assert!(capture.finish().is_err(), "{text}");
        }
    }

    #[test]
    fn receipt_preserves_cpu_and_partial_offload_for_consumer_rejection() -> Result<()> {
        let capture = StartupCapture::new();
        capture.record(&VALID.replace("29/29", "28/29").replace("MTL0", "CPU"));
        let receipt = capture.finish()?;
        assert_eq!(receipt.offloaded_layers, 28);
        assert_eq!(receipt.kv_buffers[0].backend_device, "CPU");
        Ok(())
    }

    #[test]
    fn fragmented_size_field_and_buffer_bound_are_checked() -> Result<()> {
        let capture = StartupCapture::new();
        capture.record("load_tensors: offloaded 29/29 layers to GPU\n");
        capture.record("llama_kv_cache: MTL0 KV buffer size = 22");
        capture.record("4.");
        capture.record("00 Mi");
        capture.record("B\n");
        assert_eq!(
            capture.finish()?.kv_buffers[0].allocated_bytes_rounded,
            234_881_024
        );
        for count in [MAX_KV_BUFFERS, MAX_KV_BUFFERS + 1] {
            let capture = StartupCapture::new();
            capture.record("load_tensors: offloaded 29/29 layers to GPU\n");
            for _ in 0..count {
                capture.record("llama_kv_cache: MTL0 KV buffer size = 1.00 MiB\n");
            }
            assert_eq!(capture.finish().is_ok(), count == MAX_KV_BUFFERS);
        }
        Ok(())
    }
}
