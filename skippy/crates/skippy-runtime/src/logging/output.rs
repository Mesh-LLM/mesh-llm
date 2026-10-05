//! Native callback destinations selected by the embedding product.
use std::{
    fs::{File, OpenOptions},
    io::{LineWriter, Write},
    path::Path,
    ptr,
    sync::{Arc, Mutex, OnceLock, RwLock},
};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{Context, Result, anyhow};

static NATIVE_LOG_FILE: OnceLock<Mutex<Option<LineWriter<File>>>> = OnceLock::new();
static NATIVE_LOG_SINK: RwLock<Option<Arc<dyn NativeLogSink>>> = RwLock::new(None);

/// Optional product-owned destination for raw llama.cpp, ggml and mtmd logs.
/// Called synchronously from native threads with ggml_log_level values.
/// Implementations must not panic or reenter native inference.
pub trait NativeLogSink: Send + Sync {
    fn write(&self, level: i32, text: &str);
}

/// Install after loading native libraries and before device/model initialization.
/// Raw output remains separate from file capture and parsed log events.
pub fn set_native_log_sink(sink: Arc<dyn NativeLogSink>) {
    *NATIVE_LOG_SINK
        .write()
        .unwrap_or_else(|error| error.into_inner()) = Some(sink);
    set_native_log_callback(Some(super::write_native_log));
}

pub(super) fn forward_raw_native_log(level: i32, bytes: &[u8]) {
    let sink = NATIVE_LOG_SINK
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .clone();
    if let Some(sink) = sink {
        sink.write(level, &String::from_utf8_lossy(bytes));
    }
}

pub(super) fn native_log_file() -> &'static Mutex<Option<LineWriter<File>>> {
    NATIVE_LOG_FILE.get_or_init(|| Mutex::new(None))
}

pub(super) fn flush_native_log_writer<W: Write>(writer: &mut Option<LineWriter<W>>) {
    if let Some(writer) = writer.as_mut() {
        let _ = writer.flush();
    }
}

fn clear_native_log_output() {
    *NATIVE_LOG_SINK
        .write()
        .unwrap_or_else(|error| error.into_inner()) = None;
    if let Ok(mut guard) = native_log_file().lock() {
        flush_native_log_writer(&mut guard);
        *guard = None;
    }
}

fn set_native_log_callback(callback: skippy_ffi::LlamaLogCallback) {
    if !skippy_ffi::native_runtime_loaded() {
        return;
    }
    // SAFETY: The runtime is loaded, callbacks have static lifetime and carry
    // no borrowed user data. Products configure these before native work starts.
    unsafe {
        skippy_ffi::llama_log_set(callback, ptr::null_mut());
        skippy_ffi::ggml_log_set(callback, ptr::null_mut());
        skippy_ffi::mtmd_helper_log_set(callback, ptr::null_mut());
    }
}

pub fn redirect_native_logs_to_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    options.mode(0o600);

    let file = options
        .open(path)
        .with_context(|| format!("open skippy native log file {}", path.display()))?;
    let mut guard = native_log_file()
        .lock()
        .map_err(|_| anyhow!("native log file mutex poisoned"))?;
    flush_native_log_writer(&mut guard);
    *guard = Some(LineWriter::new(file));
    drop(guard);
    set_native_log_callback(Some(super::write_native_log));
    Ok(())
}

pub fn suppress_native_logs() {
    clear_native_log_output();
    set_native_log_callback(Some(super::discard_native_log));
}

pub fn restore_native_logs() {
    clear_native_log_output();
    set_native_log_callback(None);
}

#[cfg(test)]
#[path = "tests/native_log_file.rs"]
mod tests;
