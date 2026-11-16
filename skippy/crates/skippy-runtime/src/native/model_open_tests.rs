use super::{RuntimeConfig, StageModel};
use anyhow::{Context, Result, ensure};
use skippy_ffi::{SkippyRuntimeEventReporterV1, SkippyRuntimeEventV1};
use std::sync::atomic::{AtomicUsize, Ordering};

unsafe extern "C" {
    fn skippy_set_runtime_event_reporter(
        reporter: *const SkippyRuntimeEventReporterV1,
    ) -> skippy_ffi::Status;
    fn skippy_clear_runtime_event_reporter();
}

static UNLOADS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn observe_unload(event: *const SkippyRuntimeEventV1, _: *mut std::ffi::c_void) {
    // SAFETY: the native reporter lends the record until its callback returns.
    if let Some(event) = unsafe { event.as_ref() }
        && event.kind == skippy_ffi::RUNTIME_EVENT_KIND_UNLOAD_COMPLETED
        && event.model_id != 0
    {
        UNLOADS.fetch_add(1, Ordering::Relaxed);
    }
}

struct Reporter;

impl Drop for Reporter {
    fn drop(&mut self) {
        // SAFETY: clearing waits for callbacks before the test ends.
        unsafe { skippy_clear_runtime_event_reporter() };
    }
}

#[test]
#[ignore = "requires SKIPPY_CORRECTNESS_MODEL and a native runtime"]
fn rejected_projector_unloads_the_opened_model() -> Result<()> {
    crate::test_support::require_native_runtime();
    let path = std::env::var_os("SKIPPY_CORRECTNESS_MODEL")
        .context("SKIPPY_CORRECTNESS_MODEL is required")?;
    let layers = crate::ModelInfo::open(&path)?
        .tensors()?
        .iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .context("fixture has no layers")?
        + 1;
    let reporter = SkippyRuntimeEventReporterV1 {
        abi_version: crate::runtime_events::RUNTIME_EVENT_V1_ABI_VERSION,
        struct_size: std::mem::size_of::<SkippyRuntimeEventReporterV1>() as u32,
        callback: Some(observe_unload),
        user_data: std::ptr::null_mut(),
    };
    // SAFETY: the reporter layout is current and its callback has static lifetime.
    let status = unsafe { skippy_set_runtime_event_reporter(&reporter) };
    ensure!(
        status == skippy_ffi::Status::Ok,
        "native reporter unavailable"
    );
    let _reporter = Reporter;
    UNLOADS.store(0, Ordering::Relaxed);
    let config = RuntimeConfig {
        layer_end: layers,
        ctx_size: 128,
        lane_count: 1,
        selected_backend_device: Some("CPU".into()),
        // Reject after native model allocation, before loading a projector.
        projector_path: Some("invalid\0projector.gguf".into()),
        ..RuntimeConfig::default()
    };
    let error = StageModel::open(path, &config)
        .err()
        .context("invalid projector unexpectedly loaded")?;
    ensure!(error.to_string().contains("interior NUL"), "{error:#}");
    ensure!(
        UNLOADS.load(Ordering::Relaxed) == 1,
        "projector refusal did not unload its native model"
    );
    Ok(())
}
