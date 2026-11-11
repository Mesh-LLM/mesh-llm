//! Inference receipts as the embedded Skippy runtime knows them
//! (`mesh/docs/design/RECEIPTS.md`): whether a loaded model issues receipts
//! a client can audit, and what a later stage of a split needs to record
//! its part of them.

use skippy_serving::receipts::ReceiptsOptions;

use super::SkippyModelHandle;
use super::stage::StageLoadRequest;

impl SkippyModelHandle {
    /// Whether the loaded runtime issues inference receipts a client can
    /// audit: its recorder started, whatever the settings asked for, and
    /// its backend has rounding profiles
    /// ([`skippy_receipts::audit::BackendFamily`]): CPU, CUDA and Metal.
    /// Receipts from any other backend (Vulkan, ROCm) are ones every audit
    /// skips, so it must not satisfy the receipts trust tier.
    ///
    /// The recorder does not start on an unqualified backend or with a
    /// quantized KV cache. Metal also requires a measured device, full
    /// single-device placement, and Q8_0 matrices in the actual loaded
    /// artifacts (`skippy_serving::receipts::start`). A refused runtime
    /// names no receipts backend, so this is false without duplicating
    /// admission policy in Mesh. The backend is checked here again for
    /// a runtime that predates that rule.
    pub(crate) fn issues_receipts(&self) -> bool {
        self.runtime
            .receipts_backend()
            .is_some_and(|backend| receipts_backend_is_audited(&backend))
    }
}

/// Whether receipts that name `backend` can be audited: the verifier has
/// rounding profiles for its family.
fn receipts_backend_is_audited(backend: &str) -> bool {
    skippy_receipts::audit::BackendFamily::of(backend).is_some()
}

/// What a later stage of a split needs to record its part of the receipts
/// stage 0 issues, when the coordinator asks for them. Stage 0 reaches it
/// over a receipts connection to this stage's binary address, so it serves
/// no receipts API, and it signs nothing: stage 0 signs each turn root.
pub(super) fn later_stage_options(load: &StageLoadRequest) -> Option<ReceiptsOptions> {
    load.issue_receipts
        .then(|| ReceiptsOptions::embedded(load.model_id.clone(), None))
}

#[cfg(test)]
mod tests {
    use super::receipts_backend_is_audited;

    #[test]
    #[ignore = "requires qualified Metal, local Q8/Q6 GGUFs and a native runtime; run alone"]
    fn metal_admission_controls_the_loaded_host_trust_flag() -> anyhow::Result<()> {
        use super::super::{SkippyDeviceDescriptor, SkippyModelHandle, SkippyModelLoadOptions};

        #[cfg(feature = "dynamic-native-runtime")]
        crate::system::native_runtime::load_local_native_runtime_for_embedded_serving(
            &crate::system::native_runtime::NativeRuntimeStartupSelection::current(),
        )?;
        let q8 = std::env::var_os("SKIPPY_METAL_CONTROL_Q8")
            .ok_or_else(|| anyhow::anyhow!("SKIPPY_METAL_CONTROL_Q8 is required"))?;
        let q6 = std::env::var_os("SKIPPY_METAL_CONTROL_Q6")
            .ok_or_else(|| anyhow::anyhow!("SKIPPY_METAL_CONTROL_Q6 is required"))?;
        for (path, gpu_layers, expected) in [(&q8, -1, true), (&q6, -1, false), (&q8, 0, false)] {
            let mut options = SkippyModelLoadOptions::for_direct_gguf(
                "metal-host-admission-control",
                std::path::Path::new(path),
            )
            .with_ctx_size(256)
            .with_cache_types("f16", "f16")
            .with_selected_device(SkippyDeviceDescriptor {
                backend_device: "MTL0".into(),
                stable_id: None,
                index: None,
                vram_bytes: None,
            })
            .with_receipts(Some(skippy_serving::receipts::ReceiptsOptions::embedded(
                "metal-host-admission-control".into(),
                None,
            )));
            options.n_gpu_layers = gpu_layers;
            let handle = SkippyModelHandle::load(options)?;
            let backend = handle.runtime.receipts_backend();
            let router_present = handle.runtime.receipts_router().is_some();
            let issues = handle.issues_receipts();
            handle.shutdown();
            assert_eq!(issues, expected, "{path:?}, gpu_layers={gpu_layers}");
            assert_eq!(router_present, expected);
            assert_eq!(backend.as_deref(), expected.then_some("MTL0"));
        }
        Ok(())
    }

    // A backend the verifier has no rounding profiles for records receipts
    // that every audit skips: it must not count as issuing them.
    #[test]
    fn only_a_backend_with_rounding_profiles_issues_auditable_receipts() {
        for backend in ["CPU", "CUDA0", "CUDA1", "MTL0"] {
            assert!(receipts_backend_is_audited(backend), "{backend}");
        }
        // Vulkan among them: an honest Vulkan node fails audits
        for backend in ["Vulkan0", "Metal", "ROCm0", "SYCL0", "unknown", ""] {
            assert!(!receipts_backend_is_audited(backend), "{backend}");
        }
    }
}
