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
    /// ([`skippy_receipts::audit::BackendFamily`]): CPU and CUDA. Receipts
    /// from any other backend (Vulkan, Metal, ROCm) are ones every audit
    /// skips, so it must not satisfy the receipts trust tier.
    ///
    /// The recorder of a stage on such a backend, or with a quantized KV
    /// cache, does not start (`skippy_serving::receipts::start` reads the
    /// backend and the cache types of the stage config), so the runtime
    /// then names no receipts backend and this is false: the host needs
    /// no knowledge of the cache type. The backend is checked here again
    /// for a runtime that predates that rule.
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

    // A backend the verifier has no rounding profiles for records receipts
    // that every audit skips: it must not count as issuing them.
    #[test]
    fn only_a_backend_with_rounding_profiles_issues_auditable_receipts() {
        for backend in ["CPU", "CUDA0", "CUDA1"] {
            assert!(receipts_backend_is_audited(backend), "{backend}");
        }
        // Vulkan among them: an honest Vulkan node fails audits
        for backend in ["Vulkan0", "Metal", "MTL0", "ROCm0", "SYCL0", "unknown", ""] {
            assert!(!receipts_backend_is_audited(backend), "{backend}");
        }
    }
}
