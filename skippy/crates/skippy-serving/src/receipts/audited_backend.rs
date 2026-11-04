//! Whether a stage runs on a backend an audit covers.
//!
//! An audit checks a stage's matrix products against the rounding of its
//! backend, and has measured profiles for CPU and CUDA alone
//! ([`skippy_receipts::audit::BackendFamily`]). On any other backend
//! (Vulkan, Metal, ROCm) a stage issues no receipts: an honest Vulkan node
//! with K-quant weights failed most audits, and Metal's prefill fits no
//! profile. So its recorder does not start, which also spares it the cost
//! of recording, the stage serves every request as usual, and an embedder
//! that asks whether the runtime issues receipts hears that it issues
//! none.

use skippy_receipts::audit::BackendFamily;

/// Why a stage on `backend` issues no receipts, when it issues none.
pub(super) fn unaudited(backend: &str) -> Option<String> {
    BackendFamily::of(backend).is_none().then(|| {
        format!("the stage runs on backend {backend}, and an audit covers CPU and CUDA alone")
    })
}

#[cfg(test)]
mod tests {
    use super::unaudited;

    #[test]
    fn only_a_backend_with_rounding_profiles_gets_receipts() {
        for backend in ["CPU", "CUDA0", "CUDA1"] {
            assert_eq!(unaudited(backend), None, "{backend}");
        }
        for backend in [
            "Vulkan0", "Vulkan1", "Metal", "MTL0", "ROCm0", "unknown", "",
        ] {
            let reason = unaudited(backend).unwrap();
            assert!(reason.contains("CPU and CUDA alone"), "{backend}: {reason}");
        }
        assert!(unaudited("Vulkan0").unwrap().contains("backend Vulkan0"));
    }
}
