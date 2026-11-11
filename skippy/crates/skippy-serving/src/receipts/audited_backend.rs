//! Whether a stage runs on a backend an audit covers.
//!
//! An audit checks a stage's matrix products against the rounding of its
//! backend, and has measured profiles for CPU, CUDA and Metal
//! ([`skippy_receipts::audit::BackendFamily`]). On any other backend
//! (Vulkan, ROCm) a stage issues no receipts: an honest Vulkan node with
//! K-quant weights failed most audits. So its recorder does not start,
//! which also spares it the cost of recording, the stage serves every
//! request as usual, and an embedder that asks whether the runtime issues
//! receipts hears that it issues none.
//!
//! Metal's profile is measured through `skippy serve` on one GPU, the Apple
//! M5 Max, and a receipt names only `MTL0`, not the GPU. Other devices
//! have not been qualified with these profiles: this is a measurement
//! boundary, not evidence of a chip-specific defect. A Metal stage records
//! only on a GPU in [`QUALIFIED_METAL`].

use skippy_receipts::audit::BackendFamily;

/// Exact device descriptions qualified by served audits, not chip families.
const QUALIFIED_METAL: &[&str] = &["Apple M5 Max"];

/// Why a stage on `backend`, the device `description`, issues no receipts,
/// when it issues none.
pub(super) fn unaudited(backend: &str, description: Option<&str>) -> Option<String> {
    match BackendFamily::of(backend) {
        None => Some(format!(
            "the stage runs on backend {backend}, and an audit covers CPU, CUDA and Metal alone"
        )),
        Some(BackendFamily::Metal) if !qualified_metal(description) => Some(format!(
            "the stage runs on Metal GPU {}, and an audit covers only Metal GPUs measured \
             to pass ({})",
            description.unwrap_or("unknown"),
            QUALIFIED_METAL.join(", ")
        )),
        Some(_) => None,
    }
}

fn qualified_metal(description: Option<&str>) -> bool {
    description.is_some_and(|description| QUALIFIED_METAL.contains(&description))
}

#[cfg(test)]
mod tests {
    use super::unaudited;

    #[test]
    fn only_a_backend_with_rounding_profiles_gets_receipts() {
        for backend in ["CPU", "CUDA0", "CUDA1"] {
            assert_eq!(unaudited(backend, None), None, "{backend}");
        }
        for backend in ["Vulkan0", "Vulkan1", "Metal", "ROCm0", "unknown", ""] {
            let reason = unaudited(backend, None).unwrap();
            assert!(
                reason.contains("CPU, CUDA and Metal alone"),
                "{backend}: {reason}"
            );
        }
        assert!(
            unaudited("Vulkan0", None)
                .unwrap()
                .contains("backend Vulkan0")
        );
    }

    #[test]
    fn only_a_measured_metal_gpu_gets_receipts() {
        assert_eq!(unaudited("MTL0", Some("Apple M5 Max")), None);
        for gpu in [
            Some("Apple M5"),
            Some("Apple M5 Pro"),
            Some("Apple M5 Max extra"),
            Some("Apple M6"),
            Some("Apple M6 Max"),
            Some("Apple M4 Max"),
            Some("Apple M50"),
            Some(""),
            None,
        ] {
            let reason = unaudited("MTL0", gpu).unwrap();
            assert!(reason.contains("Metal GPUs measured"), "{gpu:?}: {reason}");
        }
    }
}
