use super::{WiringBehavior, WiringEntry, WiringStatus};

const REJECTED_REASON: &str =
    "Skippy serves GGUF only and no longer quantizes SafeTensors checkpoints while loading";

pub(super) const QUANTIZATION: WiringEntry = WiringEntry {
    path: "hardware.checkpoint_quantization",
    status: WiringStatus::Rejected,
    owner: "Skippy GGUF-only serving",
    reason: REJECTED_REASON,
    behavior: WiringBehavior::Rejected,
};

pub(super) const IMATRIX: WiringEntry = WiringEntry {
    path: "hardware.checkpoint_imatrix",
    status: WiringStatus::Rejected,
    owner: "Skippy GGUF-only serving",
    reason: REJECTED_REASON,
    behavior: WiringBehavior::Rejected,
};
