use super::{WiringBehavior, WiringEntry, WiringStatus};

const UNWIRED_REASON: &str =
    "Skippy serves GGUF only and no longer quantizes SafeTensors checkpoints while loading";

pub(super) const QUANTIZATION: WiringEntry = WiringEntry {
    path: "hardware.checkpoint_quantization",
    status: WiringStatus::Unwired,
    owner: "Skippy GGUF-only serving",
    reason: UNWIRED_REASON,
    behavior: WiringBehavior::SilentNoOp,
};

pub(super) const IMATRIX: WiringEntry = WiringEntry {
    path: "hardware.checkpoint_imatrix",
    status: WiringStatus::Unwired,
    owner: "Skippy GGUF-only serving",
    reason: UNWIRED_REASON,
    behavior: WiringBehavior::SilentNoOp,
};
