use super::{WiringBehavior, WiringEntry, WiringStatus};

const PARTIAL_REASON: &str = "Single-node direct SafeTensors resolution propagates it; prepared GGUF and multi-node stage constructors do not consume it";

pub(super) const QUANTIZATION: WiringEntry = WiringEntry {
    path: "hardware.checkpoint_quantization",
    status: WiringStatus::Partial,
    owner: "SafeTensors direct-load follow-up",
    reason: PARTIAL_REASON,
    behavior: WiringBehavior::None,
};

pub(super) const IMATRIX: WiringEntry = WiringEntry {
    path: "hardware.checkpoint_imatrix",
    status: WiringStatus::Partial,
    owner: "SafeTensors direct-load follow-up",
    reason: PARTIAL_REASON,
    behavior: WiringBehavior::None,
};
