use super::{WiringBehavior, WiringEntry, WiringStatus};

pub(super) const MODE: WiringEntry = WiringEntry {
    path: "topology.mode",
    status: WiringStatus::Wired,
    owner: "n/a",
    reason: "",
    behavior: WiringBehavior::None,
};

pub(super) const MANIFEST_SHA256: WiringEntry = WiringEntry {
    path: "topology.manifest_sha256",
    status: WiringStatus::Wired,
    owner: "n/a",
    reason: "",
    behavior: WiringBehavior::None,
};

pub(super) const STAGES: WiringEntry = WiringEntry {
    path: "topology.stages",
    status: WiringStatus::Wired,
    owner: "n/a",
    reason: "",
    behavior: WiringBehavior::None,
};
