//! The `[receipts]` settings: all read by the node today.

use super::{WiringBehavior, WiringEntry, WiringStatus};

pub(super) const ENABLED: WiringEntry = WiringEntry {
    path: "receipts.enabled",
    status: WiringStatus::Wired,
    owner: "n/a",
    reason: "",
    behavior: WiringBehavior::None,
};

pub(super) const AUDIT: WiringEntry = WiringEntry {
    path: "receipts.audit",
    status: WiringStatus::Wired,
    owner: "n/a",
    reason: "",
    behavior: WiringBehavior::None,
};

pub(super) const REQUIRE: WiringEntry = WiringEntry {
    path: "receipts.require",
    status: WiringStatus::Wired,
    owner: "n/a",
    reason: "",
    behavior: WiringBehavior::None,
};

pub(super) const OFFLINE: WiringEntry = WiringEntry {
    path: "receipts.offline",
    status: WiringStatus::Wired,
    owner: "n/a",
    reason: "",
    behavior: WiringBehavior::None,
};
