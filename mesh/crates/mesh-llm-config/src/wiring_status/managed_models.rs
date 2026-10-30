//! Provider model fields consumed by host selection and plugin initialization.

use super::{WiringBehavior, WiringEntry, WiringStatus};

const fn wired(path: &'static str) -> WiringEntry {
    WiringEntry {
        path,
        status: WiringStatus::Wired,
        owner: "n/a",
        reason: "",
        behavior: WiringBehavior::None,
    }
}

pub(super) const PROVIDER: WiringEntry = wired("provider");
pub(super) const VARIANT: WiringEntry = wired("variant");
pub(super) const OPTIONS: WiringEntry = wired("options");
pub(super) const REQUEST_DEFAULTS: WiringEntry = wired("request_defaults");
pub(super) const LIFECYCLE: WiringEntry = wired("lifecycle");
pub(super) const SOURCE: WiringEntry = wired("source");
