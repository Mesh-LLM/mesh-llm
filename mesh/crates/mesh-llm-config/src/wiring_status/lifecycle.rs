//! Shared native and provider model residency controls.
use super::{WiringBehavior, WiringEntry, WiringStatus};

pub(super) const LOAD: WiringEntry = entry("lifecycle.load");
pub(super) const IDLE: WiringEntry = entry("lifecycle.idle_timeout_secs");
pub(super) const PREFETCH: WiringEntry = entry("lifecycle.prefetch");

const fn entry(path: &'static str) -> WiringEntry {
    WiringEntry {
        path,
        status: WiringStatus::Wired,
        owner: "native/provider residency",
        reason: "Native model lifecycle and plugin model selection consume loading, disk prefetch and idle residency policies",
        behavior: WiringBehavior::None,
    }
}
