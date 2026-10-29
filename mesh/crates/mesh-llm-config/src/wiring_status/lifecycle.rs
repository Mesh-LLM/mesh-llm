//! Native request-driven model residency consumers.
use super::{WiringBehavior, WiringEntry, WiringStatus};

pub(super) const LOAD: WiringEntry = entry("lifecycle.load");
pub(super) const IDLE: WiringEntry = entry("lifecycle.idle_timeout_secs");
pub(super) const PREFETCH: WiringEntry = entry("lifecycle.prefetch");

const fn entry(path: &'static str) -> WiringEntry {
    WiringEntry {
        path,
        status: WiringStatus::Wired,
        owner: "native residency",
        reason: "runtime/native_model_lifecycle.rs owns on-demand native loading, disk prefetch and idle instance eviction",
        behavior: WiringBehavior::None,
    }
}
