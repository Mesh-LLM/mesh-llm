//! Advertised memory breakdown payload, extracted from oversized status.rs.
//!
//! Mirrors `crate::mesh::AdvertisedMemory` for `/api/status` (`my_memory` and
//! `peers[].memory`); keep the shape stable, the console reads it as is.

use serde::Serialize;

/// Itemized capacity behind a node's advertised `vram_gb`, in bytes, exactly
/// as the node announces it on the mesh: `total_bytes` is the enumerated
/// accelerator memory, the three reserves are what is withheld from it, and
/// `usable_bytes` is what remains for placement, so that
/// `total_bytes == reserved_bytes + platform_reserve_bytes + configured_reserve_bytes + usable_bytes`.
/// `system_ram_bytes` and `ram_offload_bytes` describe the node's local fit
/// budget and are informational only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub(crate) struct MemoryPayload {
    pub(crate) total_bytes: u64,
    pub(crate) reserved_bytes: u64,
    pub(crate) platform_reserve_bytes: u64,
    pub(crate) configured_reserve_bytes: u64,
    pub(crate) usable_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system_ram_bytes: Option<u64>,
    pub(crate) ram_offload_bytes: u64,
}

impl From<crate::mesh::AdvertisedMemory> for MemoryPayload {
    fn from(memory: crate::mesh::AdvertisedMemory) -> Self {
        Self {
            total_bytes: memory.total_bytes,
            reserved_bytes: memory.reserved_bytes,
            platform_reserve_bytes: memory.platform_reserve_bytes,
            configured_reserve_bytes: memory.configured_reserve_bytes,
            usable_bytes: memory.usable_bytes,
            system_ram_bytes: memory.system_ram_bytes,
            ram_offload_bytes: memory.ram_offload_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_payload_mirrors_the_advertised_breakdown_and_keeps_absent_ram_absent() {
        let payload = MemoryPayload::from(crate::mesh::AdvertisedMemory {
            total_bytes: 12_000_000_000,
            reserved_bytes: 500_000_000,
            platform_reserve_bytes: 0,
            configured_reserve_bytes: 2_000_000_000,
            usable_bytes: 9_500_000_000,
            system_ram_bytes: None,
            ram_offload_bytes: 18_000_000_000,
        });

        let json = serde_json::to_value(payload).expect("memory payload serializes");
        assert_eq!(json["total_bytes"], 12_000_000_000u64);
        assert_eq!(json["reserved_bytes"], 500_000_000u64);
        assert_eq!(json["platform_reserve_bytes"], 0u64);
        assert_eq!(json["configured_reserve_bytes"], 2_000_000_000u64);
        assert_eq!(json["usable_bytes"], 9_500_000_000u64);
        assert_eq!(json["ram_offload_bytes"], 18_000_000_000u64);
        assert!(
            json.get("system_ram_bytes").is_none(),
            "absent system RAM must stay absent: {json}"
        );
    }

    #[test]
    fn memory_payload_reports_system_ram_when_the_survey_knows_it() {
        let payload = MemoryPayload::from(crate::mesh::AdvertisedMemory {
            system_ram_bytes: Some(32_000_000_000),
            ..crate::mesh::AdvertisedMemory::default()
        });

        let json = serde_json::to_value(payload).expect("memory payload serializes");
        assert_eq!(json["system_ram_bytes"], 32_000_000_000u64);
    }
}
