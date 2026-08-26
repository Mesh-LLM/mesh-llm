use std::sync::{Arc, atomic::Ordering};

use mesh_llm_events::logging::identifiers::RequestId;

use super::{
    LoggingService, MAX_TRACKED_REQUESTS, RawMeshLifecycleOwners, RemoteAttributionEntry,
    RemoteAttributionPlacement, RemoteSuppressionEntry, RequestSummaryMetadata, lock_recover,
};

/// A fail-open, process-local marker for a trusted remote tunnel request.
///
/// Unlike a raw ingress lifecycle, this never registers a logging parent. It
/// only prevents the target's embedded frontend observer from registering a
/// duplicate parent while the authenticated tunnel relay is active.
pub(crate) struct RawMeshRemoteSuppressionLease {
    owners: Arc<RawMeshLifecycleOwners>,
    request_id: RequestId,
    token: u64,
}

pub(crate) struct RawMeshRemoteAttributionLease {
    owners: Arc<RawMeshLifecycleOwners>,
    request_id: RequestId,
    token: Option<u64>,
}

impl RawMeshLifecycleOwners {
    fn acquire_remote_suppression(&self, request_id: RequestId) -> Option<u64> {
        let mut coordination = lock_recover(&self.coordination);
        if let Some(existing) = coordination.remote_suppressions.get_mut(&request_id) {
            existing.leases = existing.leases.saturating_add(1);
            return Some(existing.token);
        }
        if coordination.remote_suppressions.len() >= MAX_TRACKED_REQUESTS {
            return None;
        }

        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        coordination
            .remote_suppressions
            .insert(request_id, RemoteSuppressionEntry { token, leases: 1 });
        Some(token)
    }

    fn attribute_remote_caller(
        &self,
        service: &LoggingService,
        request_id: RequestId,
        metadata: RequestSummaryMetadata,
    ) -> Option<RemoteAttributionPlacement> {
        if !metadata.has_authenticated_remote_caller() {
            return None;
        }
        let mut coordination = lock_recover(&self.coordination);
        if coordination.owners.contains_key(&request_id) {
            service.merge_authenticated_remote_caller(request_id, metadata);
            return Some(RemoteAttributionPlacement::Applied);
        }

        if let Some(existing) = coordination.remote_attributions.get_mut(&request_id) {
            existing.leases = existing.leases.saturating_add(1);
            return Some(RemoteAttributionPlacement::Pending(existing.token));
        }
        if coordination.remote_attributions.len() >= MAX_TRACKED_REQUESTS {
            return None;
        }
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        coordination.remote_attributions.insert(
            request_id,
            RemoteAttributionEntry {
                token,
                leases: 1,
                metadata,
            },
        );
        Some(RemoteAttributionPlacement::Pending(token))
    }

    fn release_remote_attribution(&self, request_id: RequestId, token: u64) {
        let mut coordination = lock_recover(&self.coordination);
        let Some(entry) = coordination.remote_attributions.get_mut(&request_id) else {
            return;
        };
        if entry.token != token {
            return;
        }
        if entry.leases > 1 {
            entry.leases -= 1;
        } else {
            coordination.remote_attributions.remove(&request_id);
        }
    }

    fn release_remote_suppression(&self, request_id: RequestId, token: u64) {
        let mut coordination = lock_recover(&self.coordination);
        let Some(entry) = coordination.remote_suppressions.get_mut(&request_id) else {
            return;
        };
        if entry.token != token {
            return;
        }
        if entry.leases > 1 {
            entry.leases -= 1;
        } else {
            coordination.remote_suppressions.remove(&request_id);
        }
    }
}

impl RawMeshRemoteSuppressionLease {
    pub(crate) fn acquire(
        owners: Arc<RawMeshLifecycleOwners>,
        request_id: RequestId,
    ) -> Option<Self> {
        let token = owners.acquire_remote_suppression(request_id)?;
        Some(Self {
            owners,
            request_id,
            token,
        })
    }
}

impl RawMeshRemoteAttributionLease {
    pub(crate) fn acquire(
        service: &LoggingService,
        owners: Arc<RawMeshLifecycleOwners>,
        request_id: RequestId,
        metadata: RequestSummaryMetadata,
    ) -> Option<Self> {
        let placement = owners.attribute_remote_caller(service, request_id, metadata)?;
        Some(Self {
            owners,
            request_id,
            token: match placement {
                RemoteAttributionPlacement::Applied => None,
                RemoteAttributionPlacement::Pending(token) => Some(token),
            },
        })
    }
}

impl Drop for RawMeshRemoteAttributionLease {
    fn drop(&mut self) {
        if let Some(token) = self.token {
            self.owners
                .release_remote_attribution(self.request_id, token);
        }
    }
}

impl Drop for RawMeshRemoteSuppressionLease {
    fn drop(&mut self) {
        self.owners
            .release_remote_suppression(self.request_id, self.token);
    }
}
