use std::sync::{Arc, atomic::Ordering};

use mesh_llm_events::logging::{
    events::LifecycleEvent, identifiers::RequestId, replay::ReplayChannel,
};

use super::super::TerminalOutcome;

use super::{
    FrontendAdmissionDecision, LifecycleGuard, LifecycleOwnerEntry, LoggingService,
    MAX_TRACKED_REQUESTS, RawMeshLifecycleEntry, RawMeshLifecycleOwners, RawMeshRequestLifecycle,
    RequestSummaryMetadata, lock_recover,
};

impl RawMeshLifecycleOwners {
    pub(crate) fn is_claimed(&self, request_id: RequestId) -> bool {
        let coordination = lock_recover(&self.coordination);
        coordination.owners.contains_key(&request_id)
            || coordination.remote_attributions.contains_key(&request_id)
            || coordination.remote_suppressions.contains_key(&request_id)
    }

    /// Coordinate frontend eligibility and registration with raw claims,
    /// authenticated attribution, and suppression placement. The callback may
    /// briefly lock the frontend tracker and service registry; callers must not
    /// enter this method while holding either inner lock.
    pub(crate) fn admit_frontend(
        &self,
        request_id: RequestId,
        register: impl FnOnce() -> FrontendAdmissionDecision,
    ) {
        let mut coordination = lock_recover(&self.coordination);
        if coordination.owners.contains_key(&request_id)
            || coordination.remote_attributions.contains_key(&request_id)
            || coordination.remote_suppressions.contains_key(&request_id)
        {
            return;
        }
        let FrontendAdmissionDecision::Registered { evicted } = register() else {
            return;
        };
        if let Some(evicted) = evicted
            && matches!(
                coordination.owners.get(&evicted),
                Some(LifecycleOwnerEntry::Frontend)
            )
        {
            coordination.owners.remove(&evicted);
        }
        coordination
            .owners
            .insert(request_id, LifecycleOwnerEntry::Frontend);
    }

    pub(crate) fn release_frontend(&self, request_id: RequestId) {
        let mut coordination = lock_recover(&self.coordination);
        if matches!(
            coordination.owners.get(&request_id),
            Some(LifecycleOwnerEntry::Frontend)
        ) {
            coordination.owners.remove(&request_id);
        }
    }

    fn claim(
        &self,
        service: &LoggingService,
        request_id: RequestId,
        metadata: RequestSummaryMetadata,
    ) -> Option<(LifecycleGuard, u64)> {
        let mut coordination = lock_recover(&self.coordination);
        if let Some(existing) = coordination.owners.get(&request_id) {
            return match existing {
                LifecycleOwnerEntry::Raw(existing) => {
                    let claim = (existing.guard.clone(), existing.token);
                    service.merge_request_metadata(request_id, metadata);
                    Some(claim)
                }
                LifecycleOwnerEntry::Frontend => {
                    service.merge_request_metadata(request_id, metadata);
                    None
                }
            };
        }
        if coordination.raw_owner_count >= MAX_TRACKED_REQUESTS {
            return None;
        }

        let mut metadata = metadata;
        if let Some(attribution) = coordination.remote_attributions.remove(&request_id) {
            metadata.merge_authenticated_remote_caller(attribution.metadata);
        }
        let (guard, _) = service.register_request_with_metadata(request_id, metadata);
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        coordination.owners.insert(
            request_id,
            LifecycleOwnerEntry::Raw(RawMeshLifecycleEntry {
                guard: guard.clone(),
                token,
                route_selected: false,
                stream_started: false,
                first_token_recorded: false,
                stream_completed: false,
                stream_error: false,
            }),
        );
        coordination.raw_owner_count += 1;
        Some((guard, token))
    }

    pub(super) fn emit_route_selected(
        &self,
        service: &LoggingService,
        request_id: RequestId,
        token: u64,
        model: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
    ) {
        let should_emit = {
            let mut coordination = lock_recover(&self.coordination);
            let Some(LifecycleOwnerEntry::Raw(entry)) = coordination.owners.get_mut(&request_id)
            else {
                return;
            };
            if entry.token != token || entry.route_selected {
                false
            } else {
                entry.route_selected = true;
                true
            }
        };
        if !should_emit {
            return;
        }

        let metadata = RequestSummaryMetadata::from_parts(None, model, provider, engine);
        service.merge_request_metadata(request_id, metadata.clone());
        let event = LifecycleEvent::RouteSelected {
            model: metadata.model().map(str::to_owned),
            provider: metadata.provider().map(str::to_owned),
            engine: metadata.engine().map(str::to_owned),
        };
        if let Ok(payload) = serde_json::to_string(&event) {
            let _ = service.enqueue_event(request_id, ReplayChannel::Operations, payload);
        }
    }

    fn release(&self, request_id: RequestId, token: u64) {
        let mut coordination = lock_recover(&self.coordination);
        if matches!(
            coordination.owners.get(&request_id),
            Some(LifecycleOwnerEntry::Raw(entry)) if entry.token == token
        ) {
            coordination.owners.remove(&request_id);
            coordination.raw_owner_count -= 1;
        }
    }
}

impl RawMeshRequestLifecycle {
    pub(crate) fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub(crate) fn register(
        service: Arc<LoggingService>,
        owners: Arc<RawMeshLifecycleOwners>,
        request_id: RequestId,
    ) -> Option<Self> {
        Self::register_with_metadata(
            service,
            owners,
            request_id,
            RequestSummaryMetadata::default(),
        )
    }

    /// Register a raw ingress parent with the metadata known before routing.
    pub(crate) fn register_with_metadata(
        service: Arc<LoggingService>,
        owners: Arc<RawMeshLifecycleOwners>,
        request_id: RequestId,
        metadata: RequestSummaryMetadata,
    ) -> Option<Self> {
        let (guard, token) = owners.claim(&service, request_id, metadata)?;
        Some(Self {
            service,
            owners,
            request_id,
            token,
            guard,
        })
    }

    pub(crate) fn terminal(&self, outcome: TerminalOutcome) {
        let _ = self
            .service
            .transition_terminal(self.request_id, &self.guard, outcome);
        self.owners.release(self.request_id, self.token);
    }
}

impl Drop for RawMeshRequestLifecycle {
    fn drop(&mut self) {
        self.owners.release(self.request_id, self.token);
    }
}
