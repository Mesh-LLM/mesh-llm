use super::*;

#[derive(Clone, Debug)]
pub(crate) struct StageTransportBridgeOwner(Arc<()>);

impl StageTransportBridgeOwner {
    pub(crate) fn new() -> Self {
        Self(Arc::new(()))
    }

    pub(crate) fn is_same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

pub(crate) struct StageTransportBridgeLabel<'a> {
    topology_id: &'a str,
    run_id: &'a str,
    stage_id: &'a str,
}

impl<'a> StageTransportBridgeLabel<'a> {
    pub(crate) const fn new(topology_id: &'a str, run_id: &'a str, stage_id: &'a str) -> Self {
        Self {
            topology_id,
            run_id,
            stage_id,
        }
    }

    pub(crate) fn duplicate_error(&self) -> anyhow::Error {
        anyhow::anyhow!(
            "stage transport bridge already exists for {}/{}/{}",
            self.topology_id,
            self.run_id,
            self.stage_id
        )
    }

    pub(crate) fn cancelled_error(&self) -> anyhow::Error {
        anyhow::anyhow!(
            "stage transport bridge reservation was cancelled for {}/{}/{}",
            self.topology_id,
            self.run_id,
            self.stage_id
        )
    }
}

pub(crate) enum StageTransportBridge {
    Reserved {
        owner: StageTransportBridgeOwner,
    },
    Running {
        owner: StageTransportBridgeOwner,
        handle: tokio::task::JoinHandle<()>,
    },
}

impl StageTransportBridge {
    pub(crate) fn reserved(owner: StageTransportBridgeOwner) -> Self {
        Self::Reserved { owner }
    }

    pub(crate) fn running(
        owner: StageTransportBridgeOwner,
        handle: tokio::task::JoinHandle<()>,
    ) -> Self {
        Self::Running { owner, handle }
    }

    pub(crate) fn is_owned_by(&self, owner: &StageTransportBridgeOwner) -> bool {
        self.owner().is_same(owner)
    }

    pub(crate) fn is_reserved_by(&self, owner: &StageTransportBridgeOwner) -> bool {
        match self {
            Self::Reserved { owner: existing } => existing.is_same(owner),
            Self::Running { .. } => false,
        }
    }

    #[cfg(test)]
    pub(crate) const fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }

    pub(crate) fn abort(self) {
        if let Self::Running { handle, .. } = self {
            handle.abort();
        }
    }

    const fn owner(&self) -> &StageTransportBridgeOwner {
        match self {
            Self::Reserved { owner } | Self::Running { owner, .. } => owner,
        }
    }
}

impl Node {
    pub(crate) async fn reserve_stage_transport_bridge(
        &self,
        key: String,
        label: &StageTransportBridgeLabel<'_>,
    ) -> Result<StageTransportBridgeOwner> {
        let mut bridges = self.stage_transport_bridges.lock().await;
        if bridges.contains_key(&key) {
            return Err(label.duplicate_error());
        }
        let owner = StageTransportBridgeOwner::new();
        bridges.insert(key, StageTransportBridge::reserved(owner.clone()));
        Ok(owner)
    }

    pub(crate) async fn publish_stage_transport_bridge(
        &self,
        key: String,
        owner: StageTransportBridgeOwner,
        handle: tokio::task::JoinHandle<()>,
    ) -> bool {
        let mut bridges = self.stage_transport_bridges.lock().await;
        let Some(existing) = bridges.get(&key) else {
            handle.abort();
            return false;
        };
        if !existing.is_reserved_by(&owner) {
            handle.abort();
            return false;
        }
        bridges.insert(key, StageTransportBridge::running(owner, handle));
        true
    }

    pub(crate) async fn remove_stage_transport_bridge_if_owner(
        &self,
        key: &str,
        owner: &StageTransportBridgeOwner,
    ) -> Option<StageTransportBridge> {
        let mut bridges = self.stage_transport_bridges.lock().await;
        if bridges
            .get(key)
            .is_some_and(|bridge| bridge.is_owned_by(owner))
        {
            return bridges.remove(key);
        }
        None
    }

    #[cfg(test)]
    pub(crate) async fn stage_transport_bridge_is_running(&self, key: &str) -> bool {
        self.stage_transport_bridges
            .lock()
            .await
            .get(key)
            .is_some_and(StageTransportBridge::is_running)
    }

    #[cfg(test)]
    pub(crate) async fn stage_transport_bridge_owner_matches(
        &self,
        key: &str,
        owner: &StageTransportBridgeOwner,
    ) -> bool {
        self.stage_transport_bridges
            .lock()
            .await
            .get(key)
            .is_some_and(|bridge| bridge.is_owned_by(owner))
    }

    #[cfg(test)]
    pub(crate) async fn stage_transport_bridge_exists(&self, key: &str) -> bool {
        self.stage_transport_bridges.lock().await.contains_key(key)
    }

    #[cfg(test)]
    pub(crate) async fn insert_reserved_stage_transport_bridge_for_test(
        &self,
        key: String,
        owner: StageTransportBridgeOwner,
    ) {
        self.stage_transport_bridges
            .lock()
            .await
            .insert(key, StageTransportBridge::reserved(owner));
    }

    #[cfg(test)]
    pub(crate) async fn reserve_stage_transport_bridge_for_test(
        &self,
        key: String,
    ) -> StageTransportBridgeOwner {
        let owner = StageTransportBridgeOwner::new();
        self.insert_reserved_stage_transport_bridge_for_test(key, owner.clone())
            .await;
        owner
    }

    #[cfg(test)]
    pub(crate) async fn clear_stage_transport_bridge_for_test(&self, key: &str) {
        if let Some(bridge) = self.stage_transport_bridges.lock().await.remove(key) {
            bridge.abort();
        }
    }
}
