//! A stage load's entries in this node's topology registry.
//!
//! The registry decides which peers may open transport into a stage, so a
//! load that would replace a running stage records its topology only
//! provisionally. When the load is not accepted, the entries it overwrote
//! come back, and its failure is recorded only when no stage runs in its
//! place: a refused replacement leaves the running stage reachable by its
//! predecessor and reported as it is.

use iroh::EndpointId;

use super::Node;
use crate::inference::skippy::{
    StageControlCommand, StageControlRequest, StageControlResponse, StageLoadRequest,
    StageRuntimeState, StageSourceFetch, StageStatusFilter,
};
use crate::mesh::stage_proto::{
    stage_runtime_status_from_snapshot, stage_runtime_status_key, stage_status_from_load,
    stage_topology_from_load, stage_topology_key,
};
use crate::mesh::stage_transport::{
    LOCAL_STAGE_CONTROL_RESPONSE_TIMEOUT, StageRuntimeStatus, StageTopologyInstance,
    StageTopologyState, wait_local_stage_control_response,
};

type StageControlSender = tokio::sync::mpsc::UnboundedSender<StageControlCommand>;

/// The registry entries a stage load overwrote, to put back if the load is
/// not accepted.
#[must_use = "settle the load so a refused one puts back what it replaced"]
pub(crate) struct ProvisionalStageLoad {
    topology_key: String,
    topology: Option<StageTopologyInstance>,
    statuses: Vec<(String, Option<StageRuntimeStatus>)>,
}

impl StageTopologyState {
    /// Record `load`'s topology and a `Starting` status for each of its
    /// stages, returning the entries they replaced.
    fn record_load(
        &mut self,
        node_id: EndpointId,
        load: &StageLoadRequest,
    ) -> ProvisionalStageLoad {
        let topology = stage_topology_from_load(node_id, load);
        let topology_key = stage_topology_key(&topology.topology_id, &topology.run_id);
        let previous = ProvisionalStageLoad {
            topology: self.topologies.get(&topology_key).cloned(),
            statuses: topology
                .stages
                .iter()
                .map(|stage| {
                    let key = stage_runtime_status_key(
                        &topology.topology_id,
                        &topology.run_id,
                        &stage.stage_id,
                    );
                    let status = self.statuses.get(&key).cloned();
                    (key, status)
                })
                .collect(),
            topology_key,
        };
        self.record_topology(topology.clone());
        for stage in topology.stages {
            let mut snapshot = stage_status_from_load(load, StageRuntimeState::Starting);
            snapshot.stage_id = stage.stage_id;
            snapshot.stage_index = stage.stage_index;
            snapshot.layer_start = stage.layer_start;
            snapshot.layer_end = stage.layer_end;
            snapshot.bind_addr = stage.endpoint.bind_addr;
            snapshot.admission =
                (snapshot.stage_id == load.stage_id).then(|| load.admission.clone());
            snapshot.error = None;
            self.record_status(stage_runtime_status_from_snapshot(
                Some(stage.node_id),
                snapshot,
            ));
        }
        previous
    }

    /// Put back the entries a load replaced.
    fn restore(&mut self, previous: ProvisionalStageLoad) {
        match previous.topology {
            Some(topology) => {
                self.topologies.insert(previous.topology_key, topology);
            }
            None => {
                self.topologies.remove(&previous.topology_key);
            }
        }
        // In reverse, so a stage listed twice gets its first saved status.
        for (key, status) in previous.statuses.into_iter().rev() {
            match status {
                Some(status) => {
                    self.statuses.insert(key, status);
                }
                None => {
                    self.statuses.remove(&key);
                }
            }
        }
    }
}

impl Node {
    /// Record a load this node sends, as the plan its coordinator made.
    pub(crate) async fn record_stage_load_topology(&self, load: &StageLoadRequest) {
        let _ = self
            .stage_topologies
            .lock()
            .await
            .record_load(self.endpoint.id(), load);
    }

    /// Record a load this node is about to execute until it is settled.
    pub(crate) async fn record_provisional_stage_load(
        &self,
        load: &StageLoadRequest,
    ) -> ProvisionalStageLoad {
        self.stage_topologies
            .lock()
            .await
            .record_load(self.endpoint.id(), load)
    }

    /// Settle a stage control request this node executed, given its
    /// response. An accepted load keeps its records and its ready status. A
    /// load that was not accepted puts back what it replaced, and its failure
    /// is recorded only when the stage loop runs no stage in its place.
    /// Requests that are not loads are left to the caller.
    pub(crate) async fn settle_executed_stage_load(
        &self,
        control_tx: Option<&StageControlSender>,
        request: &StageControlRequest,
        provisional: Option<ProvisionalStageLoad>,
        response: &StageControlResponse,
    ) {
        let (StageControlRequest::Load(load) | StageControlRequest::LoadLocal(load)) = request
        else {
            return;
        };
        let StageControlResponse::Ready(ready) = response else {
            if let Some(previous) = provisional {
                self.stage_topologies.lock().await.restore(previous);
            }
            return;
        };
        let record_status = ready.accepted || !self.stage_is_running(control_tx, load).await;
        let mut registry = self.stage_topologies.lock().await;
        if !ready.accepted
            && let Some(previous) = provisional
        {
            registry.restore(previous);
        }
        if record_status {
            registry.record_status(stage_runtime_status_from_snapshot(
                Some(self.endpoint.id()),
                ready.status.clone(),
            ));
        }
    }

    /// Whether the stage loop runs a stage at `load`'s key. When the loop
    /// cannot answer, the stage is taken to be running, so its records stay.
    async fn stage_is_running(
        &self,
        control_tx: Option<&StageControlSender>,
        load: &StageLoadRequest,
    ) -> bool {
        let Some(tx) = control_tx else {
            return false;
        };
        let (resp, rx) = tokio::sync::oneshot::channel();
        let request = StageControlRequest::Status(StageStatusFilter {
            topology_id: Some(load.topology_id.clone()),
            run_id: Some(load.run_id.clone()),
            stage_id: Some(load.stage_id.clone()),
        });
        if tx
            .send(StageControlCommand::Execute {
                request,
                requester: self.endpoint.id(),
                source_fetch: StageSourceFetch::LocalOnly,
                resp,
            })
            .is_err()
        {
            return false;
        }
        match wait_local_stage_control_response(rx, LOCAL_STAGE_CONTROL_RESPONSE_TIMEOUT).await {
            Ok(StageControlResponse::Status(statuses)) => !statuses.is_empty(),
            Ok(_) | Err(_) => true,
        }
    }
}
