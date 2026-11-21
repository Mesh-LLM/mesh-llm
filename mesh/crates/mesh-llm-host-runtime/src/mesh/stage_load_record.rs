//! A stage load's entries in this node's topology registry.
//!
//! The registry decides which peers may open transport into a stage, so a
//! load that would replace a running stage records its topology only
//! provisionally. Unless the load is accepted, the node asks the stage loop
//! which stage it runs once the load is done and settles the records to
//! match: the load's own when it started after all, the entries it
//! overwrote when the stage it would replace still runs, and its failure
//! when nothing runs. A refused replacement therefore leaves the running
//! stage reachable by its predecessor and reported as it is.

use iroh::EndpointId;

use super::Node;
use crate::inference::skippy::{
    StageControlCommand, StageControlRequest, StageControlResponse, StageLoadRequest,
    StageRuntimeState, StageSourceFetch, StageStatusFilter, StageStatusSnapshot,
};
use crate::mesh::stage_proto::{
    stage_runtime_status_from_snapshot, stage_runtime_status_key, stage_status_from_load,
    stage_topology_from_load, stage_topology_key,
};
use crate::mesh::stage_transport::{
    LOCAL_STAGE_CONTROL_RESPONSE_TIMEOUT, StageRuntimeStatus, StageTopologyInstance,
    StageTopologyState,
};

type StageControlSender = tokio::sync::mpsc::UnboundedSender<StageControlCommand>;
type StageStatusAnswer = tokio::sync::oneshot::Receiver<anyhow::Result<StageControlResponse>>;

/// The registry entries a stage load wrote and the ones they replaced, to
/// put back if the load does not take the stage over.
#[must_use = "settle the load so a refused one puts back what it replaced"]
pub(crate) struct ProvisionalStageLoad {
    topology_key: String,
    previous_topology: Option<StageTopologyInstance>,
    written_topology: StageTopologyInstance,
    statuses: Vec<ProvisionalStatus>,
}

struct ProvisionalStatus {
    key: String,
    previous: Option<StageRuntimeStatus>,
    written: StageRuntimeStatus,
}

/// What the stage loop runs at a load's key once the load is done.
enum RunningAtKey {
    /// The load's own stage.
    TheLoad(Box<StageStatusSnapshot>),
    /// Another stage, such as the one the load would have replaced.
    AnotherStage,
    Nothing,
    /// The loop gave no answer.
    Unknown,
}

/// A load that was not accepted, waiting for the stage loop's answer.
struct UnsettledLoad {
    load: StageLoadRequest,
    provisional: Option<ProvisionalStageLoad>,
    failure: StageStatusSnapshot,
}

impl StageTopologyState {
    /// Record `load`'s topology and a `Starting` status for each of its
    /// stages, returning what they wrote and what they replaced.
    fn record_load(
        &mut self,
        node_id: EndpointId,
        load: &StageLoadRequest,
    ) -> ProvisionalStageLoad {
        let topology = stage_topology_from_load(node_id, load);
        let topology_key = stage_topology_key(&topology.topology_id, &topology.run_id);
        let previous_topology = self.topologies.get(&topology_key).cloned();
        self.record_topology(topology.clone());
        let mut statuses = Vec::with_capacity(topology.stages.len());
        for stage in &topology.stages {
            let mut snapshot = stage_status_from_load(load, StageRuntimeState::Starting);
            snapshot.stage_id = stage.stage_id.clone();
            snapshot.stage_index = stage.stage_index;
            snapshot.layer_start = stage.layer_start;
            snapshot.layer_end = stage.layer_end;
            snapshot.bind_addr = stage.endpoint.bind_addr.clone();
            snapshot.admission =
                (snapshot.stage_id == load.stage_id).then(|| load.admission.clone());
            snapshot.error = None;
            let written = stage_runtime_status_from_snapshot(Some(stage.node_id), snapshot);
            let key =
                stage_runtime_status_key(&written.topology_id, &written.run_id, &written.stage_id);
            let previous = self.statuses.get(&key).cloned();
            self.record_status(written.clone());
            statuses.push(ProvisionalStatus {
                key,
                previous,
                written,
            });
        }
        ProvisionalStageLoad {
            topology_key,
            previous_topology,
            written_topology: topology,
            statuses,
        }
    }

    /// Put back the entries a load replaced, each only while it still holds
    /// what the load wrote: a status refreshed or a topology replaced since
    /// is newer than both.
    fn restore(&mut self, provisional: ProvisionalStageLoad) {
        let topology_unchanged = self
            .topologies
            .get(&provisional.topology_key)
            .is_some_and(|current| same_plan(current, &provisional.written_topology));
        if topology_unchanged {
            match provisional.previous_topology {
                Some(topology) => {
                    self.topologies.insert(provisional.topology_key, topology);
                }
                None => {
                    self.topologies.remove(&provisional.topology_key);
                }
            }
        }
        // In reverse, so a stage listed twice gets its first saved status.
        for status in provisional.statuses.into_iter().rev() {
            if self.statuses.get(&status.key) != Some(&status.written) {
                continue;
            }
            match status.previous {
                Some(previous) => {
                    self.statuses.insert(status.key, previous);
                }
                None => {
                    self.statuses.remove(&status.key);
                }
            }
        }
    }
}

/// Whether `current` is still the topology a load wrote. Its stages'
/// endpoints may since have taken the addresses their statuses reported.
fn same_plan(current: &StageTopologyInstance, written: &StageTopologyInstance) -> bool {
    let mut current = current.clone();
    for (stage, written) in current.stages.iter_mut().zip(&written.stages) {
        stage.endpoint = written.endpoint.clone();
    }
    current == *written
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

    /// Settle a stage control request this node executed, given how it
    /// ended: its response, or the error its caller got instead. An accepted
    /// load keeps its records and its ready status. Otherwise the records
    /// follow what the stage loop runs at the load's key once the load is
    /// done; a load the loop is still running is settled when it finishes.
    /// Requests that are not loads are left to the caller.
    pub(crate) async fn settle_executed_stage_load(
        &self,
        control_tx: Option<&StageControlSender>,
        request: &StageControlRequest,
        provisional: Option<ProvisionalStageLoad>,
        outcome: Result<&StageControlResponse, &anyhow::Error>,
    ) {
        let (StageControlRequest::Load(load) | StageControlRequest::LoadLocal(load)) = request
        else {
            return;
        };
        if let Ok(StageControlResponse::Ready(ready)) = outcome
            && ready.accepted
        {
            self.record_stage_status(Some(self.endpoint.id()), ready.status.clone())
                .await;
            return;
        }
        let unsettled = UnsettledLoad {
            load: load.clone(),
            provisional,
            failure: failed_load_status(load, outcome),
        };
        let Some(mut answer) = self.ask_what_runs_at(control_tx, load) else {
            self.settle(unsettled, RunningAtKey::Nothing).await;
            return;
        };
        // The loop answers once it is done with the load, which it may
        // still be running if the caller stopped waiting.
        match tokio::time::timeout(LOCAL_STAGE_CONTROL_RESPONSE_TIMEOUT, &mut answer).await {
            Ok(answer) => {
                let running = running_at_key(&unsettled.load, answer);
                self.settle(unsettled, running).await;
            }
            Err(_) => {
                let node = self.clone();
                let wait = crate::inference::skippy::stage_load_timeout(load);
                tokio::spawn(async move {
                    let running = match tokio::time::timeout(wait, answer).await {
                        Ok(answer) => running_at_key(&unsettled.load, answer),
                        Err(_) => RunningAtKey::Unknown,
                    };
                    node.settle(unsettled, running).await;
                });
            }
        }
    }

    async fn settle(&self, unsettled: UnsettledLoad, running: RunningAtKey) {
        let UnsettledLoad {
            provisional,
            failure,
            ..
        } = unsettled;
        let mut registry = self.stage_topologies.lock().await;
        let status = match running {
            RunningAtKey::TheLoad(status) => Some(*status),
            RunningAtKey::Nothing => {
                if let Some(provisional) = provisional {
                    registry.restore(provisional);
                }
                Some(failure)
            }
            RunningAtKey::AnotherStage | RunningAtKey::Unknown => {
                if let Some(provisional) = provisional {
                    registry.restore(provisional);
                }
                None
            }
        };
        if let Some(status) = status {
            registry.record_status(stage_runtime_status_from_snapshot(
                Some(self.endpoint.id()),
                status,
            ));
        }
    }

    /// Ask the stage loop for the stage at `load`'s key. `None` when there
    /// is no loop to ask.
    fn ask_what_runs_at(
        &self,
        control_tx: Option<&StageControlSender>,
        load: &StageLoadRequest,
    ) -> Option<StageStatusAnswer> {
        let (resp, answer) = tokio::sync::oneshot::channel();
        control_tx?
            .send(StageControlCommand::Execute {
                request: StageControlRequest::Status(StageStatusFilter {
                    topology_id: Some(load.topology_id.clone()),
                    run_id: Some(load.run_id.clone()),
                    stage_id: Some(load.stage_id.clone()),
                }),
                requester: self.endpoint.id(),
                source_fetch: StageSourceFetch::LocalOnly,
                resp,
            })
            .ok()?;
        Some(answer)
    }
}

fn running_at_key(
    load: &StageLoadRequest,
    answer: Result<anyhow::Result<StageControlResponse>, tokio::sync::oneshot::error::RecvError>,
) -> RunningAtKey {
    match answer {
        Ok(Ok(StageControlResponse::Status(statuses))) => match statuses.into_iter().next() {
            Some(status) if is_the_load(&status, load) => RunningAtKey::TheLoad(Box::new(status)),
            Some(_) => RunningAtKey::AnotherStage,
            None => RunningAtKey::Nothing,
        },
        // The loop has shut down, and its stages with it.
        Err(_) => RunningAtKey::Nothing,
        Ok(_) => RunningAtKey::Unknown,
    }
}

/// Whether a running stage's status is `load`'s: the same generation from
/// the same coordinator term.
fn is_the_load(status: &StageStatusSnapshot, load: &StageLoadRequest) -> bool {
    status.shutdown_generation == load.shutdown_generation
        && status.coordinator_term == load.coordinator_term
        && status.coordinator_id == load.coordinator_id
}

/// The status a load that was not accepted reports when nothing runs in its
/// place.
fn failed_load_status(
    load: &StageLoadRequest,
    outcome: Result<&StageControlResponse, &anyhow::Error>,
) -> StageStatusSnapshot {
    if let Ok(StageControlResponse::Ready(ready)) = outcome {
        return ready.status.clone();
    }
    let mut status = stage_status_from_load(load, StageRuntimeState::Failed);
    status.error = Some(match outcome {
        Err(error) => format!("{error:#}"),
        Ok(_) => "unexpected response to a stage load".to_string(),
    });
    status
}
