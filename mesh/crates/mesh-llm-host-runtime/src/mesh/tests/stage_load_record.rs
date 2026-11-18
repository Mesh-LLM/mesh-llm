//! A stage load this node refuses leaves the stage it would replace running,
//! reachable by its predecessor, and reported as it is.

use super::*;
use crate::inference::skippy::{
    StageControlCommand, StageControlRequest, StageControlResponse, StageLoadRequest,
    StageReadyResponse, StageRuntimeState, StageSourceFetch, StageStatusFilter,
    StageTopologyStageDescriptor,
};
use crate::mesh::stage_proto::stage_status_from_load;

/// Stage 1 of a two-stage split, run by `node`, whose stage 0 runs on
/// `predecessor`. Its model path names a file, so a peer's load resolves
/// without fetching anything.
fn stage_one_load(
    node: EndpointId,
    predecessor: EndpointId,
    model: &std::path::Path,
) -> StageLoadRequest {
    let mut load = crate::inference::skippy::test_stage_load_request();
    load.stage_id = "stage-1".to_string();
    load.stage_index = 1;
    load.layer_start = 12;
    load.layer_end = 24;
    load.admission = crate::inference::skippy::test_stage_admission(12, 24);
    load.model_path = Some(model.to_string_lossy().into_owned());
    load.projector_path = None;
    load.downstream = None;
    load.topology_stages = vec![
        StageTopologyStageDescriptor {
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            node_id: predecessor,
            layer_start: 0,
            layer_end: 12,
            bind_addr: String::new(),
        },
        StageTopologyStageDescriptor {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            node_id: node,
            layer_start: 12,
            layer_end: 24,
            bind_addr: String::new(),
        },
    ];
    load
}

fn open_stage_one(requester: EndpointId) -> skippy_protocol::proto::stage::StageTransportOpen {
    skippy_protocol::proto::stage::StageTransportOpen {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: requester.as_bytes().to_vec(),
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        stage_id: "stage-1".to_string(),
    }
}

/// The topologies and statuses in `node`'s registry, sorted for comparison.
async fn registry(
    node: &Node,
) -> (
    Vec<(String, StageTopologyInstance)>,
    Vec<(String, StageRuntimeStatus)>,
) {
    let state = node.stage_topologies.lock().await;
    let mut topologies: Vec<_> = state.topologies.clone().into_iter().collect();
    topologies.sort_by(|left, right| left.0.cmp(&right.0));
    let mut statuses: Vec<_> = state.statuses.clone().into_iter().collect();
    statuses.sort_by(|left, right| left.0.cmp(&right.0));
    (topologies, statuses)
}

/// A worker node whose registry holds `running` as its accepted load left
/// it, and a model file its loads can name.
struct RunningStageNode {
    node: Node,
    running: StageLoadRequest,
    predecessor: EndpointId,
    _model: tempfile::NamedTempFile,
}

async fn node_running_a_bound_stage() -> Result<RunningStageNode> {
    let model = tempfile::NamedTempFile::new()?;
    let mut node = make_test_node(crate::mesh::NodeRole::Worker).await?;
    node.peer_stage_source_fetch = StageSourceFetch::LocalOnly;
    let predecessor = make_test_endpoint_id(0x60);
    let mut running = stage_one_load(node.endpoint.id(), predecessor, model.path());
    running.coordinator_id = Some(make_test_endpoint_id(0x70));
    running.coordinator_term = 11;
    node.record_stage_load_topology(&running).await;
    node.record_stage_status(
        Some(node.endpoint.id()),
        stage_status_from_load(&running, StageRuntimeState::Ready),
    )
    .await;
    Ok(RunningStageNode {
        node,
        running,
        predecessor,
        _model: model,
    })
}

impl RunningStageNode {
    /// An unbound load of the same stage whose stage 0 runs on `usurper`.
    fn takeover(&self, usurper: EndpointId) -> StageLoadRequest {
        let mut takeover = self.running.clone();
        takeover.coordinator_id = None;
        takeover.coordinator_term = 0;
        takeover.topology_stages[0].node_id = usurper;
        takeover
    }

    async fn assert_routes_to_the_running_stage(&self, usurper: EndpointId) {
        let node = &self.node;
        assert!(
            node.stage_transport_allowed(self.predecessor, &open_stage_one(self.predecessor))
                .await,
            "the running stage's predecessor must keep its transport edge"
        );
        assert!(
            !node
                .stage_transport_allowed(usurper, &open_stage_one(usurper))
                .await,
            "a refused load must not grant its stage 0 a transport edge"
        );
    }
}

fn refused(response: &StageControlResponse) -> &StageReadyResponse {
    match response {
        StageControlResponse::Ready(ready) if !ready.accepted => ready,
        other => panic!("the load must be refused, got {other:?}"),
    }
}

#[tokio::test]
async fn refused_takeover_leaves_the_running_stage_and_its_routing() -> Result<()> {
    let setup = node_running_a_bound_stage().await?;
    let control = crate::inference::skippy::spawn_stage_control_loop_with_placeholder_stage(
        setup.running.clone(),
    );
    *setup.node.stage_control_tx.lock().await = Some(control.sender());
    let before = registry(&setup.node).await;
    let attacker = make_test_endpoint_id(0xa1);

    let response = setup
        .node
        .authorize_resolve_and_execute_stage_control(
            attacker,
            "load",
            StageControlRequest::Load(setup.takeover(attacker)),
        )
        .await?;

    let ready = refused(&response);
    assert!(!ready.removed_running_stage);
    assert!(
        ready
            .error
            .as_deref()
            .is_some_and(|error| error.contains("cannot replace a coordinator-bound stage")),
        "{:?}",
        ready.error
    );
    assert_eq!(registry(&setup.node).await, before);
    setup.assert_routes_to_the_running_stage(attacker).await;
    let statuses = setup
        .node
        .query_local_stage_status(StageStatusFilter {
            topology_id: Some("topology-a".to_string()),
            run_id: Some("run-a".to_string()),
            stage_id: Some("stage-1".to_string()),
        })
        .await?;
    assert_eq!(statuses.len(), 1, "the bound stage must keep running");
    control.shutdown().await?;
    Ok(())
}

/// A stage control loop that passes every load's preflight, refuses the load
/// itself, and reports that `running` still runs.
fn spawn_refusing_stage_control_loop(
    running: StageLoadRequest,
) -> tokio::sync::mpsc::UnboundedSender<StageControlCommand> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(command) = rx.recv().await {
            match command {
                StageControlCommand::ValidateLoad { resp, .. } => {
                    let _ = resp.send(None);
                }
                StageControlCommand::Execute {
                    request: StageControlRequest::Load(load),
                    resp,
                    ..
                } => {
                    let mut status = stage_status_from_load(&load, StageRuntimeState::Failed);
                    status.error = Some("refused after preflight".to_string());
                    let _ = resp.send(Ok(StageControlResponse::Ready(StageReadyResponse {
                        accepted: false,
                        removed_running_stage: false,
                        status,
                        error: Some("refused after preflight".to_string()),
                    })));
                }
                StageControlCommand::Execute {
                    request: StageControlRequest::Status(_),
                    resp,
                    ..
                } => {
                    let status = stage_status_from_load(&running, StageRuntimeState::Ready);
                    let _ = resp.send(Ok(StageControlResponse::Status(vec![status])));
                }
                StageControlCommand::Execute { resp, .. } => {
                    let _ = resp.send(Err(anyhow::anyhow!("unexpected stage control request")));
                }
            }
        }
    });
    tx
}

#[tokio::test]
async fn load_refused_after_its_topology_was_recorded_puts_the_registry_back() -> Result<()> {
    let setup = node_running_a_bound_stage().await?;
    *setup.node.stage_control_tx.lock().await =
        Some(spawn_refusing_stage_control_loop(setup.running.clone()));
    let before = registry(&setup.node).await;
    let usurper = make_test_endpoint_id(0xa2);

    let response = setup
        .node
        .authorize_resolve_and_execute_stage_control(
            usurper,
            "load",
            StageControlRequest::Load(setup.takeover(usurper)),
        )
        .await?;

    refused(&response);
    assert_eq!(registry(&setup.node).await, before);
    setup.assert_routes_to_the_running_stage(usurper).await;
    Ok(())
}

#[tokio::test]
async fn refused_load_with_no_stage_in_its_place_is_reported_failed() -> Result<()> {
    let model = tempfile::NamedTempFile::new()?;
    let mut node = make_test_node(crate::mesh::NodeRole::Worker).await?;
    node.peer_stage_source_fetch = StageSourceFetch::LocalOnly;
    let control = crate::inference::skippy::spawn_stage_control_loop(Default::default());
    *node.stage_control_tx.lock().await = Some(control.sender());
    let requester = make_test_endpoint_id(0xa3);
    let mut load = stage_one_load(node.endpoint.id(), requester, model.path());
    load.backend = "not-skippy".to_string();

    let response = node
        .authorize_resolve_and_execute_stage_control(
            requester,
            "load",
            StageControlRequest::Load(load),
        )
        .await?;

    refused(&response);
    let (topologies, statuses) = registry(&node).await;
    assert!(topologies.is_empty(), "{topologies:?}");
    assert_eq!(statuses.len(), 1, "{statuses:?}");
    assert_eq!(statuses[0].1.stage_id, "stage-1");
    assert_eq!(statuses[0].1.state, StageRuntimeState::Failed);
    assert!(
        !node
            .stage_transport_allowed(requester, &open_stage_one(requester))
            .await
    );
    control.shutdown().await?;
    Ok(())
}
