//! Nodes of one unrestricted public mesh settle on a single mesh ID, the
//! lowest any of them announces, however many of them started their own mesh
//! first. Private meshes keep first-heard adoption, so a worker never moves
//! away from the fixed ID of an SDK membership it serves.

use super::*;

const LOW: &str = "1111111111111111";
const HIGH: &str = "eeeeeeeeeeeeeeee";

async fn wait_for_mesh_id(node: &Node, expected: &str) {
    let converged = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while node.mesh_id().await.as_deref() != Some(expected) {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await;
    if converged.is_err() {
        let observed = node.mesh_id().await;
        panic!("node should converge on mesh ID {expected}, still at {observed:?}");
    }
}

/// A node of an unrestricted public mesh. The flag is set before the accept
/// loop takes its copy, so the node converges on both sides of a join.
async fn make_public_test_node() -> Result<Node> {
    make_configured_test_node(
        super::super::NodeRole::Worker,
        crate::MeshRequirements::unrestricted(),
        |node| node.public_mesh = true,
    )
    .await
}

/// A public node that started its own mesh under `mesh_id` and accepts joins.
async fn public_originator(mesh_id: &str) -> Result<Node> {
    let node = make_public_test_node().await?;
    node.set_mesh_id_force(mesh_id.into()).await;
    node.start_accepting();
    Ok(node)
}

#[tokio::test]
async fn an_unrestricted_public_node_moves_to_a_lower_mesh_id_only() -> Result<()> {
    let node = make_public_test_node().await?;
    node.set_mesh_id(HIGH.into()).await;
    node.set_mesh_id(LOW.into()).await;
    assert_eq!(node.mesh_id().await.as_deref(), Some(LOW));
    node.set_mesh_id(HIGH.into()).await;
    assert_eq!(node.mesh_id().await.as_deref(), Some(LOW));
    Ok(())
}

#[tokio::test]
async fn a_private_node_keeps_the_first_mesh_id_it_heard() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    node.set_mesh_id(HIGH.into()).await;
    node.set_mesh_id(LOW.into()).await;
    assert_eq!(node.mesh_id().await.as_deref(), Some(HIGH));
    Ok(())
}

#[tokio::test]
async fn a_membership_keeps_its_mesh_id() -> Result<()> {
    let node = make_configured_test_node(
        super::super::NodeRole::Worker,
        crate::MeshRequirements::unrestricted(),
        |node| {
            node.public_mesh = true;
            node.is_membership = true;
        },
    )
    .await?;
    node.set_mesh_id(HIGH.into()).await;
    node.set_mesh_id(LOW.into()).await;
    assert_eq!(node.mesh_id().await.as_deref(), Some(HIGH));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_joining_node_moves_to_the_lower_mesh_id_it_joins() -> Result<()> {
    let low = public_originator(LOW).await?;
    let high = public_originator(HIGH).await?;

    // First-heard adoption alone would leave each side on its own ID.
    high.join(&low.invite_token().await).await?;
    wait_for_peer(&high, low.id()).await;
    wait_for_peer(&low, high.id()).await;

    wait_for_mesh_id(&high, LOW).await;
    assert_eq!(low.mesh_id().await.as_deref(), Some(LOW));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_accepting_node_moves_to_the_lower_mesh_id_of_a_joiner() -> Result<()> {
    let low = public_originator(LOW).await?;
    let high = public_originator(HIGH).await?;

    low.join(&high.invite_token().await).await?;
    wait_for_peer(&low, high.id()).await;
    wait_for_peer(&high, low.id()).await;

    wait_for_mesh_id(&high, LOW).await;
    assert_eq!(low.mesh_id().await.as_deref(), Some(LOW));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_lowest_mesh_id_reaches_a_node_that_is_not_its_direct_peer() -> Result<()> {
    const MID: &str = "8888888888888888";
    let low = public_originator(LOW).await?;
    let mid = public_originator(MID).await?;
    let high = public_originator(HIGH).await?;

    high.join(&mid.invite_token().await).await?;
    wait_for_peer(&mid, high.id()).await;
    wait_for_mesh_id(&high, MID).await;

    // The lowest ID arrives at the middle node, which is the only direct peer
    // of the far node. The next gossip round, a heartbeat in a running node,
    // carries it on.
    low.join(&mid.invite_token().await).await?;
    wait_for_peer(&mid, low.id()).await;
    wait_for_mesh_id(&mid, LOW).await;
    gossip_once(&mid, &high, "middle node gossips to the far node").await;

    wait_for_mesh_id(&high, LOW).await;
    assert_eq!(low.mesh_id().await.as_deref(), Some(LOW));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_private_worker_stays_on_its_memberships_mesh_id() -> Result<()> {
    let membership = make_configured_test_node(
        super::super::NodeRole::Worker,
        crate::MeshRequirements::unrestricted(),
        |node| node.is_membership = true,
    )
    .await?;
    let worker = make_test_node(super::super::NodeRole::Worker).await?;
    let originator = make_test_node(super::super::NodeRole::Worker).await?;
    membership.set_mesh_id_force(HIGH.into()).await;
    originator.set_mesh_id_force(LOW.into()).await;
    membership.start_accepting();
    worker.start_accepting();
    originator.start_accepting();

    worker.join(&membership.invite_token().await).await?;
    wait_for_peer(&membership, worker.id()).await;
    assert_eq!(worker.mesh_id().await.as_deref(), Some(HIGH));

    // An originator with a lower ID joins the worker. Moving to it would make
    // the membership, whose ID is fixed, refuse the worker's gossip.
    originator.join(&worker.invite_token().await).await?;
    wait_for_peer(&worker, originator.id()).await;
    assert_eq!(worker.mesh_id().await.as_deref(), Some(HIGH));

    // The membership still takes the worker's gossip after it heard the lower ID.
    gossip_once(&worker, &membership, "worker gossips to the membership").await;
    assert_eq!(worker.mesh_id().await.as_deref(), Some(HIGH));
    assert_eq!(membership.mesh_id().await.as_deref(), Some(HIGH));
    assert!(
        membership
            .peers()
            .await
            .iter()
            .any(|peer| peer.id == worker.id()),
        "the membership should keep the worker as a peer"
    );
    Ok(())
}
