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

async fn make_public_test_node() -> Result<Node> {
    let mut node = make_test_node(super::super::NodeRole::Worker).await?;
    node.public_mesh = true;
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
    let mut node = make_public_test_node().await?;
    node.is_membership = true;
    node.set_mesh_id(HIGH.into()).await;
    node.set_mesh_id(LOW.into()).await;
    assert_eq!(node.mesh_id().await.as_deref(), Some(HIGH));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_originators_converge_on_the_lower_mesh_id_after_joining() -> Result<()> {
    let low = make_public_test_node().await?;
    let high = make_public_test_node().await?;
    low.set_mesh_id_force(LOW.into()).await;
    high.set_mesh_id_force(HIGH.into()).await;
    low.start_accepting();
    high.start_accepting();

    // The node with the higher ID joins, so first-heard adoption alone would
    // leave each side on its own ID.
    high.join(&low.invite_token().await).await?;
    wait_for_peer(&high, low.id()).await;
    wait_for_peer(&low, high.id()).await;

    wait_for_mesh_id(&high, LOW).await;
    assert_eq!(low.mesh_id().await.as_deref(), Some(LOW));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_private_worker_stays_on_its_memberships_mesh_id() -> Result<()> {
    let mut membership = make_test_node(super::super::NodeRole::Worker).await?;
    membership.is_membership = true;
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
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

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
