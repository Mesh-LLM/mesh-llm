//! Nodes of one unrestricted mesh settle on a single mesh ID, the lowest any
//! of them announces, however many of them started their own mesh first.

use super::*;

const LOW: &str = "1111111111111111";
const HIGH: &str = "eeeeeeeeeeeeeeee";

async fn wait_for_mesh_id(node: &Node, expected: &str) {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while node.mesh_id().await.as_deref() != Some(expected) {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("node should converge on mesh ID {expected}"));
}

#[tokio::test]
async fn an_unrestricted_node_moves_to_a_lower_mesh_id_only() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    node.set_mesh_id(HIGH.into()).await;
    node.set_mesh_id(LOW.into()).await;
    assert_eq!(node.mesh_id().await.as_deref(), Some(LOW));
    node.set_mesh_id(HIGH.into()).await;
    assert_eq!(node.mesh_id().await.as_deref(), Some(LOW));
    Ok(())
}

#[tokio::test]
async fn a_membership_keeps_its_mesh_id() -> Result<()> {
    let mut node = make_test_node(super::super::NodeRole::Worker).await?;
    node.is_membership = true;
    node.set_mesh_id(HIGH.into()).await;
    node.set_mesh_id(LOW.into()).await;
    assert_eq!(node.mesh_id().await.as_deref(), Some(HIGH));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_originators_converge_on_the_lower_mesh_id_after_joining() -> Result<()> {
    let low = make_test_node(super::super::NodeRole::Worker).await?;
    let high = make_test_node(super::super::NodeRole::Worker).await?;
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
