// Regression coverage for #1338.
//
// iroh's `EndpointInner::drop` logs
// `Endpoint dropped without calling `Endpoint::close`. Aborting ungracefully.`
// at ERROR unless the endpoint was closed first, and it then aborts the socket
// instead of flushing the queued connection-close frames — so peers time the
// connections out rather than seeing this node depart. A node owns two
// endpoints (the mesh endpoint and, for a verified owner, the control
// listener), and shutdown has to close both.
//
// The observable invariant these tests pin is iroh's own `is_closed()` flag,
// because that is precisely the condition `EndpointInner::drop` checks before
// logging. Asserting on the log line instead would couple the tests to an
// upstream message string.

#[tokio::test]
async fn close_endpoint_closes_the_mesh_endpoint() -> anyhow::Result<()> {
    let node = Node::new_for_tests(super::NodeRole::Worker).await?;
    assert!(
        !node.endpoint.is_closed(),
        "a freshly bound endpoint should be open"
    );

    node.close_endpoint().await;

    assert!(
        node.endpoint.is_closed(),
        "shutdown must leave the mesh endpoint closed, or iroh aborts it on drop"
    );
    Ok(())
}

#[tokio::test]
async fn close_endpoint_is_idempotent() -> anyhow::Result<()> {
    let node = Node::new_for_tests(super::NodeRole::Worker).await?;

    node.close_endpoint().await;
    // A second call must not re-enter iroh's close path or wait out the
    // shutdown budget; shutdown callers should not have to track whether an
    // earlier stage already closed the endpoint.
    tokio::time::timeout(std::time::Duration::from_secs(1), node.close_endpoint())
        .await
        .expect("closing an already-closed endpoint should return immediately");

    assert!(node.endpoint.is_closed());
    Ok(())
}

#[tokio::test]
async fn accept_loop_exits_once_the_endpoint_is_closed() -> anyhow::Result<()> {
    let node = Node::new_for_tests(super::NodeRole::Worker).await?;
    let accepting = node.clone();
    let accept_loop = tokio::spawn(async move { accepting.accept_loop().await });
    node.start_accepting();

    node.close_endpoint().await;

    // `Endpoint::accept` yields `None` once the endpoint closes, which is the
    // only signal the accept loop has. If closing ever stopped terminating it,
    // the loop would outlive shutdown holding a `Node` clone.
    tokio::time::timeout(std::time::Duration::from_secs(5), accept_loop)
        .await
        .expect("accept loop should exit when the endpoint closes")?;
    Ok(())
}

#[tokio::test]
async fn shutdown_control_listener_closes_the_control_endpoint() -> anyhow::Result<()> {
    let (node, secret_key) = Node::new_for_tests_with_secret(super::NodeRole::Worker).await?;
    *node.owner_summary.lock().await = verified_owner_summary("owner-a");
    node.maybe_start_control_listener(secret_key, None, None)
        .await?;

    let control_endpoint = node
        .control_listener
        .lock()
        .await
        .as_ref()
        .map(|listener| listener.endpoint.clone())
        .expect("a verified owner should start a control listener");
    assert!(!control_endpoint.is_closed());

    node.shutdown_control_listener().await;

    assert!(
        control_endpoint.is_closed(),
        "the owner-control endpoint is a second iroh endpoint and needs the same graceful close"
    );
    Ok(())
}
