use super::*;
#[allow(dead_code)]
#[path = "../../../tests/migration_stability/tls_fixture.rs"]
mod fixture;

#[test]
fn stability_https_dropping_pending_future_joins_owned_transfer_and_closes_tls_peer() {
    let server = fixture::Server::new(vec![fixture::Reply::incomplete()]);
    let curl = Curl::fixture(&server.ca);
    let request = Request {
        endpoint: format!("{}/chat/completions", server.base),
        method: Method::POST,
        body: Some(b"{}".to_vec()),
        stream: false,
        timeout: Duration::from_secs(3),
        token: "fixture",
        started: Instant::now(),
    };
    let mut pending = Box::pin(curl.exchange(request, Cancellation::default()));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            tokio::select! {
                result = &mut pending => panic!("transfer completed before drop: {:?}", result.err()),
                () = tokio::time::sleep(Duration::from_millis(10)) => {
                    if server.requests.lock().unwrap().len() == 1 && server.response_started.load(std::sync::atomic::Ordering::SeqCst) { break; }
                    assert!(Instant::now() < deadline, "TLS fixture request did not arrive");
                }
            }
        }
        let started = Instant::now();
        drop(pending);
        assert!(started.elapsed() < Duration::from_secs(2), "dropped transfer did not complete owned cleanup");
    });
    let closed = server.held_connections_closed.clone();
    drop(server);
    assert!(closed.load(std::sync::atomic::Ordering::SeqCst));
}
