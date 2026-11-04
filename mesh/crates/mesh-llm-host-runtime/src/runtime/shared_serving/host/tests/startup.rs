use super::*;
use crate::runtime::shared_serving::control;
use skippy_inference_api::CancellationToken;

#[tokio::test]
/// Preserve total startup failure when draining retained execution leases also fails.
async fn startup_failure_survives_cleanup_error() {
    use tracing::instrument::WithSubscriber;

    let fixture = Fixture::detached().await;
    for member in &fixture.host.config.memberships {
        std::fs::write(&member.token_file, "invalid-invite-token").unwrap();
    }
    let admission = &fixture.host.admission;
    let lease = admission
        .acquire(admission.register(0), CancellationToken::new())
        .await
        .unwrap();
    let log = tempfile::NamedTempFile::new().unwrap();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_ansi(false)
        .with_writer(log.reopen().unwrap())
        .finish();

    tokio::time::pause();
    let started = tokio::time::Instant::now();
    let result = control::serve(
        fixture.host.clone(),
        fixture.host.config.state_dir.join("management.token"),
        0,
        std::future::pending(),
    )
    .with_subscriber(subscriber)
    .await;
    let elapsed = started.elapsed();
    tokio::time::resume();
    drop(lease);
    fixture.close().await;

    let error = result.unwrap_err().to_string();
    assert!(error.contains("no memberships connected (0/2)"), "{error}");
    assert!(
        elapsed >= Duration::from_secs(30),
        "cleanup must be awaited"
    );
    let records: Vec<serde_json::Value> = std::fs::read_to_string(log.path())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let warning = records
        .iter()
        .find(|record| {
            record["fields"]["message"] == "shared serving cleanup failed after serving failure"
        })
        .expect("secondary cleanup failure must remain visible in diagnostics");
    assert_eq!(warning["level"], "WARN");
    let cleanup_error = warning["fields"]["error"].as_str().unwrap();
    assert!(cleanup_error.contains("shared host has not drained"));
    assert!(cleanup_error.contains("deadline has elapsed"));
}

#[tokio::test]
/// A successful shutdown signal must not hide a failure to drain active execution.
async fn orderly_stop_returns_cleanup_error() {
    let fixture = Fixture::detached().await;
    let admission = &fixture.host.admission;
    let lease = admission
        .acquire(admission.register(0), CancellationToken::new())
        .await
        .unwrap();

    tokio::time::pause();
    let result = control::serve(
        fixture.host.clone(),
        fixture.host.config.state_dir.join("management.token"),
        0,
        std::future::ready(Ok(())),
    )
    .await;
    tokio::time::resume();
    drop(lease);
    fixture.close().await;

    let error = result.unwrap_err();
    assert!(error.to_string().contains("shared host has not drained"));
    assert!(
        error
            .downcast_ref::<tokio::time::error::Elapsed>()
            .is_some()
    );
}

#[tokio::test]
/// A total initial join failure terminates foreground serving and releases the loaded host lock.
async fn zero_connected_memberships_fail_startup_and_release_host_ownership() {
    let fixture = Fixture::detached().await;
    let config = fixture.host.config.clone();
    // Fail every initial join after configuration admission, without depending
    // on QUIC startup or handshake timing. Live trust rejection is covered by
    // the membership-local allowlist test; this test owns foreground cleanup.
    for member in &config.memberships {
        std::fs::write(&member.token_file, "invalid-invite-token").unwrap();
    }
    let weak = Arc::downgrade(&fixture.host);
    let Fixture {
        host,
        seeds,
        directory: _directory,
        ..
    } = fixture;
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        control::serve(
            host,
            config.state_dir.join("management.token"),
            0,
            std::future::pending(),
        ),
    )
    .await
    .expect("a failed startup must not wait for a shutdown signal");
    for seed in seeds {
        seed.close_endpoint().await;
    }
    let error = result.unwrap_err().to_string();
    assert!(error.contains("no memberships connected (0/2)"), "{error}");
    assert!(
        weak.upgrade().is_none(),
        "control must release its backend owner"
    );
    lock_state(&config).expect("the failed process must release its state lock");
}

#[tokio::test]
/// A partial failure keeps healthy QUIC inference and operator recovery available until shutdown.
async fn partial_startup_keeps_serving_and_allows_failed_membership_recovery() {
    use tower::ServiceExt;

    let fixture = Fixture::detached().await;
    let alpha_token = fixture.host.config.memberships[0].token_file.clone();
    std::fs::write(&alpha_token, "invalid-replacement-token").unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let serving = control::serve(
        fixture.host.clone(),
        fixture.host.config.state_dir.join("management.token"),
        0,
        async { stopped.await.map_err(anyhow::Error::from) },
    );
    let operator = async {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = fixture.host.status().await;
                if matches!(status.memberships[0].phase, MembershipPhase::Failed)
                    && matches!(status.memberships[1].phase, MembershipPhase::Connected)
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("healthy memberships must join despite a failed sibling");
        let response = fixture.request(1, "beta").await;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        std::fs::write(&alpha_token, fixture.seeds[0].invite_token().await).unwrap();
        let token = std::fs::read_to_string(fixture.host.config.state_dir.join("management.token"))
            .unwrap();
        let response = control::router(fixture.host.clone(), &token)
            .oneshot(
                axum::http::Request::post("/api/shared-serving/memberships/alpha/reconnect")
                    .header(axum::http::header::AUTHORIZATION, format!("Bearer {token}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::NO_CONTENT);
        let response = fixture.request(0, "alpha").await;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        stop.send(()).unwrap();
    };
    let (result, ()) = tokio::time::timeout(Duration::from_secs(15), async {
        tokio::join!(serving, operator)
    })
    .await
    .expect("partial startup and orderly shutdown must finish");
    result.unwrap();
    assert!(
        fixture
            .host
            .status()
            .await
            .memberships
            .iter()
            .all(|member| matches!(member.phase, MembershipPhase::Detached))
    );
    fixture.close().await;
}

#[tokio::test]
/// Initial admission succeeds when every independent mesh joins.
async fn all_memberships_connected_complete_startup() {
    let fixture = Fixture::detached().await;
    fixture.host.connect_all().await.unwrap();
    assert!(
        fixture
            .host
            .status()
            .await
            .memberships
            .iter()
            .all(|member| matches!(member.phase, MembershipPhase::Connected))
    );
    fixture.close().await;
}
