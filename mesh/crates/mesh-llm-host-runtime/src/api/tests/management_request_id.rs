use super::*;

const CLIENT_REQUEST_ID: &str = "00000000-0000-4000-8000-000000000041";
const SECOND_CLIENT_REQUEST_ID: &str = "00000000-0000-4000-8000-000000000042";

struct RequestIdObservation {
    response_id: String,
    lifecycle_id: String,
    lifecycle_state: String,
}

async fn observe_management_request_id(request_id_headers: &str) -> RequestIdObservation {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: true,
        application_state_root: Some(temporary_directory.path().join("logging")),
        replay_capacity: 16,
        ..Default::default()
    })
    .await;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let body = r#"{"toml":"version = 1\n","path":"x"}"#;
    let response = send_management_request(
        address,
        format!(
            "POST /api/runtime/config/validate HTTP/1.1\r\nHost: localhost\r\n{request_id_headers}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    server
        .await
        .expect("management task joins")
        .expect("management request succeeds");
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");

    let response_ids = response
        .lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case("x-request-id"))
        .map(|(_, value)| value.trim().to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        response_ids.len(),
        1,
        "response must contain one request ID"
    );
    let response_id = response_ids[0].clone();
    let summary = crate::logging_runtime_state()
        .expect("installed logging runtime")
        .service_for_test()
        .expect("logging service")
        .registry_ref()
        .get_recent(&response_id)
        .expect("response request ID identifies the lifecycle");
    let observation = RequestIdObservation {
        response_id,
        lifecycle_id: summary.request_id,
        lifecycle_state: summary.state,
    };

    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
    observation
}

#[tokio::test]
#[serial]
async fn single_valid_client_request_id_is_the_canonical_management_id() {
    let observation =
        observe_management_request_id(&format!("x-request-id: {CLIENT_REQUEST_ID}\r\n")).await;

    assert_eq!(observation.response_id, CLIENT_REQUEST_ID);
    assert_eq!(observation.lifecycle_id, CLIENT_REQUEST_ID);
    assert_eq!(observation.lifecycle_state, "completed");
}

#[tokio::test]
#[serial]
async fn duplicate_client_request_ids_generate_a_new_canonical_management_id() {
    let observation = observe_management_request_id(&format!(
        "x-request-id: {CLIENT_REQUEST_ID}\r\nX-Request-Id: {SECOND_CLIENT_REQUEST_ID}\r\n"
    ))
    .await;

    assert_ne!(observation.response_id, CLIENT_REQUEST_ID);
    assert_ne!(observation.response_id, SECOND_CLIENT_REQUEST_ID);
    assert_eq!(observation.lifecycle_id, observation.response_id);
    assert!(uuid::Uuid::parse_str(&observation.response_id).is_ok());
}

#[tokio::test]
#[serial]
async fn malformed_client_request_id_generates_a_canonical_management_id() {
    let observation = observe_management_request_id("x-request-id: not-a-uuid\r\n").await;

    assert_ne!(observation.response_id, "not-a-uuid");
    assert_eq!(observation.lifecycle_id, observation.response_id);
    assert!(uuid::Uuid::parse_str(&observation.response_id).is_ok());
}

#[tokio::test]
#[serial]
async fn missing_client_request_id_generates_a_canonical_management_id() {
    let observation = observe_management_request_id("").await;

    assert_eq!(observation.lifecycle_id, observation.response_id);
    assert!(uuid::Uuid::parse_str(&observation.response_id).is_ok());
}
