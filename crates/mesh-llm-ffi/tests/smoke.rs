use meshllm_ffi::{ClientEvent, EventListener, FfiError, ModelSearchQuery, create_node};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn valid_owner_keypair_hex() -> String {
    mesh_llm_api_server::OwnerKeypair::generate().to_hex()
}

fn valid_node() -> Arc<meshllm_ffi::MeshNodeHandle> {
    create_node(
        valid_owner_keypair_hex(),
        "valid-token".to_string(),
        None,
        None,
        false,
    )
    .expect("create_node should accept valid identity and token")
}

struct MockListener {
    events: Arc<Mutex<Vec<String>>>,
}

impl EventListener for MockListener {
    fn on_event(&self, event: ClientEvent) {
        let name = match &event {
            ClientEvent::Connecting => "Connecting".to_string(),
            ClientEvent::Joined { .. } => "Joined".to_string(),
            ClientEvent::ModelsUpdated { .. } => "ModelsUpdated".to_string(),
            ClientEvent::TokenDelta { .. } => "TokenDelta".to_string(),
            ClientEvent::Completed { .. } => "Completed".to_string(),
            ClientEvent::Failed { .. } => "Failed".to_string(),
            ClientEvent::Disconnected { .. } => "Disconnected".to_string(),
        };
        self.events.lock().unwrap().push(name);
    }
}

struct ReentrantListener {
    handle: Arc<meshllm_ffi::MeshNodeHandle>,
    sender: Mutex<Sender<meshllm_ffi::ClientStatus>>,
}

impl EventListener for ReentrantListener {
    fn on_event(&self, event: ClientEvent) {
        match event {
            ClientEvent::Completed { .. } | ClientEvent::Failed { .. } => {
                let status = self.handle.status();
                let _ = self.sender.lock().unwrap().send(status);
            }
            _ => {}
        }
    }
}

#[test]
fn create_node_with_invalid_token_fails() {
    let result = create_node(valid_owner_keypair_hex(), "".to_string(), None, None, false);
    assert!(matches!(result, Err(FfiError::InvalidInviteToken(_))));
}

#[test]
fn create_node_with_valid_token_succeeds() {
    let result = create_node(
        valid_owner_keypair_hex(),
        "valid-token".to_string(),
        None,
        None,
        false,
    );
    assert!(result.is_ok());
}

#[test]
fn node_handle_status_returns_disconnected() {
    let handle = valid_node();
    let status = handle.status();
    assert!(!status.connected);
    assert_eq!(status.peer_count, 0);
}

#[test]
fn node_model_management_search_and_show_work_without_joining() {
    let handle = valid_node();
    let recommended = handle
        .recommended_models()
        .expect("recommended models should be local");
    assert!(!recommended.is_empty());

    let results = handle
        .search_models(ModelSearchQuery {
            query: recommended[0].name.clone(),
            limit: Some(5),
        })
        .expect("model search should be local");
    assert!(!results.is_empty());

    let details = handle
        .show_model(recommended[0].id.clone())
        .expect("catalog model details should resolve");
    assert_eq!(details.id, recommended[0].id);
    assert_eq!(
        details.capabilities.multimodal,
        recommended[0].capabilities.multimodal
    );
}

#[test]
#[cfg(not(feature = "embedded-runtime"))]
fn node_serving_control_without_controller_is_typed_unsupported() {
    let result = create_node(
        valid_owner_keypair_hex(),
        "valid-token".to_string(),
        None,
        None,
        true,
    );
    assert!(matches!(result, Err(FfiError::ServingUnsupported(_))));
}

#[test]
fn create_node_with_empty_owner_keypair_fails() {
    // Empty keypair is rejected rather than silently generating a fresh identity.
    let result = create_node("".to_string(), "valid-token".to_string(), None, None, false);
    assert!(matches!(result, Err(FfiError::InvalidOwnerKeypair(_))));
}

#[test]
fn create_node_with_invalid_owner_keypair_fails() {
    let result = create_node(
        "deadbeef".to_string(),
        "valid-token".to_string(),
        None,
        None,
        false,
    );
    assert!(matches!(result, Err(FfiError::InvalidOwnerKeypair(_))));
}

#[test]
fn create_node_uses_supplied_owner_keypair() {
    let owner_keypair_hex = {
        let keypair = mesh_llm_api_server::OwnerKeypair::generate();
        keypair.to_hex()
    };

    let handle = create_node(
        owner_keypair_hex,
        "valid-token".to_string(),
        None,
        None,
        false,
    )
    .expect("create_node should succeed with valid inputs");
    let status = handle.status();
    assert!(!status.connected);
    assert_eq!(status.peer_count, 0);
}

#[test]
fn mock_listener_receives_events() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let listener = MockListener {
        events: events.clone(),
    };

    listener.on_event(ClientEvent::Connecting);
    listener.on_event(ClientEvent::Joined {
        node_id: "test-node".to_string(),
    });
    listener.on_event(ClientEvent::ModelsUpdated { models: vec![] });
    listener.on_event(ClientEvent::TokenDelta {
        request_id: "req-1".to_string(),
        delta: "hello".to_string(),
    });
    listener.on_event(ClientEvent::Completed {
        request_id: "req-1".to_string(),
    });
    listener.on_event(ClientEvent::Failed {
        request_id: "req-2".to_string(),
        error: "timeout".to_string(),
    });
    listener.on_event(ClientEvent::Disconnected {
        reason: "network".to_string(),
    });

    let received = events.lock().unwrap();
    assert_eq!(received.len(), 7);
    assert_eq!(received[0], "Connecting");
    assert_eq!(received[1], "Joined");
    assert_eq!(received[2], "ModelsUpdated");
    assert_eq!(received[3], "TokenDelta");
    assert_eq!(received[4], "Completed");
    assert_eq!(received[5], "Failed");
    assert_eq!(received[6], "Disconnected");
}

#[test]
fn handle_create_destroy_loop_25_times() {
    for i in 0..25 {
        let token = format!("invite-token-{}", i);
        let handle = create_node(valid_owner_keypair_hex(), token, None, None, false)
            .expect("create_node should succeed with valid inputs");
        let status = handle.status();
        assert!(!status.connected, "iteration {}: expected disconnected", i);
    }
}

#[test]
fn listener_can_reenter_handle_during_callback() {
    let handle = valid_node();
    let (tx, rx) = mpsc::channel();
    let request_id = handle
        .chat(
            meshllm_ffi::ChatRequestNative {
                model: "test-model".to_string(),
                messages: vec![meshllm_ffi::ChatMessageNative {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                }],
            },
            Box::new(ReentrantListener {
                handle: handle.clone(),
                sender: Mutex::new(tx),
            }),
        )
        .expect("chat should start");

    assert!(!request_id.is_empty(), "chat should return a request id");
    let status = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("callback should be able to reenter handle without deadlocking");
    assert!(!status.connected);
}
