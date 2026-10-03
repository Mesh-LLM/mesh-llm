use meshllm_ffi::{
    ClientEvent, EventListener, FfiError, OpenAiStreamEventNative, OpenAiStreamListener,
    create_node,
};

struct MockListener;

impl EventListener for MockListener {
    fn on_event(&self, _event: ClientEvent) {}
}

struct MockOpenAiStreamListener;

impl OpenAiStreamListener for MockOpenAiStreamListener {
    fn on_event(&self, _event: OpenAiStreamEventNative) {}
}

#[test]
fn node_stream_exports_compile() {
    let _listener: Box<dyn EventListener> = Box::new(MockListener);
    let _openai_listener: Box<dyn OpenAiStreamListener> = Box::new(MockOpenAiStreamListener);
    let result = create_node("deadbeef".to_string(), "".to_string(), None, None, false);
    assert!(matches!(result, Err(FfiError::InvalidInviteToken(_))));
}

#[test]
fn node_exports_compile() {
    let keypair = mesh_llm_sdk::OwnerKeypair::generate().to_hex();
    let result = create_node(keypair, "valid-token".to_string(), None, None, false);
    assert!(result.is_ok());
}
