mod binary_kv;
mod binary_messaging;
mod decode_batcher;
pub(crate) mod direct_return;
pub(crate) mod forwarding;
mod kv_eviction;
mod options;
mod preconnect;
mod restore_prefill_decode;
mod socket;
mod stage_execution;
mod wire;

pub(crate) use self::binary_messaging::async_forwarder::{AsyncForwardReceipt, AsyncForwarder};
pub use self::binary_messaging::{
    serve_binary, serve_binary_stage, serve_binary_stage_with_shutdown,
};
pub(crate) use self::decode_batcher::DecodeFrameBatcher;
pub use self::direct_return::PredictionReturnHub;
pub use self::direct_return::PredictionReturnListener;
pub(crate) use self::direct_return::PredictionReturnReceiver;
pub(crate) use self::forwarding::{forwarded_stage_message, forwarded_stage_message_timed};
pub use self::options::{BinaryStageOptions, EmbeddedOpenAiStageOptions, parse_wire_dtype};
pub(crate) use self::stage_execution::{
    BinaryStageExecutionOptions, connect_binary_downstream, run_binary_stage_message,
    send_client_ready_hello_if_enabled, stage_output_activation_capacity,
};
pub use self::wire::WireCondition;
pub(crate) use self::wire::write_stage_message_after_propagation;
pub(crate) use self::wire::write_stage_message_conditioned;
