use super::args::Options;
use crate::{
    automation::{daemon_readiness::ports::Ports, private_state::PrivateState},
    process::{ProcessSpec, Value},
};
use std::path::Path;

pub(super) fn spec(
    root: &Path,
    options: &Options,
    endpoint: (&PrivateState, Ports, bool),
) -> ProcessSpec {
    let (state, ports, headless) = endpoint;
    let mut environment = state.environment(&options.native);
    environment.insert("MESH_LLM_EPHEMERAL_KEY".into(), Value::Public("1".into()));
    if let Some(bytes) = options.variant.stack.bytes() {
        environment.insert(
            "MESH_TOKIO_STACK_SIZE".into(),
            Value::Public(bytes.to_string().into()),
        );
    }
    let mut arguments = vec![
        "--log-format".into(),
        "json".into(),
        "serve".into(),
        "--model".into(),
        options.model.clone(),
        "--no-draft".into(),
        "--device".into(),
        options.device.clone(),
        "--ctx-size".into(),
        options
            .context_size
            .unwrap_or(options.variant.model.context_size())
            .to_string(),
        "--port".into(),
        ports.api.to_string(),
        "--console".into(),
        ports.console.to_string(),
        "--bind-port".into(),
        ports.quic.to_string(),
        "--bind-ip".into(),
        "127.0.0.1".into(),
        "--mesh-discovery-mode".into(),
        "mdns".into(),
    ];
    if headless {
        arguments.push("--headless".into());
    }
    if let Some(projector) = &options.projector {
        arguments.push("--mmproj".into());
        arguments.push(projector.to_string_lossy().into_owned());
    }
    ProcessSpec {
        executable: options.binary.clone(),
        arguments: arguments
            .into_iter()
            .map(|value: String| Value::Public(value.into()))
            .collect(),
        cwd: root.to_owned(),
        environment,
    }
}
