//! Builtin plugins that run as a task inside the host process.
//!
//! An in-process builtin speaks the unchanged plugin protocol over an
//! in-memory duplex pipe instead of a re-exec'd child. Capability resolution,
//! operation invocation and supervision are identical to an external plugin;
//! only the transport and the absence of a child process differ.

use anyhow::Result;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Serves one plugin connection. Called with the plugin's end of the pipe each
/// time the host (re)starts the plugin; typically
/// `PluginRuntime::run_with_stream(plugin, stream)`.
pub type InProcessPluginRunner = Arc<
    dyn Fn(mesh_llm_plugin::LocalStream) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>
        + Send
        + Sync,
>;

const PIPE_BUFFER_BYTES: usize = 256 * 1024;

/// In-process builtins available to one [`super::PluginManager`], by plugin
/// name. Scoped to the manager (not process-global) so several nodes in one
/// process can each supply their own runner.
#[derive(Clone, Default)]
pub struct InProcessPlugins(Arc<BTreeMap<String, InProcessPluginRunner>>);

impl InProcessPlugins {
    /// Serves the plugin spec named `name` from `runner` when that spec has an
    /// empty command, instead of spawning a process.
    pub fn with(mut self, name: impl Into<String>, runner: InProcessPluginRunner) -> Self {
        Arc::make_mut(&mut self.0).insert(name.into(), runner);
        self
    }

    pub(crate) fn get(&self, name: &str) -> Option<InProcessPluginRunner> {
        self.0.get(name).cloned()
    }
}

/// Starts `runner` on a task and returns the host's end of the pipe.
pub(crate) fn start_in_process(
    name: &str,
    runner: &InProcessPluginRunner,
) -> super::transport::LocalStream {
    let (host, plugin) = tokio::io::duplex(PIPE_BUFFER_BYTES);
    let serve = runner(mesh_llm_plugin::LocalStream::Memory(plugin));
    let name = name.to_string();
    tokio::spawn(async move {
        if let Err(error) = serve.await {
            tracing::warn!(plugin = %name, %error, "in-process plugin exited with error");
        }
    });
    super::transport::LocalStream::Memory(host)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_plugin::{
        InternalRpcPluginBuilder, OperationRouter, PluginMetadata, PluginRuntime,
        operation_with_schema, plugin_server_info,
    };

    fn echo_plugin(name: &str) -> mesh_llm_plugin::InternalRpcPlugin {
        let mut router = OperationRouter::new();
        router.add_json(
            operation_with_schema("echo", "Echo the input.", serde_json::Map::new()),
            |args: serde_json::Value, _context| Box::pin(async move { Ok(args) }),
        );
        InternalRpcPluginBuilder::new(PluginMetadata::new(
            name.to_string(),
            "0.0.0",
            plugin_server_info("echo", "0.0.0", "Echo", "In-process echo", None::<String>),
        ))
        .with_capabilities(vec!["echo.v1".into()])
        .with_manifest(mesh_llm_plugin::plugin_manifest![
            mesh_llm_plugin::capability("echo.v1")
        ])
        .with_operation_router(router)
        .build()
    }

    #[tokio::test]
    async fn in_process_builtin_serves_operations_without_a_child_process() {
        let name = "in-process-echo";
        let runner: InProcessPluginRunner = Arc::new(move |stream| {
            Box::pin(PluginRuntime::run_with_stream(echo_plugin(name), stream))
        });
        let plugin = super::super::runtime::tests::in_process_plugin(name, runner);
        let result = plugin
            .call_tool("echo", r#"{"n":7}"#)
            .await
            .expect("in-process plugin answers");
        assert!(!result.is_error, "{}", result.content_json);
        assert!(result.content_json.contains("7"), "{}", result.content_json);
        let summary = plugin.summary().await;
        assert_eq!(summary.status, "running");
        assert_eq!(summary.pid, None);
        assert!(summary.capabilities.contains(&"echo.v1".to_string()));
    }
}
