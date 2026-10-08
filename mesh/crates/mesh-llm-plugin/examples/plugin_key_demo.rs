//! A minimal plugin that asks its host to announce a public key it signs with
//! (`PluginContext::announce_plugin_key`, host capability `plugin_keys.v1`).
//!
//! Build it with `cargo build -p mesh-llm-plugin --example plugin_key_demo`,
//! add it to a node's config:
//!
//! ```toml
//! [[plugin]]
//! name = "key-demo"
//! command = "/path/to/target/debug/examples/plugin_key_demo"
//! ```
//!
//! and read `GET /api/plugin-keys` on a directly-connected peer: the key is
//! listed under this node's endpoint id and the plugin's name.

use mesh_llm_plugin::{PluginMetadata, PluginRuntime, plugin, plugin_server_info};

/// The RFC 8032 test-vector public key: a fixed, valid Ed25519 key, so a
/// reader can recognise it on the peer. A real plugin announces its own key.
const DEMO_PUBLIC_KEY: [u8; 32] = [
    0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07, 0x3a,
    0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07, 0x51, 0x1a,
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let plugin = plugin! {
        metadata: PluginMetadata::new(
            "key-demo",
            env!("CARGO_PKG_VERSION"),
            plugin_server_info(
                "key-demo",
                env!("CARGO_PKG_VERSION"),
                "Plugin key demo",
                "Announces a public key bound to this node",
                None::<String>,
            ),
        ),
        on_initialized: |context| Box::pin(async move {
            // An older host does not list the capability: say so, announce
            // nothing, and keep running.
            if !context.host_supports(mesh_llm_plugin::host_capabilities::PLUGIN_KEYS) {
                eprintln!(
                    "key-demo: plugin keys are not supported by this host (it does not list `{}`); nothing announced",
                    mesh_llm_plugin::host_capabilities::PLUGIN_KEYS
                );
                return Ok(());
            }
            let response = context.announce_plugin_key(DEMO_PUBLIC_KEY.to_vec()).await?;
            eprintln!(
                "key-demo: announced on node {} ({} signature bytes)",
                response.node_id,
                response.binding_signature.len()
            );
            Ok(())
        }),
    };
    PluginRuntime::run(plugin).await
}
