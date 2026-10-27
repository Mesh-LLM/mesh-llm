//! Plugin contributions to the host's `GET /api/status`.
//!
//! A plugin declares [`plugin_status`] in its manifest and publishes snapshots
//! with [`crate::PluginContext::publish_status`]. The host keeps the latest one
//! and shows it in `GET /api/status` under `plugins.<plugin name>`, inside an
//! envelope the host owns:
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "state": "ready",
//!   "received_at_unix_ms": 1793404800000,
//!   "fresh_until_unix_ms": 1793404860000,
//!   "data": { "source": "macmon", "idle_watts": 8.2 }
//! }
//! ```
//!
//! `state` is `ready` until the snapshot's freshness runs out, then `stale`,
//! and `unavailable` once the plugin connection that published it is gone. The
//! freshness is the host's: it says when the host last heard from the plugin,
//! not how old the plugin's own readings are, which belong in `data`.

use crate::manifest::ManifestEntry;
use crate::proto;
use std::time::Duration;

/// The largest `data_json` a host accepts, in bytes.
pub const MAX_DATA_BYTES: usize = 16 * 1024;
/// The shortest interval between two snapshots a host accepts from one plugin
/// connection. A withdrawal is always accepted.
pub const MIN_PUBLISH_INTERVAL: Duration = Duration::from_secs(1);
/// How long a snapshot stays `ready` when it sets `fresh_for_ms` to 0.
pub const DEFAULT_FRESH_FOR: Duration = Duration::from_secs(60);
/// The shortest freshness a host keeps; a shorter one is raised to it.
pub const MIN_FRESH_FOR: Duration = Duration::from_secs(1);
/// The longest freshness a host keeps; a longer one is lowered to it.
pub const MAX_FRESH_FOR: Duration = Duration::from_secs(60 * 60);

/// Declare that this plugin publishes snapshots in version `schema_version`
/// (at least 1) of its own schema.
pub fn plugin_status(schema_version: u32) -> PluginStatusBuilder {
    PluginStatusBuilder {
        inner: proto::PluginStatusManifest {
            schema_version,
            required: false,
        },
    }
}

#[derive(Clone, Debug)]
pub struct PluginStatusBuilder {
    inner: proto::PluginStatusManifest,
}

impl PluginStatusBuilder {
    /// Refuse to initialize on a host that does not list
    /// [`crate::host_capabilities::PLUGIN_STATUS`].
    pub fn required(mut self, required: bool) -> Self {
        self.inner.required = required;
        self
    }

    pub fn build(self) -> proto::PluginStatusManifest {
        self.inner
    }
}

impl From<PluginStatusBuilder> for ManifestEntry {
    fn from(value: PluginStatusBuilder) -> Self {
        ManifestEntry::PluginStatus(value.build())
    }
}

/// How long the host keeps a snapshot published with `fresh_for_ms` as
/// `ready`.
pub fn effective_fresh_for(fresh_for_ms: u64) -> Duration {
    if fresh_for_ms == 0 {
        return DEFAULT_FRESH_FOR;
    }
    Duration::from_millis(fresh_for_ms).clamp(MIN_FRESH_FOR, MAX_FRESH_FOR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        LocalStream, PROTOCOL_VERSION, PluginMetadata, PluginRuntime, SimplePlugin,
        plugin_server_info, read_envelope, write_envelope,
    };

    #[test]
    fn freshness_defaults_to_a_minute_and_stays_between_a_second_and_an_hour() {
        assert_eq!(effective_fresh_for(0), DEFAULT_FRESH_FOR);
        assert_eq!(effective_fresh_for(1), MIN_FRESH_FOR);
        assert_eq!(effective_fresh_for(5_000), Duration::from_secs(5));
        assert_eq!(effective_fresh_for(u64::MAX), MAX_FRESH_FOR);
    }

    #[test]
    fn the_declaration_lands_in_the_manifest() {
        let manifest = crate::plugin_manifest()
            .item(plugin_status(2).required(true))
            .build();
        assert_eq!(
            manifest.plugin_status,
            Some(proto::PluginStatusManifest {
                schema_version: 2,
                required: true,
            })
        );
    }

    async fn initialize_status_plugin(
        required: bool,
        host_capabilities: Vec<String>,
    ) -> Option<proto::envelope::Payload> {
        let plugin = SimplePlugin::new(PluginMetadata::new(
            "demo",
            "1.0.0",
            plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
        ))
        .with_manifest(
            crate::plugin_manifest()
                .item(crate::plugin_status::plugin_status(1).required(required))
                .build(),
        );
        let (plugin_stream, host_stream) = tokio::io::duplex(64 * 1024);
        let runtime = tokio::spawn(PluginRuntime::run_with_stream(
            plugin,
            LocalStream::Memory(plugin_stream),
        ));
        let mut host_stream = LocalStream::Memory(host_stream);
        write_envelope(
            &mut host_stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "demo".into(),
                request_id: 1,
                payload: Some(proto::envelope::Payload::InitializeRequest(
                    proto::InitializeRequest {
                        host_protocol_version: PROTOCOL_VERSION,
                        host_capabilities,
                        ..Default::default()
                    },
                )),
            },
        )
        .await
        .unwrap();
        let response =
            tokio::time::timeout(Duration::from_secs(1), read_envelope(&mut host_stream))
                .await
                .expect("initialize response")
                .unwrap();
        runtime.abort();
        response.payload
    }

    #[tokio::test]
    async fn a_required_status_contribution_refuses_a_host_without_it() {
        let payload = initialize_status_plugin(true, Vec::new()).await;
        let Some(proto::envelope::Payload::ErrorResponse(error)) = payload else {
            panic!("expected an initialize error, got {payload:?}");
        };
        assert!(
            error.message.contains("plugin_status.v1"),
            "{}",
            error.message
        );

        let payload =
            initialize_status_plugin(true, vec![crate::host_capabilities::PLUGIN_STATUS.into()])
                .await;
        let Some(proto::envelope::Payload::InitializeResponse(init)) = payload else {
            panic!("expected an initialize response, got {payload:?}");
        };
        assert_eq!(
            init.manifest.and_then(|manifest| manifest.plugin_status),
            Some(proto::PluginStatusManifest {
                schema_version: 1,
                required: true,
            })
        );
    }

    #[tokio::test]
    async fn an_optional_status_contribution_initializes_on_an_older_host() {
        let payload = initialize_status_plugin(false, Vec::new()).await;
        assert!(
            matches!(
                payload,
                Some(proto::envelope::Payload::InitializeResponse(_))
            ),
            "{payload:?}"
        );
    }
}
