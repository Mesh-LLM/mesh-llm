use super::{PluginManager, proto};
use anyhow::{Result, bail};

/// Deliver a host-originated channel message to every loaded plugin that
/// declares `channel` in its manifest — the mirror of
/// [`PluginManager::broadcast_mesh_event`] for `PluginMeshEvent::Channel`, needed
/// because `PluginManager::dispatch_channel_message` is addressed to one
/// already-known `plugin_id` (its normal caller is a mesh-relayed message
/// that already names its target), not a broadcast from inside the host
/// runtime itself.
impl PluginManager {
    pub async fn broadcast_channel_message(
        &self,
        channel: &str,
        content_type: &str,
        body: Vec<u8>,
        correlation_id: &str,
    ) -> Result<()> {
        self.deliver_locally(
            None,
            proto::ChannelMessage {
                channel: channel.to_string(),
                content_type: content_type.to_string(),
                body,
                correlation_id: correlation_id.to_string(),
                ..Default::default()
            },
        )
        .await
    }

    /// Deliver `plugin_id`'s own message on a channel its manifest declares
    /// `local` to every other loaded plugin that declares the channel.
    /// Nothing goes to the mesh. The message carries a plugin origin
    /// ([`local_plugin_origin`]) in `source_peer_id`, so a receiver tells it
    /// from the host's own broadcasts (empty `source_peer_id`) and from a mesh
    /// peer's frame (a peer id); `metadata_json` is left as published.
    pub async fn relay_local_channel_message(
        &self,
        plugin_id: &str,
        message: proto::ChannelMessage,
    ) -> Result<()> {
        self.deliver_locally(
            Some(plugin_id),
            proto::ChannelMessage {
                source_peer_id: local_plugin_origin(plugin_id),
                ..message
            },
        )
        .await
    }

    /// One delivery to each loaded plugin declaring `template.channel`,
    /// skipping `except`; `target_peer_id` is set to the recipient.
    async fn deliver_locally(
        &self,
        except: Option<&str>,
        template: proto::ChannelMessage,
    ) -> Result<()> {
        let mut results = Vec::new();
        for (plugin_id, plugin) in &self.inner.plugins {
            if except == Some(plugin_id.as_str())
                || !self
                    .plugin_declares_mesh_channel(plugin_id, &template.channel)
                    .await
            {
                continue;
            }
            let result = plugin
                .try_send_channel_message(proto::ChannelMessage {
                    target_peer_id: plugin_id.clone(),
                    ..template.clone()
                })
                .await;
            results.push((plugin_id.clone(), result));
        }
        aggregate_broadcast_results(results)
    }

    /// How `plugin_name`'s messages on `channel` may travel, from one manifest
    /// snapshot so a restart cannot change the answer between two lookups. A
    /// plugin without a cached manifest (not yet initialized, or shut down)
    /// declares nothing, so the answer fails closed.
    pub async fn channel_route(&self, plugin_name: &str, channel: &str) -> ChannelRoute {
        match self.manifest_snapshot(plugin_name).await {
            Some(manifest) => manifest_channel_route(&manifest, channel),
            None => ChannelRoute::Undeclared,
        }
    }

    /// Whether an already-running `plugin_name` declares `channel` in its
    /// manifest. Gates delivery paths (`broadcast_channel_message`,
    /// `dispatch_channel_message`, `dispatch_bulk_transfer_message`) that
    /// must never lazily start a stopped plugin just to ask it — see
    /// [`PluginManager::manifest_snapshot`].
    pub async fn plugin_declares_mesh_channel(&self, plugin_name: &str, channel: &str) -> bool {
        self.manifest_snapshot(plugin_name)
            .await
            .is_some_and(|manifest| manifest_declares_mesh_channel(&manifest, channel))
    }

    /// Whether any currently loaded plugin declares `channel` — a cached
    /// manifest lookup per plugin, cheap next to the work a caller typically
    /// gates behind it (canonicalizing and hashing a request body, cloning a
    /// served-model descriptor). Lets a publisher skip that work entirely
    /// when nobody on [`Self::broadcast_channel_message`]'s per-plugin filter
    /// would have kept the message anyway.
    pub async fn any_plugin_declares_mesh_channel(&self, channel: &str) -> bool {
        #[cfg(test)]
        for plugin_id in &self.inner.bridged_plugins {
            if self.plugin_declares_mesh_channel(plugin_id, channel).await {
                return true;
            }
        }
        for plugin_id in self.inner.plugins.keys() {
            if self.plugin_declares_mesh_channel(plugin_id, channel).await {
                return true;
            }
        }
        false
    }
}

/// Turn per-plugin delivery outcomes from [`PluginManager::broadcast_channel_message`]
/// into one aggregate result. A single plugin rejecting or losing its channel
/// must never suppress delivery to the other declaring plugins — this only
/// summarizes failures after every recipient has already been attempted.
fn aggregate_broadcast_results(results: Vec<(String, Result<()>)>) -> Result<()> {
    let attempted = results.len();
    let failures: Vec<String> = results
        .into_iter()
        .filter_map(|(plugin_id, result)| result.err().map(|error| format!("{plugin_id}: {error}")))
        .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        bail!(
            "failed to deliver channel message to {} of {} declaring plugin(s): {}",
            failures.len(),
            attempted,
            failures.join("; ")
        )
    }
}

/// Where a plugin's message on a channel may go, per its manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelRoute {
    /// Declared `local`: only the other plugins on this node.
    Local,
    /// Declared as an ordinary mesh channel.
    Mesh,
    /// Not declared: nowhere.
    Undeclared,
}

fn manifest_channel_route(manifest: &proto::PluginManifest, channel: &str) -> ChannelRoute {
    // A name declared both ways is local: isolation wins over reach.
    if manifest_declares_local_channel(manifest, channel) {
        ChannelRoute::Local
    } else if manifest_declares_mesh_channel(manifest, channel) {
        ChannelRoute::Mesh
    } else {
        ChannelRoute::Undeclared
    }
}

/// The `source_peer_id` of a message one local plugin published to the
/// others. A mesh frame's `source_peer_id` must equal its authenticated
/// sender's endpoint id (hex), so no peer can produce this value, and it is
/// not the empty id of the host's own broadcasts. A receiver must authorize
/// the named plugin before treating the message as more than that plugin's
/// claim.
pub(crate) fn local_plugin_origin(plugin_id: &str) -> String {
    format!("{LOCAL_PLUGIN_ORIGIN_PREFIX}{plugin_id}")
}

const LOCAL_PLUGIN_ORIGIN_PREFIX: &str = "plugin:";

fn manifest_declares_local_channel(manifest: &proto::PluginManifest, channel: &str) -> bool {
    manifest
        .mesh_channels
        .iter()
        .any(|entry| entry.name == channel && entry.local)
}

fn manifest_declares_mesh_channel(manifest: &proto::PluginManifest, channel: &str) -> bool {
    manifest
        .mesh_channels
        .iter()
        .any(|entry| entry.name == channel)
}

#[cfg(test)]
mod tests {
    use super::super::{PluginHostMode, ResolvedPlugins};
    use super::*;
    use mesh_llm_plugin::MeshVisibility;
    use tokio::sync::mpsc;

    fn private_host_mode() -> PluginHostMode {
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        }
    }

    #[test]
    fn manifest_declares_mesh_channel_matches_exact_name_only() {
        let manifest = proto::PluginManifest {
            mesh_channels: vec![proto::MeshChannelManifest {
                name: "openai.exchange.v1".into(),
                local: false,
            }],
            ..proto::PluginManifest::default()
        };

        assert!(manifest_declares_mesh_channel(
            &manifest,
            "openai.exchange.v1"
        ));
        assert!(!manifest_declares_mesh_channel(&manifest, "other.channel"));
    }

    #[test]
    fn a_channel_is_local_only_when_its_declaration_says_so() {
        let manifest = proto::PluginManifest {
            mesh_channels: vec![
                proto::MeshChannelManifest {
                    name: "payment.lifecycle.v1".into(),
                    local: true,
                },
                proto::MeshChannelManifest {
                    name: "pay.ln.v1".into(),
                    local: false,
                },
            ],
            ..proto::PluginManifest::default()
        };
        assert!(manifest_declares_local_channel(
            &manifest,
            "payment.lifecycle.v1"
        ));
        assert!(!manifest_declares_local_channel(&manifest, "pay.ln.v1"));
        assert!(!manifest_declares_local_channel(&manifest, "other.channel"));
        // A local declaration is still a declaration of the channel.
        assert!(manifest_declares_mesh_channel(
            &manifest,
            "payment.lifecycle.v1"
        ));
    }

    #[test]
    fn a_channel_routes_by_how_it_is_declared() {
        let manifest = proto::PluginManifest {
            mesh_channels: vec![
                proto::MeshChannelManifest {
                    name: "local.v1".into(),
                    local: true,
                },
                proto::MeshChannelManifest {
                    name: "mesh.v1".into(),
                    local: false,
                },
                proto::MeshChannelManifest {
                    name: "both.v1".into(),
                    local: false,
                },
                proto::MeshChannelManifest {
                    name: "both.v1".into(),
                    local: true,
                },
            ],
            ..proto::PluginManifest::default()
        };
        assert_eq!(
            manifest_channel_route(&manifest, "local.v1"),
            ChannelRoute::Local
        );
        assert_eq!(
            manifest_channel_route(&manifest, "mesh.v1"),
            ChannelRoute::Mesh
        );
        assert_eq!(
            manifest_channel_route(&manifest, "both.v1"),
            ChannelRoute::Local
        );
        assert_eq!(
            manifest_channel_route(&manifest, "other.v1"),
            ChannelRoute::Undeclared
        );
    }

    #[test]
    fn a_plugin_origin_is_never_a_peer_id_or_empty() {
        let origin = local_plugin_origin("pay-lightning");
        assert_eq!(origin, "plugin:pay-lightning");
        // Peer ids are hex endpoint ids, which never contain ':'.
        assert!(origin.contains(':'));
    }

    /// One declaring plugin failing must not stop delivery to the rest —
    /// every recipient's result is collected before an aggregate error names
    /// only the ones that actually failed. Regression for the coderbot
    /// finding: the old loop used `?` on each send and returned on the first
    /// failure, silently skipping later plugins in the `BTreeMap`.
    #[test]
    fn aggregate_broadcast_results_reports_every_failure_without_dropping_successes() {
        let results = vec![
            ("alpha".to_string(), Ok(())),
            ("beta".to_string(), Err(anyhow::anyhow!("channel closed"))),
            ("gamma".to_string(), Err(anyhow::anyhow!("send timed out"))),
            ("delta".to_string(), Ok(())),
        ];

        let error = aggregate_broadcast_results(results)
            .expect_err("some plugins failed")
            .to_string();

        assert!(error.contains("2 of 4"), "error was: {error}");
        assert!(error.contains("beta: channel closed"), "error was: {error}");
        assert!(
            error.contains("gamma: send timed out"),
            "error was: {error}"
        );
        assert!(!error.contains("alpha"), "error was: {error}");
        assert!(!error.contains("delta"), "error was: {error}");
    }

    #[test]
    fn aggregate_broadcast_results_is_ok_when_every_plugin_succeeds() {
        let results = vec![("alpha".to_string(), Ok(())), ("beta".to_string(), Ok(()))];

        assert!(aggregate_broadcast_results(results).is_ok());
    }

    #[tokio::test]
    async fn broadcast_channel_message_is_a_no_op_with_no_plugins_loaded() {
        let specs = ResolvedPlugins {
            externals: Vec::new(),
            inactive: Vec::new(),
        };
        let (mesh_tx, _mesh_rx) = mpsc::channel(1);
        let manager = PluginManager::start(&specs, private_host_mode(), mesh_tx)
            .await
            .expect("empty plugin set starts cleanly");

        manager
            .broadcast_channel_message(
                "openai.exchange.v1",
                "application/json",
                b"{}".to_vec(),
                "corr-1",
            )
            .await
            .expect("broadcasting with no plugins loaded is a no-op, not an error");

        manager.shutdown().await;
    }

    #[tokio::test]
    async fn an_unknown_plugin_has_no_route_on_any_channel() {
        let specs = ResolvedPlugins {
            externals: Vec::new(),
            inactive: Vec::new(),
        };
        let (mesh_tx, _mesh_rx) = mpsc::channel(1);
        let manager = PluginManager::start(&specs, private_host_mode(), mesh_tx)
            .await
            .expect("empty plugin set starts cleanly");

        assert_eq!(
            manager.channel_route("pay-lightning", "pay.ln.v1").await,
            ChannelRoute::Undeclared
        );

        manager.shutdown().await;
    }

    #[tokio::test]
    async fn any_plugin_declares_mesh_channel_is_false_with_no_plugins_loaded() {
        let specs = ResolvedPlugins {
            externals: Vec::new(),
            inactive: Vec::new(),
        };
        let (mesh_tx, _mesh_rx) = mpsc::channel(1);
        let manager = PluginManager::start(&specs, private_host_mode(), mesh_tx)
            .await
            .expect("empty plugin set starts cleanly");

        assert!(
            !manager
                .any_plugin_declares_mesh_channel("openai.exchange.v1")
                .await
        );

        manager.shutdown().await;
    }
}
