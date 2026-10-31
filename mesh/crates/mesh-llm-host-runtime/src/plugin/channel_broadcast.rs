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
    /// `local` to every other loaded plugin that declares the channel, the
    /// same way the host delivers its own broadcasts: empty `source_peer_id`,
    /// so a receiver tells it from a mesh peer's frame. Nothing goes to the
    /// mesh. The sender's id rides in `metadata_json` as `source_plugin`.
    pub async fn relay_local_channel_message(
        &self,
        plugin_id: &str,
        message: proto::ChannelMessage,
    ) -> Result<()> {
        let metadata_json = local_metadata(&message.metadata_json, plugin_id);
        self.deliver_locally(
            Some(plugin_id),
            proto::ChannelMessage {
                source_peer_id: String::new(),
                metadata_json,
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

    /// Whether an already-running `plugin_name` declares `channel` as one it
    /// publishes on locally ([`proto::MeshChannelManifest::local`]).
    pub async fn plugin_declares_local_channel(&self, plugin_name: &str, channel: &str) -> bool {
        self.manifest_snapshot(plugin_name)
            .await
            .is_some_and(|manifest| manifest_declares_local_channel(&manifest, channel))
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

/// The publisher's `metadata_json` with `source_plugin` set by the host: a
/// JSON object keeps its other members, anything else is replaced.
fn local_metadata(published: &str, plugin_id: &str) -> String {
    let mut object = serde_json::from_str::<serde_json::Value>(published)
        .ok()
        .and_then(|value| match value {
            serde_json::Value::Object(object) => Some(object),
            _ => None,
        })
        .unwrap_or_default();
    object.insert("source_plugin".into(), plugin_id.into());
    serde_json::Value::Object(object).to_string()
}

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
    fn local_metadata_keeps_the_publishers_members_and_names_the_publisher() {
        let merged: serde_json::Value = serde_json::from_str(&local_metadata(
            r#"{"trace":"t-1","source_plugin":"forged"}"#,
            "pay-lightning",
        ))
        .unwrap();
        assert_eq!(
            merged,
            serde_json::json!({"trace": "t-1", "source_plugin": "pay-lightning"})
        );
        for not_an_object in ["", "[1]", "not json"] {
            assert_eq!(
                local_metadata(not_an_object, "pay-lightning"),
                r#"{"source_plugin":"pay-lightning"}"#
            );
        }
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
