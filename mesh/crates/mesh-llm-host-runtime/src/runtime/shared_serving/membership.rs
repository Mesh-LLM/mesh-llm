//! An independent QUIC identity and local ingress; it never owns model weights.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use skippy_inference_api::{InferenceBackend, router_for};
use tokio::sync::watch;

use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh::{
    Node, NodeRole, NodeStartupSpec, QuicBindSelection, RelayConfig, RelayPolicy,
    ScopedNodeIdentity, ServedModelDescriptor,
};
use crate::network::tunnel::start_serving_attachment;

use super::admission::{Admission, AdmissionStatus, MemberId};
use super::backend::MembershipBackend;
use super::configuration::{ServingMembershipConfig, private_directory, read_join_token};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MembershipPhase {
    Joining,
    Connected,
    Paused,
    Detached,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
pub struct MembershipStatus {
    pub name: String,
    pub phase: MembershipPhase,
    pub endpoint_id: Option<String>,
    pub mesh_id: Option<String>,
    pub peers: usize,
    pub admission: AdmissionStatus,
    pub last_error: Option<String>,
}

pub(super) struct Membership {
    pub node: Node,
    pub member: MemberId,
    admission: Arc<Admission>,
    targets: watch::Sender<ModelTargets>,
    descriptor: ServedModelDescriptor,
    http_port: u16,
    http_shutdown: skippy_inference_api::CancellationToken,
    http_task: tokio::task::JoinHandle<()>,
    maintenance: tokio::task::JoinHandle<()>,
    ingress: tokio::task::JoinHandle<()>,
}

pub(super) struct AttachmentInputs<'a> {
    pub config: &'a ServingMembershipConfig,
    pub state_dir: &'a std::path::Path,
    pub backend: Arc<dyn InferenceBackend>,
    pub descriptor: ServedModelDescriptor,
    pub admission: Arc<Admission>,
    pub queue_capacity: usize,
    pub lan_only: bool,
    pub bind_ip: Option<std::net::IpAddr>,
    pub ingress_slots: Arc<tokio::sync::Semaphore>,
    pub ingress_capacity: usize,
    pub release_attestation: Result<crate::runtime::release_attestation::LoadedReleaseAttestation>,
}

impl Membership {
    pub(super) async fn attach(inputs: AttachmentInputs<'_>) -> Result<Self> {
        let token = read_join_token(&inputs.config.token_file)?;
        let directory = inputs.state_dir.join(&inputs.config.name);
        private_directory(&directory)?;
        let key_path = directory.join("node.key");
        let key = if key_path.exists() {
            mesh_llm_membership::load_node_key_from_path(&key_path)?
        } else {
            let key = iroh::SecretKey::generate();
            mesh_llm_membership::save_node_key_to_path(&key_path, &key)?;
            key
        };
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
        let port = listener.local_addr()?.port();
        let config_path = directory.join("config.toml");
        let owner = owner_config(inputs.config)?;
        let (node, channels) = Node::start_with_spec(NodeStartupSpec {
            role: NodeRole::Host { http_port: port },
            relay: RelayConfig {
                urls: &[],
                auths: &HashMap::new(),
                policy: if inputs.lan_only {
                    RelayPolicy::Disabled
                } else {
                    RelayPolicy::DefaultPublic
                },
            },
            quic_bind: QuicBindSelection {
                ip: inputs.bind_ip,
                port: None,
            },
            max_vram_gb: Some(0.0),
            enumerate_host: false,
            owner_config: Some(owner),
            config_path: Some(&config_path),
            local_mesh_requirements: Default::default(),
            identity: Some(ScopedNodeIdentity {
                key,
                state_dir: directory,
            }),
        })
        .await?;
        crate::runtime::release_attestation::attach_to_node(&node, inputs.release_attestation)
            .await;
        let member = inputs.admission.register(inputs.queue_capacity);
        let backend = Arc::new(MembershipBackend::new(
            inputs.backend,
            inputs.admission.clone(),
            member,
        ));
        let http_shutdown = skippy_inference_api::CancellationToken::new();
        let listener = super::local_http::AttachmentListener::new(listener, http_shutdown.clone());
        let stop_http = http_shutdown.clone();
        let http_task = tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, router_for(backend))
                .with_graceful_shutdown(async move { stop_http.cancelled().await })
                .await
            {
                tracing::error!(%error, "shared membership HTTP ingress stopped");
            }
        });
        let (targets, rx) = watch::channel(ModelTargets::default());
        let ingress = start_serving_attachment(
            node.clone(),
            channels,
            rx,
            inputs.ingress_slots,
            inputs.ingress_capacity,
        );
        let maintenance = membership_maintenance(node.clone());
        let attachment = Self {
            node,
            member,
            admission: inputs.admission,
            targets,
            descriptor: inputs.descriptor,
            http_port: port,
            http_shutdown,
            http_task,
            maintenance,
            ingress,
        };
        attachment.node.start_accepting();
        let joined =
            tokio::time::timeout(Duration::from_secs(30), attachment.node.join(&token)).await;
        match joined {
            Ok(Ok(())) => Ok(attachment),
            result => {
                attachment.close().await;
                match result {
                    Ok(Err(error)) => Err(error).context("membership join rejected"),
                    Err(_) => anyhow::bail!("membership join timed out"),
                    Ok(Ok(())) => unreachable!(),
                }
            }
        }
    }

    pub(super) async fn advertise(&self, ready: bool) {
        let model = self.descriptor.identity.model_name.clone();
        let models = if ready {
            vec![model.clone()]
        } else {
            Vec::new()
        };
        let mut targets = ModelTargets::default();
        if ready {
            targets
                .targets
                .insert(model, vec![InferenceTarget::Local(self.http_port)]);
        }
        self.targets.send_replace(targets);
        self.node.set_models(models.clone()).await;
        self.node.set_hosted_models(models.clone()).await;
        self.node.set_serving_models(models).await;
        self.node
            .set_served_model_descriptors(if ready {
                vec![self.descriptor.clone()]
            } else {
                Vec::new()
            })
            .await;
        self.node.regossip().await;
    }

    pub(super) async fn pause(&self) {
        self.admission.pause(self.member, false);
        self.advertise(false).await;
    }

    pub(super) async fn resume(&self) {
        self.admission.resume(self.member);
        self.advertise(true).await;
    }

    pub(super) async fn close(self) {
        self.admission.retire(self.member);
        self.http_shutdown.cancel();
        self.advertise(false).await;
        self.node.broadcast_leaving().await;
        self.maintenance.abort();
        self.http_task.abort();
        self.ingress.abort();
        self.node.close_endpoint().await;
    }
}

impl Drop for Membership {
    fn drop(&mut self) {
        self.admission.retire(self.member);
        self.http_shutdown.cancel();
        self.maintenance.abort();
        self.http_task.abort();
        self.ingress.abort();
        let endpoint = self.node.endpoint.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                endpoint.close().await;
            });
        }
    }
}

fn membership_maintenance(node: Node) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut failures = HashMap::new();
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            node.run_heartbeat_cycle(&mut failures).await;
        }
    })
}

fn owner_config(config: &ServingMembershipConfig) -> Result<crate::mesh::OwnerRuntimeConfig> {
    let trust_store = match &config.trust_store {
        Some(path) => super::configuration::read_explicit_trust_store(path)?,
        None => Default::default(),
    };
    let keypair = config
        .owner_key
        .as_deref()
        .map(super::super::startup_models::load_owner_keypair_for_runtime)
        .transpose()?;
    Ok(crate::mesh::OwnerRuntimeConfig {
        keypair,
        control_bind: None,
        control_advertise_addr: None,
        node_label: Some(config.name.clone()),
        trust_policy: trust_store.policy,
        trust_store,
        activity: Default::default(),
        drain_timeout_secs: 30,
        drain_timeout_max_secs: 30,
        public_mesh: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_explicit_trust_store_cannot_downgrade_reconnect_policy() {
        let directory = tempfile::tempdir().unwrap();
        let config = ServingMembershipConfig {
            name: "alpha".into(),
            token_file: directory.path().join("unused.token"),
            trust_store: Some(directory.path().join("missing.trust")),
            owner_key: None,
        };
        assert!(owner_config(&config).is_err());
        let implicit = ServingMembershipConfig {
            trust_store: None,
            ..config
        };
        assert!(matches!(
            owner_config(&implicit).unwrap().trust_policy,
            crate::crypto::TrustPolicy::Off
        ));
    }
}
