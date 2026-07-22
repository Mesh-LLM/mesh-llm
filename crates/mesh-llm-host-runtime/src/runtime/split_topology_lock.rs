use super::local_package::SplitParticipant;
use crate::inference::skippy;
use crate::mesh;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const SPLIT_TOPOLOGY_LOCK_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LockedSplitStageAssignment {
    pub(super) node_id: iroh::EndpointId,
    pub(super) layer_start: u32,
    pub(super) layer_end: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitTopologyLockFile {
    version: u32,
    model: String,
    manifest_sha256: String,
    stages: Vec<SplitTopologyLockStage>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitTopologyLockStage {
    node: String,
    layer_start: u32,
    layer_end: u32,
}

#[derive(Clone, Debug)]
struct ParticipantIdentity {
    node_id: iroh::EndpointId,
    hostname: Option<String>,
}

pub(super) async fn load_locked_split_assignments(
    path: &Path,
    node: &mesh::Node,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
) -> Result<Vec<LockedSplitStageAssignment>> {
    let file =
        File::open(path).with_context(|| format!("open split topology lock {}", path.display()))?;
    let topology: SplitTopologyLockFile = serde_json::from_reader(file)
        .with_context(|| format!("parse split topology lock {}", path.display()))?;
    validate_lock_identity(&topology, model_ref, package)?;
    let identities = participant_identities(node, participants).await;
    topology
        .stages
        .iter()
        .enumerate()
        .map(|(index, stage)| resolve_stage(index, stage, &identities))
        .collect()
}

fn validate_lock_identity(
    topology: &SplitTopologyLockFile,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
) -> Result<()> {
    anyhow::ensure!(
        topology.version == SPLIT_TOPOLOGY_LOCK_VERSION,
        "unsupported split topology lock version {}; expected {}",
        topology.version,
        SPLIT_TOPOLOGY_LOCK_VERSION
    );
    anyhow::ensure!(
        topology.model == model_ref || topology.model == package.package_ref,
        "split topology lock model {} does not match requested model {} or package {}",
        topology.model,
        model_ref,
        package.package_ref
    );
    anyhow::ensure!(
        topology.manifest_sha256 == package.manifest_sha256,
        "split topology lock manifest {} does not match resolved package manifest {}",
        topology.manifest_sha256,
        package.manifest_sha256
    );
    anyhow::ensure!(
        topology.stages.len() >= super::local::SPLIT_DEFAULT_MIN_PARTICIPANTS,
        "split topology lock requires at least two stages"
    );
    Ok(())
}

async fn participant_identities(
    node: &mesh::Node,
    participants: &[SplitParticipant],
) -> Vec<ParticipantIdentity> {
    let hostnames = node
        .peers()
        .await
        .into_iter()
        .map(|peer| (peer.id, peer.hostname))
        .collect::<HashMap<_, _>>();
    participants
        .iter()
        .map(|participant| ParticipantIdentity {
            node_id: participant.node_id,
            hostname: if participant.node_id == node.id() {
                node.hostname.clone()
            } else {
                hostnames.get(&participant.node_id).cloned().flatten()
            },
        })
        .collect()
}

fn resolve_stage(
    index: usize,
    stage: &SplitTopologyLockStage,
    identities: &[ParticipantIdentity],
) -> Result<LockedSplitStageAssignment> {
    let selector = stage.node.trim();
    anyhow::ensure!(
        !selector.is_empty(),
        "split topology lock stage {index} has an empty node selector"
    );
    let matches = identities
        .iter()
        .filter(|identity| participant_matches(identity, selector))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        matches.len() == 1,
        "split topology lock stage {index} selector {selector:?} matched {} eligible nodes; available: {}",
        matches.len(),
        participant_identity_labels(identities).join(", ")
    );
    Ok(LockedSplitStageAssignment {
        node_id: matches[0].node_id,
        layer_start: stage.layer_start,
        layer_end: stage.layer_end,
    })
}

fn participant_matches(identity: &ParticipantIdentity, selector: &str) -> bool {
    identity.node_id.to_string() == selector
        || identity
            .hostname
            .as_deref()
            .is_some_and(|hostname| normalize_hostname(hostname) == normalize_hostname(selector))
}

fn normalize_hostname(hostname: &str) -> String {
    hostname.trim().trim_end_matches('.').to_ascii_lowercase()
}

fn participant_identity_labels(identities: &[ParticipantIdentity]) -> Vec<String> {
    identities
        .iter()
        .map(|identity| {
            identity.hostname.as_ref().map_or_else(
                || identity.node_id.to_string(),
                |hostname| format!("{hostname}={}", identity.node_id),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(seed: u8) -> iroh::EndpointId {
        iroh::SecretKey::from_bytes(&[seed; 32]).public()
    }

    #[test]
    fn hostname_selector_is_case_insensitive_and_ignores_trailing_dot() {
        let identities = vec![ParticipantIdentity {
            node_id: endpoint(1),
            hostname: Some("micstudio.local".to_string()),
        }];
        let stage = SplitTopologyLockStage {
            node: "MICSTUDIO.LOCAL.".to_string(),
            layer_start: 0,
            layer_end: 12,
        };

        let resolved = resolve_stage(0, &stage, &identities).unwrap();

        assert_eq!(resolved.node_id, endpoint(1));
        assert_eq!((resolved.layer_start, resolved.layer_end), (0, 12));
    }

    #[test]
    fn ambiguous_hostname_selector_is_rejected() {
        let identities = vec![
            ParticipantIdentity {
                node_id: endpoint(1),
                hostname: Some("worker.local".to_string()),
            },
            ParticipantIdentity {
                node_id: endpoint(2),
                hostname: Some("worker.local".to_string()),
            },
        ];
        let stage = SplitTopologyLockStage {
            node: "worker.local".to_string(),
            layer_start: 0,
            layer_end: 12,
        };

        let error = resolve_stage(0, &stage, &identities).unwrap_err();

        assert!(error.to_string().contains("matched 2 eligible nodes"));
    }
}
