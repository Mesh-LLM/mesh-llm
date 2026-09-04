use serde::{Deserialize, Serialize};
use skippy_cache::{
    L3ActivitySnapshot, L3EffectiveState, L3EffectiveStatus, L3InventoryEntry, L3StateReason,
    MANIFEST_VERSION, StoreReconciliation, StoreUsage,
};
use tokio::net::TcpStream;

use super::runtime::ensure_loopback_control_caller;
use crate::api::http::{respond_error, respond_json};
use crate::runtime::kv_disk_config::{
    KvDiskConfigSources, node_kv_disk_cache, node_kv_disk_manager,
};

#[derive(Debug, Serialize)]
struct KvCacheConfiguredPayload {
    mode: mesh_llm_config::KvDiskTierMode,
    directory: std::path::PathBuf,
    budget_bytes: Option<u64>,
    minimum_free_bytes: u64,
    sources: KvDiskConfigSources,
}

#[derive(Debug, Serialize)]
struct KvCacheEffectivePayload {
    state: String,
    reason: Option<String>,
    manager: Option<L3EffectiveStatus>,
}

#[derive(Debug, Serialize)]
pub(crate) struct KvCacheStatusPayload {
    version: u32,
    format_version: u32,
    configured: KvCacheConfiguredPayload,
    effective: KvCacheEffectivePayload,
    usage: Option<StoreUsage>,
    activity: Option<L3ActivitySnapshot>,
    reconciliation: Option<StoreReconciliation>,
    inventory: Vec<L3InventoryEntry>,
}

#[derive(Debug, Default, Deserialize)]
struct KvCachePruneRequest {
    target_bytes: Option<u64>,
    model_identity: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct KvCacheClearRequest {
    model_identity: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) enum KvCacheOperation {
    Status,
    Prune {
        target_bytes: Option<u64>,
        model_identity: Option<String>,
    },
    Clear {
        model_identity: Option<String>,
    },
}

#[derive(Debug)]
pub(crate) struct KvCacheOperationResult {
    pub(crate) status_json: Vec<u8>,
    pub(crate) freed_bytes: Option<u64>,
}

pub(super) async fn handle_status(stream: &mut TcpStream) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    match status_payload() {
        Ok(status) => respond_json(stream, 200, &status).await,
        Err(error) => respond_error(stream, 500, &error.to_string()).await,
    }
}

pub(super) async fn handle_prune(stream: &mut TcpStream, body: &str) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request = if body.trim().is_empty() {
        KvCachePruneRequest::default()
    } else {
        match serde_json::from_str::<KvCachePruneRequest>(body) {
            Ok(request) => request,
            Err(error) => return respond_error(stream, 400, &error.to_string()).await,
        }
    };
    if node_kv_disk_manager().is_none() {
        return respond_error(stream, 409, "disk prompt cache is not active").await;
    }
    match execute_operation(KvCacheOperation::Prune {
        target_bytes: request.target_bytes,
        model_identity: request.model_identity,
    })
    .await
    {
        Ok(result) => respond_operation(stream, result).await,
        Err(error) => respond_error(stream, 500, &error.to_string()).await,
    }
}

pub(super) async fn handle_clear(stream: &mut TcpStream, body: &str) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request = if body.trim().is_empty() {
        KvCacheClearRequest::default()
    } else {
        match serde_json::from_str::<KvCacheClearRequest>(body) {
            Ok(request) => request,
            Err(error) => return respond_error(stream, 400, &error.to_string()).await,
        }
    };
    if node_kv_disk_manager().is_none() {
        return respond_error(stream, 409, "disk prompt cache is not active").await;
    }
    match execute_operation(KvCacheOperation::Clear {
        model_identity: request.model_identity,
    })
    .await
    {
        Ok(result) => respond_operation(stream, result).await,
        Err(error) => respond_error(stream, 500, &error.to_string()).await,
    }
}

async fn respond_operation(
    stream: &mut TcpStream,
    result: KvCacheOperationResult,
) -> anyhow::Result<()> {
    let status = serde_json::from_slice::<serde_json::Value>(&result.status_json)?;
    respond_json(
        stream,
        200,
        &serde_json::json!({
            "freed_bytes": result.freed_bytes.unwrap_or(0),
            "status": status,
        }),
    )
    .await
}

pub(crate) async fn execute_operation(
    operation: KvCacheOperation,
) -> anyhow::Result<KvCacheOperationResult> {
    let freed_bytes = match operation {
        KvCacheOperation::Status => None,
        KvCacheOperation::Prune {
            target_bytes,
            model_identity,
        } => {
            let manager = node_kv_disk_manager()
                .ok_or_else(|| anyhow::anyhow!("disk prompt cache is not active"))?;
            let target = target_bytes.unwrap_or_else(|| {
                manager
                    .limits()
                    .budget_bytes
                    .saturating_mul(85)
                    .checked_div(100)
                    .unwrap_or(0)
            });
            Some(
                tokio::task::spawn_blocking(move || match model_identity.as_deref() {
                    Some(identity) if identity.trim().is_empty() => {
                        anyhow::bail!("model_identity must not be empty")
                    }
                    Some(identity) => manager.prune_model_to(identity, target),
                    None => manager.prune_to(target),
                })
                .await??,
            )
        }
        KvCacheOperation::Clear { model_identity } => {
            let manager = node_kv_disk_manager()
                .ok_or_else(|| anyhow::anyhow!("disk prompt cache is not active"))?;
            Some(
                tokio::task::spawn_blocking(move || match model_identity.as_deref() {
                    Some(identity) if identity.trim().is_empty() => {
                        anyhow::bail!("model_identity must not be empty")
                    }
                    Some(identity) => manager.clear_model(identity),
                    None => manager.clear(),
                })
                .await??,
            )
        }
    };
    Ok(KvCacheOperationResult {
        status_json: serde_json::to_vec(&status_payload()?)?,
        freed_bytes,
    })
}

fn status_payload() -> anyhow::Result<KvCacheStatusPayload> {
    let cache =
        node_kv_disk_cache().ok_or_else(|| anyhow::anyhow!("disk cache not initialized"))?;
    let configured = cache.configured;
    let (effective, usage, activity, reconciliation, inventory) = match cache.manager {
        Some(manager) => {
            let manager_status = manager.effective_status();
            (
                KvCacheEffectivePayload {
                    state: match manager_status.state {
                        L3EffectiveState::Active => "active",
                        L3EffectiveState::ReadOnlyLowSpace => "read_only_low_space",
                        L3EffectiveState::Degraded => "degraded",
                    }
                    .to_string(),
                    reason: manager_status.reason.map(|reason| {
                        match reason {
                            L3StateReason::ReadOnlyLowSpace => "read_only_low_space",
                            L3StateReason::InsufficientSpace => "insufficient_space",
                            L3StateReason::StorageError => "storage_error",
                        }
                        .to_string()
                    }),
                    manager: Some(manager_status),
                },
                Some(manager.usage()?),
                Some(manager.activity()?),
                Some(manager.reconciliation()),
                manager.inventory()?,
            )
        }
        None if !configured.enabled() => (
            KvCacheEffectivePayload {
                state: "off".to_string(),
                reason: None,
                manager: None,
            },
            None,
            None,
            None,
            Vec::new(),
        ),
        None => (
            KvCacheEffectivePayload {
                state: "degraded".to_string(),
                reason: Some(if configured.budget_bytes == Some(0) {
                    "budget_below_entry_floor".to_string()
                } else {
                    "storage_unavailable".to_string()
                }),
                manager: None,
            },
            None,
            None,
            None,
            Vec::new(),
        ),
    };
    Ok(KvCacheStatusPayload {
        version: 1,
        format_version: MANIFEST_VERSION,
        configured: KvCacheConfiguredPayload {
            mode: configured.mode,
            directory: configured.directory,
            budget_bytes: configured.budget_bytes,
            minimum_free_bytes: configured.minimum_free_bytes,
            sources: configured.sources,
        },
        effective,
        usage,
        activity,
        reconciliation,
        inventory,
    })
}
