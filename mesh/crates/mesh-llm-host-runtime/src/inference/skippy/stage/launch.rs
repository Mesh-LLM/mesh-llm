//! Preparing an admitted stage load for launch: everything that can reject
//! the load after its claim is accepted, run before a stage it replaces is
//! shut down.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use mesh_llm_skippy_adapter::readiness::{materialize_stage_bind_addr, parse_bind_addr};
use skippy_protocol::StageConfig;

use super::{StageLoadRequest, stage_config};
use crate::inference::skippy::StageSourceFetch;
use crate::inference::skippy::materialization::{ResolvedStagePackage, resolve_stage_load_package};

/// A stage load that passed every check that can reject it, ready to start.
pub(super) struct PreparedStageLaunch {
    pub(super) load: StageLoadRequest,
    pub(super) bind_addr: SocketAddr,
    pub(super) package: Option<ResolvedStagePackage>,
    pub(super) config: StageConfig,
}

/// A peer's load under `runtime.peer_model_downloads = false` resolves only
/// what is already on this node (`source_fetch`), and fails here, before a
/// stage it replaces stops, when that is not enough.
pub(super) async fn prepare_stage_launch(
    load: StageLoadRequest,
    source_fetch: StageSourceFetch,
) -> Result<PreparedStageLaunch> {
    let bind_addr = loopback_stage_bind_addr(&load.bind_addr)?;
    let mut effective_load = load;
    crate::inference::skippy::configure_materialized_stage_cache();
    if let Some(plan) = effective_load.dflash.as_mut() {
        crate::inference::skippy::materialize_split_dflash_draft(plan, source_fetch).await;
    }
    settle_final_stage_dflash(&mut effective_load);
    let package_request = effective_load.clone();
    let mut resolved_package = None;
    if let Some(package) = tokio::task::spawn_blocking(move || {
        resolve_stage_load_package_with_borrowed_tensors(&package_request, source_fetch)
    })
    .await
    .context("join resolve stage load package task")??
    {
        effective_load.model_path = Some(package.local_ref.clone());
        effective_load.source_model_bytes = package.source_model_bytes;
        resolved_package = Some(package);
    }
    // Reserve the ephemeral port last, so it is free again for as short a
    // time as possible before the stage binds it.
    let bind_addr = materialize_stage_bind_addr(bind_addr)?;
    effective_load.bind_addr = bind_addr.to_string();
    let config = stage_config(&effective_load, resolved_package.as_ref())?;
    Ok(PreparedStageLaunch {
        load: effective_load,
        bind_addr,
        package: resolved_package,
        config,
    })
}

/// The stage server socket carries no authentication; other nodes reach it
/// only through the stage transport bridge, so it never listens beyond
/// loopback.
pub(super) fn loopback_stage_bind_addr(bind_addr: &str) -> Result<SocketAddr> {
    let bind_addr = parse_bind_addr(bind_addr)?;
    anyhow::ensure!(
        bind_addr.ip().is_loopback(),
        "stage bind_addr {bind_addr} is not a loopback address"
    );
    Ok(bind_addr)
}

/// A final stage with a split DFlash plan serves the draft when this node has
/// it, which replaces native MTP; without it the stage drops the plan (no
/// borrowed tensors, no extra rollback depth) and keeps the native MTP the
/// planner sent for that case. The stage reports which it did in its status.
fn settle_final_stage_dflash(load: &mut StageLoadRequest) {
    if load.downstream.is_some() {
        return;
    }
    match load.dflash.as_ref() {
        Some(plan) if plan.drafts() => load.native_mtp_enabled = false,
        Some(_) => load.dflash = None,
        None => {}
    }
}

/// [`resolve_stage_load_package`] with the target tensors a drafting final
/// stage's DFlash draft borrows added to the admitted closure, so the stage
/// fetches the artifacts that hold them. The admission itself, which the
/// status attests, is unchanged. When the package cannot supply them the
/// stage resolves its admission alone and serves without the draft.
fn resolve_stage_load_package_with_borrowed_tensors(
    load: &StageLoadRequest,
    fetch: StageSourceFetch,
) -> Result<Option<ResolvedStagePackage>> {
    let borrowed = load.dflash.as_ref().and_then(|plan| {
        skippy_serving::split_dflash::borrowed_resident_tensor_ids(
            plan,
            load.downstream.is_none(),
            &load.admission.resident_tensor_ids,
        )
    });
    if let Some(resident_tensor_ids) = borrowed {
        let mut widened = load.clone();
        widened.admission.resident_tensor_ids = resident_tensor_ids;
        match resolve_stage_load_package(&widened, fetch) {
            Ok(package) => return Ok(package),
            Err(error) => tracing::warn!(
                stage = load.stage_id,
                error = format!("{error:#}"),
                "cannot fetch the tensors the split DFlash draft borrows; resolving the stage without them"
            ),
        }
    }
    resolve_stage_load_package(load, fetch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_bind_address_must_be_loopback() {
        for bind_addr in ["127.0.0.1:0", "[::1]:0"] {
            assert!(loopback_stage_bind_addr(bind_addr).is_ok(), "{bind_addr}");
        }
        for bind_addr in ["0.0.0.0:0", "[::]:0", "192.0.2.10:9000"] {
            let error = loopback_stage_bind_addr(bind_addr)
                .expect_err("a non-loopback stage bind address must be rejected")
                .to_string();
            assert!(error.contains("not a loopback address"), "{error}");
        }
    }
}
