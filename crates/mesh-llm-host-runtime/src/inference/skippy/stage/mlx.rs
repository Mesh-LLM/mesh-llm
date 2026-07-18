use std::{
    net::SocketAddr,
    sync::{Arc, atomic::AtomicBool},
};

use anyhow::{Context, Result, bail, ensure};
use model_hf::safetensors_stage::{SafetensorsStageMaterializer, SafetensorsStageRequest};
use skippy_engine_mlx::{
    MlxComputeDtype, MlxDerivationControl, MlxDerivedStageCacheConfig, MlxDerivedStageCacheResult,
    MlxStageEngine, MlxStageEngineConfig, MlxWeightQuantization, derive_quantized_stage_cached,
    load_prepared_quantized_stage, mlx_derived_stage_cache_root,
};
use skippy_protocol::LoadMode;
use skippy_protocol::binary::WireActivationDType;
use skippy_server::{
    EmbeddedServerHandle, EmbeddedState, engine_transport::EngineStageServerOptions,
};

use super::{
    MlxStageArtifact, RunningStage, StageControlState, StageLoadRequest, StageReadyResponse,
    StageStatusFilter, StageWeightQuantization, StageWireDType, stage_load_failure_context,
};

const HF_MODEL_PREFIX: &str = "hf-model://";
const DERIVED_SHARD_SIZE_BYTES: usize = 256 * 1024 * 1024;

pub(super) struct MlxStageLaunch {
    pub(super) load: StageLoadRequest,
    pub(super) server: EmbeddedServerHandle,
    pub(super) artifact: MlxDerivedStageCacheResult,
}

pub(super) async fn load_stage(
    state: &mut StageControlState,
    key: String,
    load: StageLoadRequest,
    bind_addr: SocketAddr,
) -> Result<StageReadyResponse> {
    let launch = launch_stage(load, bind_addr).await?;
    if let Err(error) = wait_for_engine_stage_ready(&launch.server, bind_addr).await {
        let last_error = launch.server.status().last_error;
        let context = stage_load_failure_context(
            &launch.load,
            "MLX engine stage did not become ready",
            last_error.as_deref(),
        );
        let _ = launch.server.shutdown().await;
        return Err(error.context(context));
    }
    let effective_load = launch.load;
    let artifact = MlxStageArtifact {
        path: launch.artifact.output_dir,
    };
    state.stages.insert(
        key,
        RunningStage {
            load: effective_load.clone(),
            server: launch.server,
            materialized: None,
            mlx_artifact: Some(artifact),
            package: None,
            _materialized_pin: None,
        },
    );
    let status = state
        .statuses(&StageStatusFilter {
            topology_id: Some(effective_load.topology_id),
            run_id: Some(effective_load.run_id),
            stage_id: Some(effective_load.stage_id),
        })
        .into_iter()
        .next()
        .context("MLX stage status missing after load")?;
    Ok(StageReadyResponse {
        accepted: true,
        status,
        error: None,
    })
}

async fn wait_for_engine_stage_ready(
    server: &EmbeddedServerHandle,
    bind_addr: SocketAddr,
) -> Result<()> {
    const STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
    let deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
    loop {
        let status = server.status();
        match status.state {
            EmbeddedState::Failed => bail!(
                "MLX engine stage startup failed: {}",
                status.last_error.as_deref().unwrap_or("unknown error")
            ),
            EmbeddedState::Stopped => bail!("MLX engine stage stopped during startup"),
            EmbeddedState::Ready => {
                let ready = tokio::task::spawn_blocking(move || {
                    super::probe_binary_stage_ready(
                        bind_addr,
                        std::time::Duration::from_millis(500),
                    )
                })
                .await
                .context("join MLX engine stage readiness probe")?;
                if ready.is_ok() {
                    return Ok(());
                }
            }
            EmbeddedState::Starting | EmbeddedState::Stopping => {}
        }
        if tokio::time::Instant::now() >= deadline {
            bail!(
                "MLX engine stage did not become ready at {bind_addr} within {STARTUP_TIMEOUT:?}"
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

pub(super) async fn prepare_stage(
    load: &StageLoadRequest,
    cancelled: Arc<AtomicBool>,
) -> Result<MlxDerivedStageCacheResult> {
    let load = load.clone();
    tokio::task::spawn_blocking(move || derive_stage_blocking(&load, Some(cancelled)))
        .await
        .context("join MLX derived stage preparation")?
}

pub(super) async fn launch_stage(
    mut load: StageLoadRequest,
    bind_addr: SocketAddr,
) -> Result<MlxStageLaunch> {
    let blocking_load = load.clone();
    let (artifact, engine) = tokio::task::spawn_blocking(move || {
        let artifact = resolve_stage_blocking(&blocking_load, None, false)?;
        let engine = Arc::new(MlxStageEngine::spawn(MlxStageEngineConfig {
            model_dir: artifact.output_dir.clone(),
            model_id: blocking_load.model_id.clone(),
            stage_index: blocking_load.stage_index,
            layer_start: blocking_load.layer_start,
            layer_end: blocking_load.layer_end,
            compute_dtype: MlxComputeDtype::Bf16,
            weight_quantization: None,
            ctx_size: Some(blocking_load.ctx_size),
        })?);
        ensure!(
            engine.stage_info().activation_width == blocking_load.activation_width.max(0) as u32,
            "MLX stage activation width {} does not match requested {}",
            engine.stage_info().activation_width,
            blocking_load.activation_width
        );
        Ok::<_, anyhow::Error>((artifact, engine))
    })
    .await
    .context("join MLX stage load task")??;

    load.bind_addr = bind_addr.to_string();
    load.model_path = Some(artifact.output_dir.to_string_lossy().into_owned());
    let server = skippy_server::start_stage_engine(
        engine,
        EngineStageServerOptions {
            bind_addr,
            downstream_addr: downstream_addr(&load)?,
            wire_dtype: wire_dtype(load.wire_dtype)?,
        },
    );
    Ok(MlxStageLaunch {
        load,
        server,
        artifact,
    })
}

fn derive_stage_blocking(
    load: &StageLoadRequest,
    cancelled: Option<Arc<AtomicBool>>,
) -> Result<MlxDerivedStageCacheResult> {
    resolve_stage_blocking(load, cancelled, true)
}

fn resolve_stage_blocking(
    load: &StageLoadRequest,
    cancelled: Option<Arc<AtomicBool>>,
    build_on_miss: bool,
) -> Result<MlxDerivedStageCacheResult> {
    let request = request_from_load(load)?;
    let materializer = SafetensorsStageMaterializer::from_environment()?;
    let config = MlxDerivedStageCacheConfig {
        source: request,
        cache_root: mlx_derived_stage_cache_root(),
        quantization: mlx_weight_quantization(load.weight_quantization),
        control: MlxDerivationControl::new(Some(load.manifest_sha256.clone()), cancelled),
        shard_size_bytes: DERIVED_SHARD_SIZE_BYTES,
    };
    let artifact = if build_on_miss {
        derive_quantized_stage_cached(&materializer, &config)?
    } else {
        load_prepared_quantized_stage(&materializer, &config)?
    };
    ensure!(
        artifact.report.checkpoint_sha256 == load.manifest_sha256,
        "MLX checkpoint identity {} does not match stage claim {}",
        artifact.report.checkpoint_sha256,
        load.manifest_sha256
    );
    tracing::info!(
        cache_hit = artifact.cache_hit,
        source_range_request_count = artifact.source_range_request_count,
        derivation_recipe_sha256 = %artifact.report.derivation_recipe_sha256,
        stage_id = %load.stage_id,
        "MLX derived stage cache resolved"
    );
    Ok(artifact)
}

fn mlx_weight_quantization(quantization: StageWeightQuantization) -> MlxWeightQuantization {
    match quantization {
        StageWeightQuantization::Auto | StageWeightQuantization::Affine4 => {
            MlxWeightQuantization::Affine {
                group_size: 64,
                bits: 4,
            }
        }
        StageWeightQuantization::Affine8 => MlxWeightQuantization::Affine {
            group_size: 64,
            bits: 8,
        },
        StageWeightQuantization::MxFp4 => MlxWeightQuantization::MxFp4,
    }
}

fn request_from_load(load: &StageLoadRequest) -> Result<SafetensorsStageRequest> {
    validate_load_settings(load)?;
    ensure!(load.backend == "mlx", "MLX stage requires backend=mlx");
    ensure!(
        load.load_mode == LoadMode::ArtifactSlice,
        "MLX SafeTensors stages require load_mode=artifact-slice"
    );
    ensure!(
        load.lane_count == 1,
        "MLX reduced stage transport currently supports lane_count=1"
    );
    let model_ref = load
        .package_ref
        .strip_prefix(HF_MODEL_PREFIX)
        .context("MLX stage package_ref must be hf-model://org/repo@commit")?;
    let model_ref = model_ref::parse_model_ref(model_ref).context("parse MLX HF model ref")?;
    ensure!(
        model_ref.selector.is_none(),
        "MLX stage HF model ref must not contain a selector"
    );
    let revision = model_ref
        .revision
        .context("MLX stage HF model ref requires an immutable commit revision")?;
    ensure!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "MLX stage HF model ref revision must be a 40-character commit SHA"
    );
    Ok(SafetensorsStageRequest {
        repo: model_ref.repo,
        revision,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        include_prefixes: Vec::new(),
    })
}

fn validate_load_settings(load: &StageLoadRequest) -> Result<()> {
    ensure!(load.ctx_size > 0, "MLX stage ctx_size must be positive");
    ensure!(
        load.wire_dtype != StageWireDType::Q8,
        "MLX stages do not support Q8 activation wire dtype"
    );
    ensure!(
        load.n_batch.is_none() && load.n_ubatch.is_none(),
        "MLX reduced stage transport does not support batch overrides"
    );
    ensure!(
        !load.native_mtp_enabled,
        "MLX staged execution does not support native MTP"
    );
    ensure!(
        load.flash_attn_type == skippy_protocol::FlashAttentionType::Auto,
        "MLX staged execution does not support flash-attention overrides"
    );
    ensure!(
        load.mmap.is_none() && !load.mlock,
        "MLX staged execution does not support mmap/mlock overrides"
    );
    ensure!(
        load.projector_path.is_none(),
        "MLX staged execution does not support multimodal projectors"
    );
    ensure!(
        matches!(load.n_gpu_layers, -1 | 0),
        "MLX staged execution does not support partial GPU offload"
    );
    if let Some(device) = load.selected_device.as_ref() {
        ensure!(
            device.backend_device.to_ascii_lowercase().contains("metal")
                && device.index.is_none_or(|index| index == 0),
            "MLX stage selected_device must be Metal device 0"
        );
    }
    Ok(())
}

fn downstream_addr(load: &StageLoadRequest) -> Result<Option<SocketAddr>> {
    load.downstream
        .as_ref()
        .map(|peer| {
            peer.endpoint
                .parse()
                .with_context(|| format!("parse MLX downstream endpoint {}", peer.endpoint))
        })
        .transpose()
}

fn wire_dtype(dtype: StageWireDType) -> Result<WireActivationDType> {
    match dtype {
        StageWireDType::F32 => Ok(WireActivationDType::F32),
        StageWireDType::F16 => Ok(WireActivationDType::F16),
        StageWireDType::Q8 => bail!("MLX stages do not support Q8 activation wire dtype"),
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Write, net::TcpStream, time::Duration};

    use skippy_engine_mlx::MlxDerivedStageReport;
    use skippy_protocol::FlashAttentionType;
    use skippy_protocol::binary::{
        StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind, recv_ready, recv_reply,
        write_stage_message,
    };

    use super::super::{
        StageControlCommand, StageControlRequest, StageControlResponse, StageInventoryRequest,
        StagePeerDescriptor, StagePreparationState, StagePrepareRequest, StageStopRequest,
        spawn_stage_control_loop,
    };
    use super::*;

    const SMOL_REPO: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    const SMOL_REVISION: &str = "12fd25f77366fa6b3b4b768ec3050bf629380bac";
    const SMOL_PROMPT: &[i32] = &[1, 1531, 314, 260, 3575, 28];
    const SMOL_EXPECTED: &[i32] = &[260, 2240, 314, 253, 1379, 282, 25801, 28];

    #[test]
    fn parses_commit_addressed_mlx_stage_ref() {
        let request = request_from_load(&load_request()).unwrap();

        assert_eq!(request.repo, "org/model");
        assert_eq!(request.revision, "a".repeat(40));
        assert_eq!((request.layer_start, request.layer_end), (4, 8));
    }

    #[test]
    fn rejects_mutable_mlx_stage_ref() {
        let mut load = load_request();
        load.package_ref = "hf-model://org/model@main".to_string();

        assert!(request_from_load(&load).is_err());
    }

    #[test]
    fn rejects_unsupported_load_mode_and_lanes() {
        let mut load = load_request();
        load.load_mode = LoadMode::LayerPackage;
        assert!(request_from_load(&load).is_err());

        load.load_mode = LoadMode::ArtifactSlice;
        load.lane_count = 2;
        assert!(request_from_load(&load).is_err());
    }

    #[test]
    fn rejects_unsupported_runtime_overrides_before_download() {
        let mut load = load_request();
        load.wire_dtype = StageWireDType::Q8;
        assert!(request_from_load(&load).is_err());

        load.wire_dtype = StageWireDType::F16;
        load.n_batch = Some(8);
        assert!(request_from_load(&load).is_err());

        load.n_batch = None;
        load.native_mtp_enabled = true;
        assert!(request_from_load(&load).is_err());
    }

    #[test]
    fn maps_explicit_and_hardware_auto_weight_quantization() {
        assert_eq!(
            mlx_weight_quantization(StageWeightQuantization::Auto),
            MlxWeightQuantization::Affine {
                group_size: 64,
                bits: 4
            }
        );
        assert_eq!(
            mlx_weight_quantization(StageWeightQuantization::Affine8),
            MlxWeightQuantization::Affine {
                group_size: 64,
                bits: 8
            }
        );
        assert_eq!(
            mlx_weight_quantization(StageWeightQuantization::MxFp4),
            MlxWeightQuantization::MxFp4
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "downloads about 310 MiB and requires Apple Silicon Metal"]
    async fn real_control_plane_derives_and_runs_two_quantized_range_stages() -> Result<()> {
        let plan = tokio::task::spawn_blocking(|| {
            SafetensorsStageMaterializer::from_environment()?.plan(SafetensorsStageRequest {
                repo: SMOL_REPO.to_string(),
                revision: SMOL_REVISION.to_string(),
                layer_start: 0,
                layer_end: 15,
                include_prefixes: Vec::new(),
            })
        })
        .await
        .context("join SmolLM2 range plan")??;
        ensure!(
            plan.planned_download_bytes < plan.source_shard_bytes,
            "SmolLM2 stage plan did not avoid the complete source shard"
        );

        let control = spawn_stage_control_loop(None);
        let final_load = smol_load(1, 15, 30, None, &plan.checkpoint_sha256);
        prepare_and_wait(&control, &final_load).await?;
        let final_ready = load_through_control(&control, &final_load).await?;
        ensure!(final_ready.accepted, "final MLX stage load was rejected");
        assert_range_only_status(&final_ready.status, &final_load)?;

        let downstream = StagePeerDescriptor {
            stage_id: final_load.stage_id.clone(),
            stage_index: final_load.stage_index,
            endpoint: final_ready.status.bind_addr.clone(),
            node_id: None,
        };
        let first_load = smol_load(0, 0, 15, Some(downstream), &plan.checkpoint_sha256);
        prepare_and_wait(&control, &first_load).await?;
        let first_ready = load_through_control(&control, &first_load).await?;
        ensure!(first_ready.accepted, "first MLX stage load was rejected");
        assert_range_only_status(&first_ready.status, &first_load)?;

        let first_addr = first_ready.status.bind_addr.parse()?;
        let generated = tokio::task::spawn_blocking(move || prove_chain(first_addr))
            .await
            .context("join MLX control-plane proof client")??;
        ensure!(generated == SMOL_EXPECTED, "MLX stage tokens diverged");

        stop_stage(&control, &first_load).await?;
        stop_stage(&control, &final_load).await?;
        Ok(())
    }

    async fn prepare_and_wait(
        control: &tokio::sync::mpsc::UnboundedSender<StageControlCommand>,
        load: &StageLoadRequest,
    ) -> Result<()> {
        let response = send_control(
            control,
            StageControlRequest::Prepare(StagePrepareRequest {
                load: load.clone(),
                coordinator_id: None,
            }),
        )
        .await?;
        let StageControlResponse::PrepareAccepted(accepted) = response else {
            bail!("MLX prepare returned the wrong control response");
        };
        ensure!(accepted.accepted, "MLX stage prepare was rejected");
        tokio::time::timeout(Duration::from_secs(600), async {
            loop {
                let response = send_control(
                    control,
                    StageControlRequest::Inventory(StageInventoryRequest {
                        model_id: load.model_id.clone(),
                        package_ref: load.package_ref.clone(),
                        manifest_sha256: load.manifest_sha256.clone(),
                        weight_quantization: load.weight_quantization,
                    }),
                )
                .await?;
                let StageControlResponse::Inventory(inventory) = response else {
                    bail!("MLX inventory returned the wrong control response");
                };
                if let Some(status) = inventory
                    .preparing_ranges
                    .into_iter()
                    .find(|status| status.stage_id == load.stage_id)
                {
                    match status.state {
                        StagePreparationState::Available => return Ok(()),
                        StagePreparationState::Failed => {
                            bail!(
                                "MLX stage preparation failed: {}",
                                status.error.as_deref().unwrap_or("unknown error")
                            );
                        }
                        _ => {}
                    }
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .context("timed out preparing MLX SafeTensors stage")?
    }

    async fn load_through_control(
        control: &tokio::sync::mpsc::UnboundedSender<StageControlCommand>,
        load: &StageLoadRequest,
    ) -> Result<StageReadyResponse> {
        let response = send_control(control, StageControlRequest::Load(load.clone())).await?;
        let StageControlResponse::Ready(ready) = response else {
            bail!("MLX load returned the wrong control response");
        };
        Ok(ready)
    }

    fn assert_range_only_status(
        status: &super::super::StageStatusSnapshot,
        load: &StageLoadRequest,
    ) -> Result<()> {
        ensure!(
            status.manifest_sha256.as_deref() == Some(load.manifest_sha256.as_str()),
            "running MLX stage checkpoint identity changed"
        );
        ensure!(status.source_model_sha256.is_none());
        ensure!(status.source_model_bytes.is_none());
        ensure!(!status.materialized_pinned);
        let artifact_path = status
            .materialized_path
            .as_deref()
            .context("running MLX stage has no materialized path")?;
        let plan: model_hf::safetensors_stage::SafetensorsStagePlan = serde_json::from_slice(
            &std::fs::read(std::path::Path::new(artifact_path).join("stage-plan.json"))?,
        )?;
        let report: MlxDerivedStageReport = serde_json::from_slice(&std::fs::read(
            std::path::Path::new(artifact_path).join("derived-stage.json"),
        )?)?;
        ensure!(
            plan.planned_download_bytes < plan.source_shard_bytes,
            "running MLX stage planned a complete source shard download"
        );
        ensure!(
            report.checkpoint_sha256 == load.manifest_sha256,
            "running MLX stage changed source checkpoint identity"
        );
        ensure!(
            report.quantization_label == "affine-4bit-g64",
            "running MLX stage did not use the requested quantization"
        );
        Ok(())
    }

    async fn stop_stage(
        control: &tokio::sync::mpsc::UnboundedSender<StageControlCommand>,
        load: &StageLoadRequest,
    ) -> Result<()> {
        let response = send_control(
            control,
            StageControlRequest::Stop(StageStopRequest {
                topology_id: load.topology_id.clone(),
                run_id: load.run_id.clone(),
                stage_id: load.stage_id.clone(),
                shutdown_generation: load.shutdown_generation + 1,
                coordinator_term: load.coordinator_term,
            }),
        )
        .await?;
        let StageControlResponse::Ready(response) = response else {
            bail!("MLX stop returned the wrong control response");
        };
        ensure!(response.accepted, "MLX stage stop was rejected");
        Ok(())
    }

    async fn send_control(
        control: &tokio::sync::mpsc::UnboundedSender<StageControlCommand>,
        request: StageControlRequest,
    ) -> Result<StageControlResponse> {
        let (resp, rx) = tokio::sync::oneshot::channel();
        control
            .send(StageControlCommand { request, resp })
            .map_err(|_| anyhow::anyhow!("MLX stage control loop stopped"))?;
        rx.await.context("MLX stage control response dropped")?
    }

    fn smol_load(
        stage_index: u32,
        layer_start: u32,
        layer_end: u32,
        downstream: Option<StagePeerDescriptor>,
        checkpoint_sha256: &str,
    ) -> StageLoadRequest {
        let mut load = load_request();
        load.topology_id = "mlx-control-proof".to_string();
        load.run_id = "smollm2-range-only".to_string();
        load.model_id = SMOL_REPO.to_string();
        load.package_ref = format!("hf-model://{SMOL_REPO}@{SMOL_REVISION}");
        load.manifest_sha256 = checkpoint_sha256.to_string();
        load.stage_id = format!("stage-{stage_index}");
        load.stage_index = stage_index;
        load.layer_start = layer_start;
        load.layer_end = layer_end;
        load.activation_width = 576;
        load.downstream = downstream;
        load
    }

    fn prove_chain(connect: SocketAddr) -> Result<Vec<i32>> {
        let wire_dtype = WireActivationDType::F16;
        let mut stream = TcpStream::connect(connect)
            .with_context(|| format!("connect first MLX stage at {connect}"))?;
        stream.set_nodelay(true).ok();
        recv_ready(&mut stream).context("first MLX stage did not become ready")?;

        let session_id = 1;
        let request_id = 1;
        let prefill = execution_message(
            WireMessageKind::PrefillFinalEmbd,
            SMOL_PROMPT,
            *SMOL_PROMPT.last().expect("non-empty prompt"),
            request_id,
            session_id,
            0,
        );
        let mut generated = Vec::with_capacity(SMOL_EXPECTED.len());
        generated.push(send_predicted(&mut stream, &prefill)?);
        while generated.len() < SMOL_EXPECTED.len() {
            let current = *generated.last().expect("generated has first token");
            let decode = execution_message(
                WireMessageKind::DecodeEmbd,
                &[],
                current,
                request_id,
                session_id,
                i32::try_from(generated.len())?,
            );
            generated.push(send_predicted(&mut stream, &decode)?);
        }

        let stop = StageWireMessage::stop_with_identity(wire_dtype, request_id, session_id);
        write_stage_message(&mut stream, &stop, wire_dtype)?;
        stream.flush().ok();
        ensure!(recv_reply(&mut stream)?.kind == WireReplyKind::Ack);
        Ok(generated)
    }

    fn execution_message(
        kind: WireMessageKind,
        tokens: &[i32],
        current_token: i32,
        request_id: u64,
        session_id: u64,
        decode_step: i32,
    ) -> StageWireMessage {
        let wire_dtype = WireActivationDType::F16;
        let mut state = StageStateHeader::new(kind, wire_dtype);
        state.current_token = current_token;
        state.prompt_token_count = i32::try_from(tokens.len()).unwrap_or_default();
        state.decode_step = decode_step;
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count: if kind == WireMessageKind::DecodeEmbd {
                1
            } else {
                i32::try_from(tokens.len()).unwrap_or_default()
            },
            state,
            request_id,
            session_id,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: tokens.to_vec(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn send_predicted(stream: &mut TcpStream, message: &StageWireMessage) -> Result<i32> {
        write_stage_message(&mut *stream, message, WireActivationDType::F16)?;
        stream.flush().ok();
        let reply = recv_reply(&mut *stream)?;
        ensure!(
            matches!(
                reply.kind,
                WireReplyKind::PredictedToken | WireReplyKind::PredictedTokens
            ),
            "MLX stage chain did not return a predicted token"
        );
        Ok(reply.predicted)
    }

    fn load_request() -> StageLoadRequest {
        StageLoadRequest {
            topology_id: "topology".to_string(),
            run_id: "run".to_string(),
            model_id: "org/model".to_string(),
            backend: "mlx".to_string(),
            package_ref: format!("hf-model://org/model@{}", "a".repeat(40)),
            manifest_sha256: "b".repeat(64),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 4,
            layer_end: 8,
            model_path: None,
            source_model_bytes: None,
            projector_path: None,
            selected_device: None,
            bind_addr: "127.0.0.1:0".to_string(),
            activation_width: 2,
            wire_dtype: StageWireDType::F16,
            ctx_size: 128,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            weight_quantization: StageWeightQuantization::Affine4,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            native_mtp_enabled: false,
            shutdown_generation: 0,
            coordinator_term: 0,
            coordinator_id: None,
            lease_until_unix_ms: 0,
            load_mode: LoadMode::ArtifactSlice,
            upstream: None,
            downstream: None,
        }
    }
}
