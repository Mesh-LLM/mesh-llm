use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use model_artifact::ModelIdentity;
use model_ref::ModelRef;
use serde::{Deserialize, Serialize};
use serde_json::json;
use skippy_protocol::binary::{StageWireMessage, recv_ready, write_stage_message};
use skippy_protocol::{LoadMode, StageTopology, StageTopologyEntry};
use skippy_runtime::write_gguf_from_parts;
use skippy_topology::{
    BoundaryDecision, NodeSpec, PlannerPolicy, TopologyPlanRequest, WireValidation,
    dense_attention_layers, infer_family_capability, plan_contiguous_with_splits,
};

use super::{path_string, write_json_file};
use crate::{
    cli::RunArgs,
    support::{ChildGuard, parse_wire_dtype},
};

#[derive(Debug, Clone, Serialize)]
pub(super) struct StageAssignment {
    pub(super) stage_id: String,
    pub(super) stage_index: u32,
    pub(super) host: String,
    pub(super) local: bool,
    pub(super) layer_start: u32,
    pub(super) layer_end: u32,
    pub(super) bind_addr: String,
    pub(super) endpoint: String,
    pub(super) config_path: PathBuf,
    pub(super) remote_config_path: String,
    pub(super) remote_log_path: String,
    pub(super) remote_pid_path: String,
    pub(super) remote_exit_code_path: String,
    pub(super) remote_model_path: Option<String>,
    pub(super) local_materialized_model_path: Option<PathBuf>,
    pub(super) local_shared_model_path: Option<PathBuf>,
    pub(super) selected_package_files: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct DeploymentPlan {
    pub(super) run_id: String,
    pub(super) topology_id: String,
    pub(super) model_id: String,
    pub(super) model_identity: ModelIdentity,
    pub(super) hosts: Vec<String>,
    pub(super) stage_load_mode: String,
    pub(super) remote_root: String,
    pub(super) remote_roots: BTreeMap<String, String>,
    pub(super) remote_shared_roots: BTreeMap<String, PathBuf>,
    pub(super) endpoint_hosts: BTreeMap<String, String>,
    pub(super) work_dir: PathBuf,
    pub(super) metrics_http: String,
    pub(super) metrics_otlp_grpc: String,
    pub(super) driver_return_endpoint: String,
    pub(super) stages: Vec<StageAssignment>,
    pub(super) execute_remote: bool,
    pub(super) keep_remote: bool,
    pub(super) rsync_model_artifacts: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct RemoteStageStatus {
    pub(super) stage_id: String,
    pub(super) host: String,
    pub(super) pid: Option<u32>,
    pub(super) pid_alive: bool,
    pub(super) log_ready: bool,
    pub(super) protocol_ready: bool,
    pub(super) exit_code: Option<i32>,
    pub(super) log_tail: String,
    pub(super) collected_log_path: Option<PathBuf>,
    pub(super) terminated: bool,
}
#[derive(Debug, Deserialize)]
struct PackageManifest {
    model_id: String,
    source_model: PackageSourceModel,
    shared: PackageShared,
    layers: Vec<PackageLayer>,
}

#[derive(Debug, Deserialize)]
struct PackageSourceModel {
    repo: Option<String>,
    revision: Option<String>,
    primary_file: Option<String>,
    canonical_ref: Option<String>,
    distribution_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PackageShared {
    metadata: PackageArtifact,
    embeddings: PackageArtifact,
    output: PackageArtifact,
}

#[derive(Debug, Deserialize)]
struct PackageArtifact {
    path: String,
}

#[derive(Debug, Deserialize)]
struct PackageLayer {
    layer_index: u32,
    path: String,
}

pub(super) fn validate_topology_plan(
    args: &RunArgs,
    hosts: &[String],
    ranges: &[(u32, u32)],
) -> Result<()> {
    let identity = format!(
        "{} {} {}",
        args.model_id,
        args.model_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        args.stage_model
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default()
    );
    let activation_width =
        u32::try_from(args.activation_width).context("activation_width must be non-negative")?;
    let family = infer_family_capability(&identity, args.layer_end, activation_width);
    let request = TopologyPlanRequest {
        topology_id: args.topology_id.clone(),
        model_id: args.model_id.clone(),
        layers: dense_attention_layers(args.layer_end, 0),
        nodes: hosts
            .iter()
            .map(|host| NodeSpec {
                node_id: host.clone(),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect(),
        family: family.clone(),
        policy: PlannerPolicy::default(),
    };
    let splits = split_boundaries_from_ranges(ranges);
    let plan = plan_contiguous_with_splits(&request, &splits).context("topology planner failed")?;

    if args.activation_wire_dtype.eq_ignore_ascii_case("q8") {
        match family.as_ref().map(|family| family.q8_wire_validation) {
            Some(WireValidation::Validated) => {}
            Some(WireValidation::Rejected) => {
                bail!(
                    "topology planner rejected q8 activation wire dtype for {}; use f16 or add a passing q8 correctness record",
                    args.model_id
                );
            }
            Some(WireValidation::Untested) => {
                bail!(
                    "topology planner has no q8 validation for {}; use f16 until this family/split passes correctness",
                    args.model_id
                );
            }
            None => {}
        }
    }

    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == BoundaryDecision::Rejected)
        .collect::<Vec<_>>();
    if !rejected.is_empty() {
        let reasons = rejected
            .iter()
            .map(|boundary| {
                format!(
                    "layer {}: {:?}: {}",
                    boundary.layer_boundary,
                    boundary.reason_codes,
                    boundary.messages.join("; ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        bail!("topology planner rejected split plan:\n{reasons}");
    }

    Ok(())
}

fn split_boundaries_from_ranges(ranges: &[(u32, u32)]) -> Vec<u32> {
    ranges
        .iter()
        .take(ranges.len().saturating_sub(1))
        .map(|(_, end)| *end)
        .collect()
}

pub(super) fn build_deployment_plan(
    args: &RunArgs,
    run_id: &str,
    hosts: &[String],
    ranges: &[(u32, u32)],
    config_dir: &Path,
    model_ref: &str,
    model_identity: ModelIdentity,
) -> Result<DeploymentPlan> {
    let metrics_http = format!("http://{}", args.metrics_http_addr);
    let metrics_otlp = metrics_otlp_grpc_url(args);
    let remote_root_map = parse_remote_root_map(args.remote_root_map.as_deref())?;
    let remote_shared_root_map = parse_path_map(args.remote_shared_root_map.as_deref())?;
    let endpoint_host_map = parse_remote_root_map(args.endpoint_host_map.as_deref())?;
    let package_manifest = if args.stage_load_mode == "layer-package" {
        args.stage_model
            .as_ref()
            .filter(|path| path.is_dir())
            .map(|path| load_package_manifest(path))
            .transpose()?
    } else {
        None
    };
    let coordinator_materializes =
        coordinator_materializes_layer_package(args) && package_manifest.is_some();
    let mut stages = Vec::with_capacity(ranges.len());
    for (index, (layer_start, layer_end)) in ranges.iter().copied().enumerate() {
        let stage_id = format!("stage-{index}");
        let host = hosts[index % hosts.len()].clone();
        let local = args.execute_remote && index == 0;
        let port = args
            .first_stage_port
            .checked_add(u16::try_from(index).context("stage index exceeds u16")?)
            .context("stage port overflow")?;
        let endpoint_host = endpoint_host_map
            .get(&host)
            .map(String::as_str)
            .unwrap_or(&host);
        let bind_host = endpoint_host_map
            .get(&host)
            .map(String::as_str)
            .unwrap_or(&args.remote_bind_host);
        let bind_addr = format!("{bind_host}:{port}");
        let endpoint = format!("tcp://{endpoint_host}:{port}");
        let host_remote_root = remote_root_map
            .get(&host)
            .map(String::as_str)
            .unwrap_or(&args.remote_root);
        let remote_stage_dir = format!("{host_remote_root}/{run_id}/{stage_id}");
        let selected_package_files = package_manifest
            .as_ref()
            .map(|manifest| {
                selected_package_files(
                    manifest,
                    layer_start,
                    layer_end,
                    index == 0,
                    index + 1 == ranges.len(),
                )
            })
            .transpose()?
            .unwrap_or_default();
        let stage_cache_key = if coordinator_materializes {
            Some(stage_model_cache_key(
                args,
                &stage_id,
                layer_start,
                layer_end,
            ))
        } else {
            None
        };
        let local_materialized_model_path = stage_cache_key.as_ref().map(|key| {
            args.work_dir
                .join("model-cache")
                .join(key)
                .join("stage.gguf")
        });
        let remote_model_path = stage_cache_key.as_ref().map(|key| {
            format!(
                "{host_remote_root}/model-cache/{}/stage.gguf",
                key.display()
            )
        });
        let local_shared_model_path = if let Some(key) = stage_cache_key.as_ref() {
            remote_shared_root_map
                .get(&host)
                .map(|root| root.join("model-cache").join(key).join("stage.gguf"))
        } else {
            None
        };
        stages.push(StageAssignment {
            stage_id,
            stage_index: index as u32,
            host,
            local,
            layer_start,
            layer_end,
            bind_addr,
            endpoint,
            config_path: config_dir.join(format!("stage-{index}.json")),
            remote_config_path: format!("{remote_stage_dir}/stage.json"),
            remote_log_path: format!("{remote_stage_dir}/stage.log"),
            remote_pid_path: format!("{remote_stage_dir}/stage.pid"),
            remote_exit_code_path: format!("{remote_stage_dir}/stage.exit"),
            remote_model_path,
            local_materialized_model_path,
            local_shared_model_path,
            selected_package_files,
        });
    }

    let model_identity = package_manifest
        .as_ref()
        .map(model_identity_from_package_manifest)
        .transpose()?
        .unwrap_or(model_identity);
    let plan = DeploymentPlan {
        run_id: run_id.to_string(),
        topology_id: args.topology_id.clone(),
        model_id: model_identity.model_id.clone(),
        model_identity,
        hosts: hosts.to_vec(),
        stage_load_mode: args.stage_load_mode.clone(),
        remote_root: args.remote_root.clone(),
        remote_roots: remote_root_map,
        remote_shared_roots: remote_shared_root_map,
        endpoint_hosts: endpoint_host_map,
        work_dir: args.work_dir.clone(),
        metrics_http,
        metrics_otlp_grpc: metrics_otlp,
        driver_return_endpoint: driver_return_endpoint(args, &stages)?,
        stages,
        execute_remote: args.execute_remote,
        keep_remote: args.keep_remote,
        rsync_model_artifacts: args.rsync_model_artifacts,
    };

    let _ = model_ref;
    Ok(plan)
}

fn model_identity_from_package_manifest(manifest: &PackageManifest) -> Result<ModelIdentity> {
    let model_ref = ModelRef::parse(&manifest.model_id).with_context(|| {
        format!(
            "package manifest model_id must be a model coordinate, got {:?}",
            manifest.model_id
        )
    })?;
    Ok(ModelIdentity {
        model_id: model_ref.display_id(),
        source_repo: manifest.source_model.repo.clone(),
        source_revision: manifest.source_model.revision.clone(),
        source_file: manifest.source_model.primary_file.clone(),
        canonical_ref: manifest.source_model.canonical_ref.clone(),
        distribution_id: manifest.source_model.distribution_id.clone(),
        selector: model_ref.selector,
    })
}

pub(super) fn write_stage_configs(
    args: &RunArgs,
    plan: &DeploymentPlan,
    model_ref: &str,
) -> Result<()> {
    for stage in &plan.stages {
        let stage_model_ref = if let Some(remote_model_path) = stage.remote_model_path.as_ref() {
            if stage.local {
                stage
                    .local_materialized_model_path
                    .as_ref()
                    .map(|path| path_string(path))
                    .unwrap_or_else(|| remote_model_path.clone())
            } else {
                remote_model_path.clone()
            }
        } else if args.execute_remote
            && args.rsync_model_artifacts
            && args.stage_load_mode == "layer-package"
        {
            format!("{}/package", remote_parent(&stage.remote_config_path)?)
        } else {
            model_ref.to_string()
        };
        let config_load_mode = stage_config_load_mode(args, stage);
        let upstream = if stage.stage_index == 0 {
            json!(null)
        } else {
            let previous = &plan.stages[stage.stage_index as usize - 1];
            json!({
                "stage_id": previous.stage_id,
                "stage_index": previous.stage_index,
                "endpoint": if stage.stage_index == 1 { "driver".to_string() } else { previous.endpoint.clone() }
            })
        };
        let downstream = plan
            .stages
            .get(stage.stage_index as usize + 1)
            .map(|next| {
                json!({
                    "stage_id": next.stage_id,
                    "stage_index": next.stage_index,
                    "endpoint": next.endpoint,
                })
            })
            .unwrap_or_else(|| json!(null));
        let config = json!({
            "run_id": plan.run_id,
            "topology_id": plan.topology_id,
            "model_id": plan.model_id,
            "model_path": stage_model_ref,
            "stage_id": stage.stage_id,
            "stage_index": stage.stage_index,
            "layer_start": stage.layer_start,
            "layer_end": stage.layer_end,
            "ctx_size": args.ctx_size,
            "n_gpu_layers": args.n_gpu_layers,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "filter_tensors_on_load": config_load_mode != "runtime-slice",
            "load_mode": config_load_mode,
            "bind_addr": stage.bind_addr,
            "upstream": upstream,
            "downstream": downstream,
        });
        write_json_file(&stage.config_path, &config)?;
    }
    Ok(())
}

pub(super) fn write_stage_topology(
    args: &RunArgs,
    plan: &DeploymentPlan,
    topology_path: &Path,
) -> Result<()> {
    let topology = StageTopology {
        topology_id: plan.topology_id.clone(),
        model_id: plan.model_id.clone(),
        stages: plan
            .stages
            .iter()
            .map(|stage| {
                Ok(StageTopologyEntry {
                    stage_id: stage.stage_id.clone(),
                    stage_index: stage.stage_index,
                    host: Some(stage.host.clone()),
                    endpoint: if stage.stage_index == 0 {
                        format!("tcp://{}", plan.driver_return_endpoint)
                    } else {
                        stage.endpoint.clone()
                    },
                    layer_start: stage.layer_start,
                    layer_end: stage.layer_end,
                    load_mode: parse_load_mode(stage_config_load_mode(args, stage))?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    write_json_file(topology_path, &topology)
}

fn stage_config_load_mode<'a>(args: &'a RunArgs, stage: &StageAssignment) -> &'a str {
    if stage.remote_model_path.is_some() && args.stage_load_mode == "layer-package" {
        "artifact-slice"
    } else {
        args.stage_load_mode.as_str()
    }
}

fn parse_load_mode(load_mode: &str) -> Result<LoadMode> {
    match load_mode {
        "artifact-slice" => Ok(LoadMode::ArtifactSlice),
        "layer-package" => Ok(LoadMode::LayerPackage),
        "runtime-slice" => Ok(LoadMode::RuntimeSlice),
        _ => bail!("unsupported stage load mode for topology: {load_mode}"),
    }
}

fn driver_return_endpoint(args: &RunArgs, stages: &[StageAssignment]) -> Result<String> {
    let first = stages.first().context("deployment plan has no stages")?;
    let endpoint = first
        .endpoint
        .strip_prefix("tcp://")
        .unwrap_or(&first.endpoint);
    let host = endpoint_host(endpoint)?;
    let host = if host == "localhost" {
        "127.0.0.1"
    } else {
        host
    };
    Ok(format!("{host}:{}", driver_return_port(args)))
}

fn endpoint_host(endpoint: &str) -> Result<&str> {
    if let Some(rest) = endpoint.strip_prefix('[') {
        let (host, _) = rest
            .split_once(']')
            .with_context(|| format!("invalid bracketed endpoint host: {endpoint}"))?;
        return Ok(host);
    }
    endpoint
        .rsplit_once(':')
        .map(|(host, _)| host)
        .with_context(|| format!("endpoint is missing port: {endpoint}"))
}

fn driver_return_port(args: &RunArgs) -> u16 {
    args.first_stage_port.saturating_add(1000).max(1)
}

pub(super) fn execute_remote_plan(
    args: &RunArgs,
    plan: &DeploymentPlan,
) -> Result<Vec<ChildGuard>> {
    let mut sessions = Vec::with_capacity(plan.stages.len());
    let mut started_stages = Vec::with_capacity(plan.stages.len());
    for stage in plan.stages.iter().rev() {
        if stage.local {
            prepare_local_stage(args, stage)?;
            let command = local_start_command(args, plan, stage);
            let mut local = Command::new("sh");
            local
                .arg("-c")
                .arg(command)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            sessions.push(ChildGuard::spawn(local).with_context(|| {
                format!("start local stage {} on {}", stage.stage_id, stage.host)
            })?);
        } else {
            let remote_stage_dir = remote_parent(&stage.remote_config_path)?;
            run_command(
                Command::new("ssh")
                    .arg(&stage.host)
                    .arg(format!("mkdir -p {remote_stage_dir}")),
            )
            .with_context(|| format!("create remote stage dir on {}", stage.host))?;

            run_command(
                Command::new("rsync")
                    .arg("-az")
                    .arg(&args.stage_server_bin)
                    .arg(format!("{}:{remote_stage_dir}/skippy-server", stage.host)),
            )
            .with_context(|| format!("rsync stage server to {}", stage.host))?;

            run_command(
                Command::new("rsync")
                    .arg("-az")
                    .arg(&stage.config_path)
                    .arg(format!("{}:{}", stage.host, stage.remote_config_path)),
            )
            .with_context(|| format!("rsync config to {}", stage.host))?;
            run_command(
                Command::new("rsync")
                    .arg("-az")
                    .arg(stage_topology_source_path(stage)?)
                    .arg(format!(
                        "{}:{}",
                        stage.host,
                        stage_remote_topology_path(stage)?
                    )),
            )
            .with_context(|| format!("rsync topology to {}", stage.host))?;

            if args.rsync_model_artifacts {
                rsync_model_artifacts(args, stage)?;
            }

            let remote_bin = format!("{remote_stage_dir}/skippy-server");
            let command = remote_start_command(args, plan, stage, &remote_bin);
            let mut ssh = Command::new("ssh");
            ssh.arg(&stage.host)
                .arg(command)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            sessions
                .push(ChildGuard::spawn(ssh).with_context(|| {
                    format!("start stage {} on {}", stage.stage_id, stage.host)
                })?);
        }
        started_stages.push(stage);
        if let Err(error) = wait_stage_log_ready(stage, args.startup_timeout_secs)
            .with_context(|| format!("wait for {} on {} to listen", stage.stage_id, stage.host))
        {
            for started_stage in &started_stages {
                let pid = remote_pid(started_stage).ok().flatten();
                let _ = terminate_remote_stage(started_stage, pid);
            }
            return Err(error);
        }
    }
    Ok(sessions)
}

fn wait_stage_log_ready(stage: &StageAssignment, timeout_secs: u64) -> Result<()> {
    let attempts = timeout_secs.saturating_mul(2).max(1);
    for _ in 0..attempts {
        if remote_log_ready(stage).unwrap_or(false) {
            return Ok(());
        }
        if let Some(exit_code) = remote_exit_code(stage).ok().flatten() {
            let log_tail = remote_log_tail(stage)
                .unwrap_or_else(|error| format!("failed to read remote log tail: {error:#}"));
            bail!("stage exited before listening with code {exit_code}; log tail:\n{log_tail}");
        }
        thread::sleep(Duration::from_millis(500));
    }
    bail!(
        "stage did not report listening in {}",
        stage.remote_log_path
    )
}

fn prepare_local_stage(args: &RunArgs, stage: &StageAssignment) -> Result<()> {
    let local_stage_dir = remote_parent(&stage.remote_config_path)?;
    fs::create_dir_all(&local_stage_dir)
        .with_context(|| format!("create local stage dir {local_stage_dir}"))?;
    fs::copy(&stage.config_path, &stage.remote_config_path).with_context(|| {
        format!(
            "copy local stage config {} to {}",
            stage.config_path.display(),
            stage.remote_config_path
        )
    })?;
    fs::copy(
        stage_topology_source_path(stage)?,
        stage_remote_topology_path(stage)?,
    )
    .with_context(|| {
        format!(
            "copy local stage topology {} to {}",
            stage_topology_source_path(stage)
                .map(|path| path.display().to_string())
                .unwrap_or_else(|_| "<unknown>".to_string()),
            stage_remote_topology_path(stage).unwrap_or_else(|_| "<unknown>".to_string())
        )
    })?;
    if !args.rsync_model_artifacts {
        return Ok(());
    }
    let Some(stage_model) = args.stage_model.as_ref() else {
        return Ok(());
    };
    let Some(local_model) = stage.local_materialized_model_path.as_ref() else {
        return Ok(());
    };
    materialize_stage_model_on_coordinator(stage_model, stage, local_model)?;
    Ok(())
}

fn stage_topology_source_path(stage: &StageAssignment) -> Result<PathBuf> {
    let parent = stage
        .config_path
        .parent()
        .context("stage config path has no parent")?;
    Ok(parent.join("topology.json"))
}

fn stage_remote_topology_path(stage: &StageAssignment) -> Result<String> {
    Ok(format!(
        "{}/topology.json",
        remote_parent(&stage.remote_config_path)?
    ))
}

fn local_start_command(args: &RunArgs, plan: &DeploymentPlan, stage: &StageAssignment) -> String {
    stage_start_wrapper(
        args,
        plan,
        stage,
        &path_string(&args.stage_server_bin),
        &stage.remote_config_path,
    )
}

fn remote_start_command(
    args: &RunArgs,
    plan: &DeploymentPlan,
    stage: &StageAssignment,
    remote_bin: &str,
) -> String {
    stage_start_wrapper(args, plan, stage, remote_bin, &stage.remote_config_path)
}

fn stage_start_wrapper(
    args: &RunArgs,
    plan: &DeploymentPlan,
    stage: &StageAssignment,
    bin: &str,
    config_path: &str,
) -> String {
    let stage_command = stage_server_command(args, plan, stage, bin, config_path);
    let exit_path = shell_quote(&stage.remote_exit_code_path);
    let log_path = shell_quote(&stage.remote_log_path);
    let pid_path = shell_quote(&stage.remote_pid_path);
    let wrapper = format!(
        "trap 'kill \"$child\" 2>/dev/null || true; wait \"$child\" 2>/dev/null; status=$?; printf \"%s\\n\" \"$status\" > {exit_path}; exit \"$status\"' TERM INT HUP; {stage_command} > {log_path} 2>&1 & child=$!; printf \"%s\\n\" \"$child\" > {pid_path}; wait \"$child\"; status=$?; printf \"%s\\n\" \"$status\" > {exit_path}; exit \"$status\""
    );
    format!(
        "chmod +x {} && rm -f {} {} && sh -c {}",
        shell_quote(bin),
        shell_quote(&stage.remote_exit_code_path),
        shell_quote(&stage.remote_pid_path),
        shell_quote(&wrapper),
    )
}

fn stage_server_command(
    args: &RunArgs,
    plan: &DeploymentPlan,
    stage: &StageAssignment,
    bin: &str,
    config_path: &str,
) -> String {
    let reply_credit_arg = args
        .stage_reply_credit_limit
        .map(|limit| format!(" --reply-credit-limit {limit}"))
        .unwrap_or_default();
    let async_prefill_forward_arg = if args.stage_async_prefill_forward {
        " --async-prefill-forward"
    } else {
        ""
    };
    let downstream_wire_mbps_arg = args
        .stage_downstream_wire_mbps
        .map(|mbps| format!(" --downstream-wire-mbps {mbps}"))
        .unwrap_or_default();
    format!(
        "{} serve-binary --config {} --topology {} --activation-width {} --activation-wire-dtype {} --metrics-otlp-grpc {} --telemetry-queue-capacity {} --telemetry-level {} --max-inflight {}{}{} --downstream-wire-delay-ms {}{}",
        shell_quote(bin),
        shell_quote(config_path),
        shell_quote(
            &stage_remote_topology_path(stage).unwrap_or_else(|_| "topology.json".to_string())
        ),
        args.activation_width,
        shell_quote(&args.activation_wire_dtype),
        shell_quote(&plan.metrics_otlp_grpc),
        args.stage_telemetry_queue_capacity,
        shell_quote(&args.stage_telemetry_level),
        args.stage_max_inflight,
        reply_credit_arg,
        async_prefill_forward_arg,
        args.stage_downstream_wire_delay_ms,
        downstream_wire_mbps_arg,
    )
}

pub(super) fn wait_remote_readiness(
    args: &RunArgs,
    plan: &DeploymentPlan,
) -> Result<Vec<RemoteStageStatus>> {
    let attempts = args.startup_timeout_secs.saturating_mul(2).max(1);
    let mut last_statuses = Vec::new();
    for _ in 0..attempts {
        last_statuses = plan
            .stages
            .iter()
            .map(remote_stage_status)
            .collect::<Vec<_>>();
        if last_statuses
            .iter()
            .all(|status| status.pid_alive && status.log_ready)
        {
            match probe_remote_chain_readiness(args, plan) {
                Ok(()) => {
                    if let Some(first) = last_statuses.first_mut() {
                        first.protocol_ready = true;
                    }
                    return Ok(last_statuses);
                }
                Err(error) => {
                    if let Some(first) = last_statuses.first_mut() {
                        first.log_tail = format!(
                            "{}\nprotocol readiness probe failed: {error:#}",
                            first.log_tail
                        );
                    }
                }
            }
        }
        thread::sleep(Duration::from_millis(500));
    }

    Err(anyhow!(
        "remote stages did not become ready: {}",
        serde_json::to_string(&last_statuses)?
    ))
}

fn probe_remote_chain_readiness(args: &RunArgs, plan: &DeploymentPlan) -> Result<()> {
    let first = plan
        .stages
        .first()
        .context("deployment plan has no stages")?;
    let mut stream = connect_endpoint_ready(&first.endpoint, args.startup_timeout_secs)
        .with_context(|| format!("connect to first binary stage {}", first.endpoint))?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    write_stage_message(&mut stream, &StageWireMessage::stop(wire_dtype), wire_dtype)
        .context("send readiness stop frame")?;
    Ok(())
}

pub(super) fn collect_and_cleanup_remote(
    args: &RunArgs,
    plan: &DeploymentPlan,
    run_dir: &Path,
    protocol_ready: bool,
) -> Result<Vec<RemoteStageStatus>> {
    let logs_dir = run_dir.join("logs");
    fs::create_dir_all(&logs_dir)
        .with_context(|| format!("create remote log directory {}", logs_dir.display()))?;
    let mut statuses = Vec::with_capacity(plan.stages.len());
    for stage in &plan.stages {
        let mut status = remote_stage_status(stage);
        status.protocol_ready = protocol_ready && stage.stage_index == 0;
        status.collected_log_path = collect_remote_log(stage, &logs_dir);
        if !args.keep_remote {
            status.terminated = terminate_remote_stage(stage, status.pid).is_ok();
            let _ = wait_remote_exit_code(stage, Duration::from_secs(5));
            let exit_code = remote_exit_code(stage).ok().flatten();
            if status.terminated {
                status.pid_alive = remote_pid_alive_opt(stage, status.pid).unwrap_or(false);
                status.exit_code = exit_code;
            }
        }
        statuses.push(status);
    }
    Ok(statuses)
}
fn remote_stage_status(stage: &StageAssignment) -> RemoteStageStatus {
    let pid = remote_pid(stage).ok().flatten();
    let pid_alive = pid
        .map(|pid| remote_pid_alive(stage, pid).unwrap_or(false))
        .unwrap_or(false);
    let log_ready = remote_log_ready(stage).unwrap_or(false);
    let exit_code = remote_exit_code(stage).ok().flatten();
    let log_tail = remote_log_tail(stage)
        .unwrap_or_else(|error| format!("failed to read remote log tail: {error:#}"));
    RemoteStageStatus {
        stage_id: stage.stage_id.clone(),
        host: stage.host.clone(),
        pid,
        pid_alive,
        log_ready,
        protocol_ready: false,
        exit_code,
        log_tail,
        collected_log_path: None,
        terminated: false,
    }
}

fn remote_pid(stage: &StageAssignment) -> Result<Option<u32>> {
    if stage.local {
        let output = fs::read_to_string(&stage.remote_pid_path).unwrap_or_default();
        let output = output.trim();
        if output.is_empty() {
            return Ok(None);
        }
        return Ok(Some(
            output
                .parse::<u32>()
                .with_context(|| format!("parse pid for {}", stage.stage_id))?,
        ));
    }
    let output = ssh_capture(
        &stage.host,
        &format!(
            "cat {} 2>/dev/null || true",
            shell_quote(&stage.remote_pid_path)
        ),
    )?;
    let output = output.trim();
    if output.is_empty() {
        return Ok(None);
    }
    Ok(Some(output.parse::<u32>().with_context(|| {
        format!("parse pid for {}", stage.stage_id)
    })?))
}

fn remote_pid_alive(stage: &StageAssignment, pid: u32) -> Result<bool> {
    if stage.local {
        return Ok(Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .status()
            .with_context(|| format!("check local pid {pid} for {}", stage.stage_id))?
            .success());
    }
    ssh_success(&stage.host, &format!("kill -0 {pid} 2>/dev/null"))
}

fn remote_pid_alive_opt(stage: &StageAssignment, pid: Option<u32>) -> Result<bool> {
    pid.map(|pid| remote_pid_alive(stage, pid))
        .unwrap_or(Ok(false))
}

fn remote_exit_code(stage: &StageAssignment) -> Result<Option<i32>> {
    if stage.local {
        let output = fs::read_to_string(&stage.remote_exit_code_path).unwrap_or_default();
        let output = output.trim();
        if output.is_empty() {
            return Ok(None);
        }
        return Ok(Some(output.parse::<i32>().with_context(|| {
            format!("parse exit code for {}", stage.stage_id)
        })?));
    }
    let output = ssh_capture(
        &stage.host,
        &format!(
            "cat {} 2>/dev/null || true",
            shell_quote(&stage.remote_exit_code_path)
        ),
    )?;
    let output = output.trim();
    if output.is_empty() {
        return Ok(None);
    }
    Ok(Some(output.parse::<i32>().with_context(|| {
        format!("parse exit code for {}", stage.stage_id)
    })?))
}

fn wait_remote_exit_code(stage: &StageAssignment, timeout: Duration) -> Result<()> {
    let attempts = (timeout.as_millis() / 250).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match remote_exit_code(stage) {
            Ok(Some(_)) => return Ok(()),
            Ok(None) => {}
            Err(error) => last_error = Some(error),
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out waiting for remote exit code")))
}

fn remote_log_ready(stage: &StageAssignment) -> Result<bool> {
    if stage.local {
        let log = fs::read_to_string(&stage.remote_log_path).unwrap_or_default();
        return Ok(log.contains("skippy-server listening: binary="));
    }
    ssh_success(
        &stage.host,
        &format!(
            "test -f {} && grep -q {} {}",
            shell_quote(&stage.remote_log_path),
            shell_quote("skippy-server listening: binary="),
            shell_quote(&stage.remote_log_path)
        ),
    )
}

fn remote_log_tail(stage: &StageAssignment) -> Result<String> {
    if stage.local {
        let log = fs::read_to_string(&stage.remote_log_path).unwrap_or_default();
        let tail = log
            .lines()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(tail);
    }
    ssh_capture(
        &stage.host,
        &format!(
            "tail -n 40 {} 2>/dev/null || true",
            shell_quote(&stage.remote_log_path)
        ),
    )
}

fn collect_remote_log(stage: &StageAssignment, logs_dir: &Path) -> Option<PathBuf> {
    let local_path = logs_dir.join(format!("{}-{}.log", stage.stage_index, stage.stage_id));
    if stage.local {
        return fs::copy(&stage.remote_log_path, &local_path)
            .ok()
            .map(|_| local_path);
    }
    let status = Command::new("rsync")
        .arg("-az")
        .arg(format!("{}:{}", stage.host, stage.remote_log_path))
        .arg(&local_path)
        .status()
        .ok()?;
    status.success().then_some(local_path)
}

fn terminate_remote_stage(stage: &StageAssignment, pid: Option<u32>) -> Result<()> {
    let Some(pid) = pid else {
        return Ok(());
    };
    if stage.local {
        run_command(
            Command::new("sh").arg("-c").arg(format!(
                "kill -TERM {pid} 2>/dev/null || true; for i in 1 2 3 4 5; do kill -0 {pid} 2>/dev/null || exit 0; sleep 1; done; kill -KILL {pid} 2>/dev/null || true"
            )),
        )?;
        return Ok(());
    }
    ssh_success(
        &stage.host,
        &format!(
            "kill -TERM {pid} 2>/dev/null || true; for i in 1 2 3 4 5; do kill -0 {pid} 2>/dev/null || exit 0; sleep 1; done; kill -KILL {pid} 2>/dev/null || true"
        ),
    )?;
    Ok(())
}

pub(super) fn connect_endpoint_ready(endpoint: &str, timeout_secs: u64) -> Result<TcpStream> {
    let endpoint = endpoint.strip_prefix("tcp://").unwrap_or(endpoint);
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match TcpStream::connect(endpoint) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                match recv_ready(&mut stream) {
                    Ok(()) => return Ok(stream),
                    Err(error) => {
                        last_error = Some(anyhow!(error).context("ready handshake failed"))
                    }
                }
            }
            Err(error) => last_error = Some(anyhow!(error).context("connect failed")),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out")))
}

fn rsync_model_artifacts(args: &RunArgs, stage: &StageAssignment) -> Result<()> {
    let Some(stage_model) = args.stage_model.as_ref() else {
        return Ok(());
    };
    if let (Some(local_model), Some(remote_model)) = (
        stage.local_materialized_model_path.as_ref(),
        stage.remote_model_path.as_ref(),
    ) {
        materialize_stage_model_on_coordinator(stage_model, stage, local_model)?;
        if let Some(shared_model) = stage.local_shared_model_path.as_ref() {
            place_stage_model_on_shared_root(local_model, shared_model).with_context(|| {
                format!(
                    "place coordinator-materialized stage model for {} at {}",
                    stage.stage_id,
                    shared_model.display()
                )
            })?;
        } else {
            let remote_parent = remote_parent(remote_model)?;
            run_command(
                Command::new("ssh")
                    .arg(&stage.host)
                    .arg(format!("mkdir -p {}", shell_quote(&remote_parent))),
            )
            .with_context(|| format!("create remote model cache dir on {}", stage.host))?;
            run_command(
                Command::new("rsync")
                    .arg("-az")
                    .arg(local_model)
                    .arg(format!("{}:{}", stage.host, remote_model)),
            )
            .with_context(|| {
                format!(
                    "rsync coordinator-materialized stage model for {} to {}",
                    stage.stage_id, stage.host
                )
            })?;
        }
    } else if args.stage_load_mode == "layer-package" && stage_model.is_dir() {
        let remote_package_dir = format!("{}/package", remote_parent(&stage.remote_config_path)?);
        run_command(
            Command::new("ssh")
                .arg(&stage.host)
                .arg(format!("mkdir -p {remote_package_dir}")),
        )?;
        let mut rsync = Command::new("rsync");
        rsync.arg("-azR");
        rsync.arg("./model-package.json");
        for path in &stage.selected_package_files {
            rsync.arg(format!("./{path}"));
        }
        rsync.arg(format!("{}:{remote_package_dir}/", stage.host));
        rsync.current_dir(stage_model);
        run_command(&mut rsync).with_context(|| {
            format!(
                "rsync selected package artifacts for {} to {}",
                stage.stage_id, stage.host
            )
        })?;
    } else {
        run_command(
            Command::new("rsync")
                .arg("-az")
                .arg(stage_model)
                .arg(format!("{}:{}/model", stage.host, args.remote_root)),
        )
        .with_context(|| format!("rsync model artifact to {}", stage.host))?;
    }
    Ok(())
}

fn place_stage_model_on_shared_root(source: &Path, destination: &Path) -> Result<()> {
    let source_metadata =
        fs::metadata(source).with_context(|| format!("read stage model {}", source.display()))?;
    if destination.is_file() {
        let destination_metadata = fs::metadata(destination)
            .with_context(|| format!("read shared stage model {}", destination.display()))?;
        if destination_metadata.len() == source_metadata.len()
            && destination_metadata.modified().with_context(|| {
                format!("read shared stage model mtime {}", destination.display())
            })? >= source_metadata
                .modified()
                .with_context(|| format!("read stage model mtime {}", source.display()))?
        {
            return Ok(());
        }
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create shared stage model dir {}", parent.display()))?;
    }
    let tmp = destination.with_extension("gguf.tmp");
    let _ = fs::remove_file(&tmp);
    match fs::hard_link(source, destination) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, &tmp).with_context(|| {
                format!("copy stage model {} to {}", source.display(), tmp.display())
            })?;
            fs::rename(&tmp, destination).with_context(|| {
                format!(
                    "move shared stage model {} to {}",
                    tmp.display(),
                    destination.display()
                )
            })?;
            Ok(())
        }
    }
}

fn materialize_stage_model_on_coordinator(
    package_dir: &Path,
    stage: &StageAssignment,
    output: &Path,
) -> Result<()> {
    let input_paths = stage
        .selected_package_files
        .iter()
        .map(|path| package_dir.join(path))
        .collect::<Vec<_>>();
    if materialized_stage_model_is_current(output, &input_paths)? {
        return Ok(());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create materialized stage model dir {}", parent.display()))?;
    }
    write_gguf_from_parts(&input_paths, output).with_context(|| {
        format!(
            "materialize {} package files for {} into {}",
            input_paths.len(),
            stage.stage_id,
            output.display()
        )
    })
}

fn materialized_stage_model_is_current(output: &Path, inputs: &[PathBuf]) -> Result<bool> {
    if !output.is_file() {
        return Ok(false);
    }
    let output_metadata = fs::metadata(output)
        .with_context(|| format!("read materialized stage model {}", output.display()))?;
    if output_metadata.len() == 0 {
        return Ok(false);
    }
    let output_modified = output_metadata
        .modified()
        .with_context(|| format!("read materialized stage model mtime {}", output.display()))?;
    for input in inputs {
        let input_modified = fs::metadata(input)
            .with_context(|| format!("read package part {}", input.display()))?
            .modified()
            .with_context(|| format!("read package part mtime {}", input.display()))?;
        if input_modified > output_modified {
            return Ok(false);
        }
    }
    Ok(true)
}

fn run_command(command: &mut Command) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("failed to spawn {:?}", command))?;
    if !status.success() {
        bail!("command failed with status {status}: {:?}", command);
    }
    Ok(())
}

fn ssh_success(host: &str, remote_command: &str) -> Result<bool> {
    let status = Command::new("ssh")
        .arg(host)
        .arg(remote_command)
        .status()
        .with_context(|| format!("run ssh command on {host}"))?;
    Ok(status.success())
}

fn ssh_capture(host: &str, remote_command: &str) -> Result<String> {
    let output = Command::new("ssh")
        .arg(host)
        .arg(remote_command)
        .output()
        .with_context(|| format!("run ssh command on {host}"))?;
    if !output.status.success() {
        bail!(
            "ssh command on {host} failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

pub(super) fn configure_child_logs(command: &mut Command, child_logs: bool) {
    if child_logs {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
}

fn metrics_otlp_grpc_url(args: &RunArgs) -> String {
    args.metrics_otlp_grpc_url
        .clone()
        .unwrap_or_else(|| format!("http://{}", args.metrics_otlp_grpc_addr))
}

fn coordinator_materializes_layer_package(args: &RunArgs) -> bool {
    args.execute_remote
        && args.rsync_model_artifacts
        && args.stage_load_mode == "layer-package"
        && args
            .stage_model
            .as_ref()
            .map(|path| path.is_dir())
            .unwrap_or(false)
}

fn parse_remote_root_map(value: Option<&str>) -> Result<BTreeMap<String, String>> {
    let mut roots = BTreeMap::new();
    let Some(value) = value else {
        return Ok(roots);
    };
    for entry in value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (host, root) = entry.split_once('=').with_context(|| {
            format!("invalid remote root mapping {entry:?}; expected host=/path")
        })?;
        let host = host.trim();
        let root = root.trim().trim_end_matches('/');
        if host.is_empty() || root.is_empty() {
            bail!("invalid remote root mapping {entry:?}; expected host=/path");
        }
        roots.insert(host.to_string(), root.to_string());
    }
    Ok(roots)
}

fn parse_path_map(value: Option<&str>) -> Result<BTreeMap<String, PathBuf>> {
    Ok(parse_remote_root_map(value)?
        .into_iter()
        .map(|(host, path)| (host, PathBuf::from(path)))
        .collect())
}

pub(super) fn model_ref_for_configs(args: &RunArgs) -> Result<String> {
    match args.stage_load_mode.as_str() {
        "runtime-slice" => args
            .model_path
            .as_ref()
            .map(|path| path_string(path))
            .context("--model-path is required when --stage-load-mode runtime-slice"),
        "artifact-slice" | "layer-package" => {
            let stage_model = args.stage_model.as_ref().with_context(|| {
                format!(
                    "--stage-model is required when --stage-load-mode {}",
                    args.stage_load_mode
                )
            })?;
            Ok(path_string(stage_model))
        }
        other => bail!(
            "unsupported --stage-load-mode {other}; expected runtime-slice, artifact-slice, or layer-package"
        ),
    }
}

pub(super) fn parse_hosts(hosts: &str) -> Result<Vec<String>> {
    let parsed = hosts
        .split(',')
        .map(str::trim)
        .filter(|host| !host.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if parsed.is_empty() {
        bail!("--hosts must contain at least one host");
    }
    let unique = parsed.iter().collect::<BTreeSet<_>>();
    if unique.len() != parsed.len() {
        bail!("--hosts must not repeat hosts; each stage must run on a separate node");
    }
    Ok(parsed)
}

pub(super) fn validate_distinct_stage_hosts(hosts: &[String], stage_count: usize) -> Result<()> {
    if hosts.len() < stage_count {
        bail!(
            "--hosts supplies {} unique hosts for {stage_count} stages; provide one separate node per stage",
            hosts.len()
        );
    }
    Ok(())
}

pub(super) fn validate_balanced_stage_ranges(ranges: &[(u32, u32)]) -> Result<()> {
    let Some(first) = ranges.first() else {
        bail!("at least one stage range is required");
    };
    let mut min_len = first.1 - first.0;
    let mut max_len = min_len;
    let mut lengths = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        let len = end - start;
        lengths.push(len);
        min_len = min_len.min(len);
        max_len = max_len.max(len);
    }
    if max_len - min_len > 1 {
        bail!(
            "stage layer ranges must be evenly balanced across nodes; got lengths {:?}",
            lengths
        );
    }
    Ok(())
}

pub(super) fn parse_stage_ranges(splits: &str, layer_end: u32) -> Result<Vec<(u32, u32)>> {
    let mut boundaries = vec![0];
    for split in splits
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        boundaries.push(
            split
                .parse::<u32>()
                .with_context(|| format!("invalid split {split}"))?,
        );
    }
    boundaries.push(layer_end);
    for pair in boundaries.windows(2) {
        if pair[0] >= pair[1] {
            bail!("splits must be strictly ascending and less than layer_end");
        }
    }
    Ok(boundaries
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect())
}

fn load_package_manifest(package_dir: &Path) -> Result<PackageManifest> {
    let path = package_dir.join("model-package.json");
    let contents = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("parse {}", path.display()))
}

fn selected_package_files(
    manifest: &PackageManifest,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<Vec<String>> {
    let mut files = vec![manifest.shared.metadata.path.clone()];
    if include_embeddings {
        files.push(manifest.shared.embeddings.path.clone());
    }
    for layer_index in layer_start..layer_end {
        let layer = manifest
            .layers
            .iter()
            .find(|layer| layer.layer_index == layer_index)
            .with_context(|| format!("package manifest is missing layer {layer_index}"))?;
        files.push(layer.path.clone());
    }
    if include_output {
        files.push(manifest.shared.output.path.clone());
    }
    Ok(files)
}

fn stage_model_cache_key(
    args: &RunArgs,
    stage_id: &str,
    layer_start: u32,
    layer_end: u32,
) -> PathBuf {
    PathBuf::from(safe_cache_component(&args.model_id))
        .join(safe_cache_component(&args.topology_id))
        .join(format!(
            "{}-{}-{}",
            safe_cache_component(stage_id),
            layer_start,
            layer_end
        ))
}

fn safe_cache_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn remote_parent(path: &str) -> Result<String> {
    path.rsplit_once('/')
        .map(|(parent, _)| parent.to_string())
        .context("remote path has no parent")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lab_hosts() {
        assert_eq!(
            parse_hosts("shadowfax.local,black.local,studio54.local,build.local").unwrap(),
            vec![
                "shadowfax.local",
                "black.local",
                "studio54.local",
                "build.local"
            ]
        );
    }

    #[test]
    fn rejects_duplicate_stage_hosts() {
        assert!(parse_hosts("shadowfax.local,black.local,shadowfax.local").is_err());
    }

    #[test]
    fn requires_one_host_per_stage() {
        let hosts = parse_hosts("shadowfax.local,black.local").unwrap();
        assert!(validate_distinct_stage_hosts(&hosts, 3).is_err());
        assert!(validate_distinct_stage_hosts(&hosts, 2).is_ok());
    }

    #[test]
    fn builds_stable_stage_model_cache_key() {
        let args = RunArgs {
            metrics_server_bin: PathBuf::from("metrics-server"),
            stage_server_bin: PathBuf::from("skippy-server"),
            hosts: "shadowfax.local,black.local".to_string(),
            run_id: Some("run-1".to_string()),
            topology_id: "quad/small".to_string(),
            model_id: "Qwen/Qwen3-4B:Q4_K_M".to_string(),
            model_path: None,
            stage_model: Some(PathBuf::from("model-package")),
            stage_load_mode: "layer-package".to_string(),
            splits: "1".to_string(),
            layer_end: 2,
            ctx_size: 128,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            activation_width: 2048,
            activation_wire_dtype: "f32".to_string(),
            prompt: "Hello".to_string(),
            prompt_corpus: None,
            prompt_limit: None,
            prompt_token_ids: None,
            max_new_tokens: Some(1),
            prefill_chunk_size: None,
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            metrics_http_addr: "127.0.0.1:18080".parse().unwrap(),
            metrics_otlp_grpc_addr: "127.0.0.1:14317".parse().unwrap(),
            metrics_otlp_grpc_url: None,
            db: None,
            output: None,
            work_dir: PathBuf::from("/tmp/work"),
            remote_root: "/tmp/remote".to_string(),
            remote_root_map: None,
            remote_shared_root_map: None,
            endpoint_host_map: None,
            remote_bind_host: "0.0.0.0".to_string(),
            first_stage_port: 19031,
            execute_remote: false,
            keep_remote: false,
            rsync_model_artifacts: true,
            child_logs: false,
            startup_timeout_secs: 60,
            stage_max_inflight: 4,
            stage_reply_credit_limit: None,
            stage_async_prefill_forward: false,
            stage_downstream_wire_delay_ms: 0.0,
            stage_downstream_wire_mbps: None,
            stage_telemetry_queue_capacity: 8192,
            stage_telemetry_level: "summary".to_string(),
        };

        assert_eq!(
            stage_model_cache_key(&args, "stage-0", 0, 1),
            PathBuf::from("Qwen_Qwen3-4B_Q4_K_M")
                .join("quad_small")
                .join("stage-0-0-1")
        );
    }

    #[test]
    fn builds_stage_ranges_from_splits() {
        assert_eq!(
            parse_stage_ranges("1,4", 40).unwrap(),
            vec![(0, 1), (1, 4), (4, 40)]
        );
        assert!(parse_stage_ranges("4,1", 40).is_err());
        assert!(parse_stage_ranges("1,40", 40).is_err());
    }

    #[test]
    fn requires_balanced_stage_ranges() {
        validate_balanced_stage_ranges(&parse_stage_ranges("14,27", 40).unwrap()).unwrap();
        validate_balanced_stage_ranges(&parse_stage_ranges("13,27", 40).unwrap()).unwrap();
        assert!(
            validate_balanced_stage_ranges(&parse_stage_ranges("12,20,28", 40).unwrap()).is_err()
        );
        assert!(validate_balanced_stage_ranges(&parse_stage_ranges("1,4,7", 40).unwrap()).is_err());
    }

    #[test]
    fn selects_minimal_package_files() {
        let manifest = PackageManifest {
            model_id: "org/repo:Q4_K_M".to_string(),
            source_model: PackageSourceModel {
                repo: Some("org/repo".to_string()),
                revision: Some("abc123".to_string()),
                primary_file: Some("Model-Q4_K_M.gguf".to_string()),
                canonical_ref: Some("org/repo@abc123/Model-Q4_K_M.gguf".to_string()),
                distribution_id: Some("Model-Q4_K_M".to_string()),
            },
            shared: PackageShared {
                metadata: PackageArtifact {
                    path: "shared/metadata.gguf".to_string(),
                },
                embeddings: PackageArtifact {
                    path: "shared/embeddings.gguf".to_string(),
                },
                output: PackageArtifact {
                    path: "shared/output.gguf".to_string(),
                },
            },
            layers: vec![
                PackageLayer {
                    layer_index: 0,
                    path: "layers/layer-000.gguf".to_string(),
                },
                PackageLayer {
                    layer_index: 1,
                    path: "layers/layer-001.gguf".to_string(),
                },
            ],
        };
        let files = selected_package_files(&manifest, 0, 1, true, false).unwrap();
        assert_eq!(
            files,
            vec![
                "shared/metadata.gguf",
                "shared/embeddings.gguf",
                "layers/layer-000.gguf"
            ]
        );
    }

    #[test]
    fn reads_model_identity_from_package_manifest() {
        let manifest = PackageManifest {
            model_id: "org/repo:Q4_K_M".to_string(),
            source_model: PackageSourceModel {
                repo: Some("org/repo".to_string()),
                revision: Some("abc123".to_string()),
                primary_file: Some("Model-Q4_K_M.gguf".to_string()),
                canonical_ref: Some("org/repo@abc123/Model-Q4_K_M.gguf".to_string()),
                distribution_id: Some("Model-Q4_K_M".to_string()),
            },
            shared: PackageShared {
                metadata: PackageArtifact {
                    path: "shared/metadata.gguf".to_string(),
                },
                embeddings: PackageArtifact {
                    path: "shared/embeddings.gguf".to_string(),
                },
                output: PackageArtifact {
                    path: "shared/output.gguf".to_string(),
                },
            },
            layers: Vec::new(),
        };

        let identity = model_identity_from_package_manifest(&manifest).unwrap();
        assert_eq!(identity.model_id, "org/repo:Q4_K_M");
        assert_eq!(identity.source_repo.as_deref(), Some("org/repo"));
        assert_eq!(identity.source_revision.as_deref(), Some("abc123"));
        assert_eq!(identity.source_file.as_deref(), Some("Model-Q4_K_M.gguf"));
        assert_eq!(identity.selector.as_deref(), Some("Q4_K_M"));
    }
    #[test]
    fn parses_remote_root_map() {
        let roots = parse_remote_root_map(Some(
            "build.local=/Users/jdumay/models/bench, black.local=/tmp/bench",
        ))
        .unwrap();
        assert_eq!(
            roots.get("build.local").map(String::as_str),
            Some("/Users/jdumay/models/bench")
        );
        assert_eq!(
            roots.get("black.local").map(String::as_str),
            Some("/tmp/bench")
        );
        assert!(parse_remote_root_map(Some("build.local")).is_err());
    }

    #[test]
    fn remote_start_command_records_exit_code() {
        let args = RunArgs {
            metrics_server_bin: PathBuf::from("metrics-server"),
            stage_server_bin: PathBuf::from("skippy-server"),
            hosts: "host.local".to_string(),
            run_id: Some("run-1".to_string()),
            topology_id: "topology".to_string(),
            model_id: "test-org/bench-model-GGUF:Q4_K_M".to_string(),
            model_path: Some(PathBuf::from("model.gguf")),
            stage_model: None,
            stage_load_mode: "runtime-slice".to_string(),
            splits: "1".to_string(),
            layer_end: 2,
            ctx_size: 128,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            activation_width: 2048,
            activation_wire_dtype: "f32".to_string(),
            prompt: "Hello".to_string(),
            prompt_corpus: None,
            prompt_limit: None,
            prompt_token_ids: None,
            max_new_tokens: Some(1),
            prefill_chunk_size: None,
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            metrics_http_addr: "127.0.0.1:18080".parse().unwrap(),
            metrics_otlp_grpc_addr: "127.0.0.1:14317".parse().unwrap(),
            metrics_otlp_grpc_url: Some("http://coordinator.local:14317".to_string()),
            db: None,
            output: None,
            work_dir: PathBuf::from("/tmp/work"),
            remote_root: "/tmp/remote".to_string(),
            remote_root_map: None,
            remote_shared_root_map: None,
            endpoint_host_map: None,
            remote_bind_host: "0.0.0.0".to_string(),
            first_stage_port: 19031,
            execute_remote: true,
            keep_remote: false,
            rsync_model_artifacts: false,
            child_logs: false,
            startup_timeout_secs: 60,
            stage_max_inflight: 4,
            stage_reply_credit_limit: Some(2),
            stage_async_prefill_forward: true,
            stage_downstream_wire_delay_ms: 1.0,
            stage_downstream_wire_mbps: Some(1000.0),
            stage_telemetry_queue_capacity: 8192,
            stage_telemetry_level: "summary".to_string(),
        };
        let plan = DeploymentPlan {
            run_id: "run-1".to_string(),
            topology_id: "topology".to_string(),
            model_id: "test-org/bench-model-GGUF:Q4_K_M".to_string(),
            model_identity: ModelIdentity::from_model_id("test-org/bench-model-GGUF:Q4_K_M"),
            hosts: vec!["host.local".to_string()],
            stage_load_mode: "runtime-slice".to_string(),
            remote_root: "/tmp/remote".to_string(),
            remote_roots: BTreeMap::new(),
            remote_shared_roots: BTreeMap::new(),
            endpoint_hosts: BTreeMap::new(),
            work_dir: PathBuf::from("/tmp/work"),
            metrics_http: "http://127.0.0.1:18080".to_string(),
            metrics_otlp_grpc: "http://coordinator.local:14317".to_string(),
            driver_return_endpoint: "host.local:20031".to_string(),
            stages: Vec::new(),
            execute_remote: true,
            keep_remote: false,
            rsync_model_artifacts: false,
        };
        let stage = StageAssignment {
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            host: "host.local".to_string(),
            local: false,
            layer_start: 0,
            layer_end: 1,
            bind_addr: "0.0.0.0:19031".to_string(),
            endpoint: "tcp://host.local:19031".to_string(),
            config_path: PathBuf::from("/tmp/local/stage.json"),
            remote_config_path: "/tmp/remote/run-1/stage-0/stage.json".to_string(),
            remote_log_path: "/tmp/remote/run-1/stage-0/stage.log".to_string(),
            remote_pid_path: "/tmp/remote/run-1/stage-0/stage.pid".to_string(),
            remote_exit_code_path: "/tmp/remote/run-1/stage-0/stage.exit".to_string(),
            remote_model_path: None,
            local_materialized_model_path: None,
            local_shared_model_path: None,
            selected_package_files: Vec::new(),
        };
        let command = remote_start_command(
            &args,
            &plan,
            &stage,
            "/tmp/remote/run-1/stage-0/skippy-server",
        );
        assert!(command.contains("stage.exit"));
        assert!(command.contains("stage.pid"));
        assert!(command.contains("wait \"$child\""));
        assert!(!command.contains("nohup"));
        assert!(command.contains("--metrics-otlp-grpc"));
        assert!(command.contains("coordinator.local:14317"));
        assert!(command.contains("--max-inflight 4"));
        assert!(command.contains("--reply-credit-limit 2"));
        assert!(command.contains("--async-prefill-forward"));
        assert!(command.contains("--downstream-wire-delay-ms 1"));
        assert!(command.contains("--downstream-wire-mbps 1000"));
        assert!(command.contains("--telemetry-level"));
        assert!(command.contains("summary"));
    }
    #[test]
    fn planner_rejects_qwen_q8_before_launch() {
        let mut args = test_run_args();
        args.model_id = "Qwen/Qwen3-0.6B".to_string();
        args.hosts = "host-a,host-b".to_string();
        args.splits = "14".to_string();
        args.layer_end = 28;
        args.activation_width = 1024;
        args.activation_wire_dtype = "q8".to_string();
        let hosts = parse_hosts(&args.hosts).unwrap();
        let ranges = parse_stage_ranges(&args.splits, args.layer_end).unwrap();

        let err = validate_topology_plan(&args, &hosts, &ranges).unwrap_err();

        assert!(err.to_string().contains("rejected q8"));
    }

    #[test]
    fn planner_rejects_gemma_known_bad_split_before_launch() {
        let mut args = test_run_args();
        args.model_id = "gemma-4-e4b".to_string();
        args.hosts = "host-a,host-b,host-c".to_string();
        args.splits = "14,28".to_string();
        args.layer_end = 42;
        args.activation_width = 2560;
        args.activation_wire_dtype = "f16".to_string();
        let hosts = parse_hosts(&args.hosts).unwrap();
        let ranges = parse_stage_ranges(&args.splits, args.layer_end).unwrap();

        let err = validate_topology_plan(&args, &hosts, &ranges).unwrap_err();

        assert!(err.to_string().contains("SharedKvRegionCut"));
    }

    #[test]
    fn planner_accepts_gemma_validated_split() {
        let mut args = test_run_args();
        args.model_id = "gemma-4-e4b".to_string();
        args.hosts = "host-a,host-b".to_string();
        args.splits = "21".to_string();
        args.layer_end = 42;
        args.activation_width = 2560;
        args.activation_wire_dtype = "f16".to_string();
        let hosts = parse_hosts(&args.hosts).unwrap();
        let ranges = parse_stage_ranges(&args.splits, args.layer_end).unwrap();

        validate_topology_plan(&args, &hosts, &ranges).unwrap();
    }

    fn test_run_args() -> RunArgs {
        RunArgs {
            metrics_server_bin: PathBuf::from("metrics-server"),
            stage_server_bin: PathBuf::from("skippy-server"),
            hosts: "host.local".to_string(),
            run_id: Some("run-1".to_string()),
            topology_id: "topology".to_string(),
            model_id: "test-org/bench-model-GGUF:Q4_K_M".to_string(),
            model_path: Some(PathBuf::from("model.gguf")),
            stage_model: None,
            stage_load_mode: "runtime-slice".to_string(),
            splits: "1".to_string(),
            layer_end: 2,
            ctx_size: 128,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            activation_width: 2048,
            activation_wire_dtype: "f32".to_string(),
            prompt: "Hello".to_string(),
            prompt_corpus: None,
            prompt_limit: None,
            prompt_token_ids: None,
            max_new_tokens: None,
            prefill_chunk_size: None,
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            metrics_http_addr: "127.0.0.1:18080".parse().unwrap(),
            metrics_otlp_grpc_addr: "127.0.0.1:14317".parse().unwrap(),
            metrics_otlp_grpc_url: None,
            db: None,
            output: None,
            work_dir: PathBuf::from("/tmp/work"),
            remote_root: "/tmp/remote".to_string(),
            remote_root_map: None,
            remote_shared_root_map: None,
            endpoint_host_map: None,
            remote_bind_host: "0.0.0.0".to_string(),
            first_stage_port: 19031,
            execute_remote: false,
            keep_remote: false,
            rsync_model_artifacts: false,
            child_logs: false,
            startup_timeout_secs: 60,
            stage_max_inflight: 4,
            stage_reply_credit_limit: None,
            stage_async_prefill_forward: false,
            stage_downstream_wire_delay_ms: 0.0,
            stage_downstream_wire_mbps: None,
            stage_telemetry_queue_capacity: 8192,
            stage_telemetry_level: "summary".to_string(),
        }
    }
}
