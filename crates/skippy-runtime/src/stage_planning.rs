use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr;

use anyhow::{Context, Result};
use model_ref::split_gguf_shard_info;
use sha2::{Digest, Sha256};

use crate::{ModelInfo, RuntimeConfig, ensure_ok};

struct Planner(*mut skippy_ffi::StagePlanner);

impl Drop for Planner {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { skippy_ffi::skippy_stage_planner_free(self.0) };
        }
    }
}

struct Plan(*mut skippy_ffi::StagePlan);

impl Drop for Plan {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { skippy_ffi::skippy_stage_plan_free(self.0) };
        }
    }
}

struct TensorDescriptor {
    name: String,
    ggml_type: u32,
    dimensions: Vec<u64>,
    split_no: u32,
    data_offset: u64,
    stored_length: u64,
}

/// Native planner output required to load and execute one filtered stage.
///
/// The resident tensor names define the GGUF storage closure. The activation
/// identities and bindings define the live graph frontier used by the typed
/// activation transport. They must be carried together from the same plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufStageRuntimePlan {
    pub resident_tensor_names: Vec<String>,
    pub activation_import_identities: Vec<String>,
    pub activation_import_bindings: Vec<String>,
    pub activation_export_identities: Vec<String>,
    pub activation_export_bindings: Vec<String>,
}

impl GgufStageRuntimePlan {
    /// Install this planner result into the runtime configuration it describes.
    pub fn apply_to(self, config: &mut RuntimeConfig) {
        config.resident_tensor_names = self.resident_tensor_names;
        config.activation_import_identities = self.activation_import_identities;
        config.activation_import_bindings = self.activation_import_bindings;
        config.activation_export_identities = self.activation_export_identities;
        config.activation_export_bindings = self.activation_export_bindings;
    }
}

/// Derive each stage's exact resident tensor closure with the native graph
/// planner. This is the same source-of-truth planner used by production stage
/// admission; callers receive native GGUF names suitable for
/// [`crate::RuntimeConfig::resident_tensor_names`].
pub fn plan_gguf_stage_resident_tensor_names(
    model_path: &Path,
    ranges: &[(u32, u32)],
    ctx_size: u32,
    lane_count: u32,
) -> Result<Vec<Vec<String>>> {
    Ok(
        plan_gguf_stage_runtime_plans(model_path, ranges, ctx_size, lane_count)?
            .into_iter()
            .map(|plan| plan.resident_tensor_names)
            .collect(),
    )
}

/// Derive each stage's exact filtered-load closure and activation frontier.
///
/// All guarded execution profiles must expose the same frontier identities and
/// live tensor bindings. A disagreement fails planning before any model loads.
pub fn plan_gguf_stage_runtime_plans(
    model_path: &Path,
    ranges: &[(u32, u32)],
    ctx_size: u32,
    lane_count: u32,
) -> Result<Vec<GgufStageRuntimePlan>> {
    plan_gguf_stage_runtime_plans_impl(model_path, ranges, ctx_size, lane_count, true)
}

/// Derive the exact resident tensor closure for one independently loaded slice.
///
/// This does not require the requested range to form a complete model partition.
pub fn plan_gguf_stage_resident_tensor_names_for_range(
    model_path: &Path,
    range: (u32, u32),
    ctx_size: u32,
    lane_count: u32,
) -> Result<Vec<String>> {
    Ok(
        plan_gguf_stage_runtime_plan_for_range(model_path, range, ctx_size, lane_count)?
            .resident_tensor_names,
    )
}

/// Derive one independently loaded stage's filtered-load closure and frontier.
///
/// This does not require the requested range to form a complete model partition.
pub fn plan_gguf_stage_runtime_plan_for_range(
    model_path: &Path,
    range: (u32, u32),
    ctx_size: u32,
    lane_count: u32,
) -> Result<GgufStageRuntimePlan> {
    plan_gguf_stage_runtime_plans_impl(model_path, &[range], ctx_size, lane_count, false)?
        .into_iter()
        .next()
        .context("native stage planner returned no resident tensor closure")
}

fn plan_gguf_stage_runtime_plans_impl(
    model_path: &Path,
    ranges: &[(u32, u32)],
    ctx_size: u32,
    lane_count: u32,
    validate_chain: bool,
) -> Result<Vec<GgufStageRuntimePlan>> {
    anyhow::ensure!(!ranges.is_empty(), "stage plan chain is empty");
    anyhow::ensure!(ctx_size > 0, "stage planning context size must be positive");
    anyhow::ensure!(lane_count > 0, "stage planning lane count must be positive");

    let shard_paths = gguf_shard_paths(model_path)?;
    let tensors = tensor_descriptors(&shard_paths)?;
    anyhow::ensure!(!tensors.is_empty(), "GGUF tensor inventory is empty");

    let package_id = package_id(&shard_paths, &tensors);
    let package_id = CString::new(package_id).expect("package identity contains no NUL");
    let shard_paths = shard_paths
        .iter()
        .map(|path| {
            CString::new(path.to_string_lossy().as_bytes()).with_context(|| {
                format!(
                    "GGUF shard path contains an interior NUL: {}",
                    path.display()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let shard_path_ptrs = shard_paths
        .iter()
        .map(|path| path.as_ptr())
        .collect::<Vec<_>>();
    let tensor_names = tensors
        .iter()
        .map(|tensor| {
            CString::new(tensor.name.as_bytes()).context("tensor name contains an interior NUL")
        })
        .collect::<Result<Vec<_>>>()?;
    let raw_tensors = tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            let mut dimensions = [0_i64; skippy_ffi::STAGE_PLAN_MAX_DIMS];
            anyhow::ensure!(
                !tensor.dimensions.is_empty()
                    && tensor.dimensions.len() <= skippy_ffi::STAGE_PLAN_MAX_DIMS,
                "tensor {:?} has unsupported rank {}",
                tensor.name,
                tensor.dimensions.len()
            );
            for (destination, source) in dimensions.iter_mut().zip(&tensor.dimensions) {
                *destination = i64::try_from(*source)
                    .with_context(|| format!("tensor {:?} dimension exceeds i64", tensor.name))?;
                anyhow::ensure!(
                    *destination > 0,
                    "tensor {:?} has an empty dimension",
                    tensor.name
                );
            }
            Ok(skippy_ffi::StagePlannerTensorV1 {
                abi_version: skippy_ffi::STAGE_PLANNER_TENSOR_V1_ABI_VERSION,
                struct_size: u32::try_from(std::mem::size_of::<skippy_ffi::StagePlannerTensorV1>())
                    .expect("stage planner tensor descriptor size fits u32"),
                tensor_id: tensor_names[index].as_ptr(),
                native_name: tensor_names[index].as_ptr(),
                ggml_type: i32::try_from(tensor.ggml_type)
                    .context("GGML tensor type exceeds i32")?,
                dimension_count: u32::try_from(tensor.dimensions.len())
                    .expect("validated tensor rank fits u32"),
                dimensions,
                split_no: tensor.split_no,
                reserved: 0,
                data_offset: tensor.data_offset,
                stored_length: tensor.stored_length,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let batched_tokens = lane_count
        .checked_mul(8)
        .context("batched stage planning token count overflow")?;
    let profile_specs = [
        ("batched", batched_tokens, lane_count, batched_tokens),
        ("decode", lane_count, lane_count, lane_count),
        ("prefill", 8, 1, 8),
    ];
    let profile_names = profile_specs
        .iter()
        .map(|(name, ..)| CString::new(*name).expect("fixed profile ID contains no NUL"))
        .collect::<Vec<_>>();
    let profiles = profile_specs
        .iter()
        .enumerate()
        .map(
            |(index, (_, n_tokens, n_sequences, n_outputs))| skippy_ffi::StagePlannerProfileV1 {
                abi_version: skippy_ffi::STAGE_PLANNER_PROFILE_V1_ABI_VERSION,
                struct_size:
                    u32::try_from(std::mem::size_of::<skippy_ffi::StagePlannerProfileV1>())
                        .expect("stage planner profile descriptor size fits u32"),
                profile_id: profile_names[index].as_ptr(),
                n_tokens: *n_tokens,
                n_sequences: *n_sequences,
                n_outputs: *n_outputs,
                n_recurrent_rollback_sequences: 0,
            },
        )
        .collect::<Vec<_>>();

    let mut graph_identity = Sha256::new();
    graph_identity.update(b"skippy-correctness-graph-configuration:v1\0");
    graph_identity.update(ctx_size.to_le_bytes());
    graph_identity.update(lane_count.to_le_bytes());
    let graph_configuration_id = CString::new(format!(
        "skippy-graph-configuration:v1:{}",
        hex::encode(graph_identity.finalize())
    ))
    .expect("graph configuration identity contains no NUL");
    let backend_id = CString::new("correctness").expect("fixed backend ID contains no NUL");
    let config = skippy_ffi::StagePlannerConfigV1 {
        abi_version: skippy_ffi::STAGE_PLANNER_CONFIG_V1_ABI_VERSION,
        struct_size: u32::try_from(std::mem::size_of::<skippy_ffi::StagePlannerConfigV1>())
            .expect("stage planner config size fits u32"),
        package_id: package_id.as_ptr(),
        shard_paths: shard_path_ptrs.as_ptr(),
        shard_count: shard_path_ptrs.len(),
        tensors: raw_tensors.as_ptr(),
        tensor_count: raw_tensors.len(),
        profiles: profiles.as_ptr(),
        profile_count: profiles.len(),
        graph_configuration_id: graph_configuration_id.as_ptr(),
        backend_id: backend_id.as_ptr(),
    };
    let mut planner = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status =
        unsafe { skippy_ffi::skippy_stage_planner_create_v1(&config, &mut planner, &mut error) };
    if let Err(error) = ensure_ok(status, error).context("create native stage planner") {
        if !planner.is_null() {
            unsafe { skippy_ffi::skippy_stage_planner_free(planner) };
        }
        return Err(error);
    }
    anyhow::ensure!(
        !planner.is_null(),
        "native stage planner returned a null handle"
    );
    let planner = Planner(planner);

    let plans = ranges
        .iter()
        .map(|(layer_start, layer_end)| realize_plan(&planner, *layer_start, *layer_end))
        .collect::<Result<Vec<_>>>()?;
    if validate_chain {
        let plan_ptrs = plans
            .iter()
            .map(|plan| plan.0.cast_const())
            .collect::<Vec<_>>();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_stage_plan_validate_chain_v1(
                plan_ptrs.as_ptr(),
                plan_ptrs.len(),
                &mut error,
            )
        };
        ensure_ok(status, error).context("validate native stage plan chain")?;
    }

    plans
        .iter()
        .enumerate()
        .map(|(index, plan)| {
            runtime_plan(plan).with_context(|| format!("read stage {index} runtime plan"))
        })
        .collect()
}

pub fn gguf_shard_paths(model_path: &Path) -> Result<Vec<PathBuf>> {
    let Some(file_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        anyhow::bail!("GGUF path has no UTF-8 filename: {}", model_path.display());
    };
    let Some(shard) = split_gguf_shard_info(file_name) else {
        let canonical = model_path
            .canonicalize()
            .with_context(|| format!("canonicalize GGUF path {}", model_path.display()))?;
        return Ok(vec![canonical]);
    };
    anyhow::ensure!(
        shard.part == "00001",
        "split GGUF inputs must point at the first shard, got {}",
        model_path.display()
    );
    let total = shard
        .total
        .parse::<u32>()
        .context("parse split GGUF shard count")?;
    anyhow::ensure!(total > 0, "split GGUF shard count must be positive");
    // Resolve siblings in the referenced directory: HF caches expose snapshot
    // files as per-file symlinks into blobs/, so canonicalizing the input
    // first would erase the shard name pattern.
    let directory = model_path
        .parent()
        .map(|parent| parent.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    (1..=total)
        .map(|index| {
            directory
                .join(format!("{}-{index:05}-of-{:05}.gguf", shard.prefix, total))
                .canonicalize()
                .with_context(|| format!("resolve split GGUF shard {index}/{total}"))
        })
        .collect()
}

fn tensor_descriptors(shard_paths: &[PathBuf]) -> Result<Vec<TensorDescriptor>> {
    let mut tensors = BTreeMap::new();
    for (split_no, path) in shard_paths.iter().enumerate() {
        let catalog = skippy_model::gguf_catalog::read_gguf_tensor_catalog(path)
            .with_context(|| format!("read GGUF catalog {}", path.display()))?;
        let native = ModelInfo::open(path)
            .with_context(|| format!("open native GGUF inventory {}", path.display()))?
            .tensors()
            .with_context(|| format!("read native GGUF inventory {}", path.display()))?;
        let native = native
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<BTreeMap<_, _>>();
        anyhow::ensure!(
            native.len() == catalog.tensors.len(),
            "native and GGUF tensor inventories differ for {}",
            path.display()
        );
        for tensor in catalog.tensors {
            let info = native
                .get(&tensor.name)
                .with_context(|| format!("native inventory is missing tensor {:?}", tensor.name))?;
            let elements = tensor
                .dimensions
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .context("GGUF tensor element count overflow")?;
            anyhow::ensure!(
                info.ggml_type == tensor.ggml_type && info.element_count == elements,
                "native and GGUF metadata disagree for {:?}",
                tensor.name
            );
            anyhow::ensure!(
                info.byte_size > 0,
                "tensor {:?} has empty storage",
                tensor.name
            );
            let end = tensor
                .data_offset
                .checked_add(info.byte_size)
                .context("GGUF tensor extent overflow")?;
            anyhow::ensure!(
                end <= catalog.artifact_bytes,
                "tensor {:?} storage exceeds its shard",
                tensor.name
            );
            let descriptor = TensorDescriptor {
                name: tensor.name.clone(),
                ggml_type: tensor.ggml_type,
                dimensions: tensor.dimensions,
                split_no: u32::try_from(split_no).context("GGUF shard index exceeds u32")?,
                data_offset: tensor.data_offset,
                stored_length: info.byte_size,
            };
            anyhow::ensure!(
                tensors.insert(tensor.name.clone(), descriptor).is_none(),
                "duplicate tensor {:?} across GGUF shards",
                tensor.name
            );
        }
    }
    Ok(tensors.into_values().collect())
}

fn package_id(shard_paths: &[PathBuf], tensors: &[TensorDescriptor]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"skippy-correctness-package:v1\0");
    for (index, path) in shard_paths.iter().enumerate() {
        hash.update((index as u64).to_le_bytes());
        let file_name = path.file_name().unwrap_or_default().to_string_lossy();
        hash_len_prefixed(&mut hash, file_name.as_bytes());
    }
    for tensor in tensors {
        hash_len_prefixed(&mut hash, tensor.name.as_bytes());
        hash.update(tensor.ggml_type.to_le_bytes());
        hash.update(tensor.split_no.to_le_bytes());
        hash.update(tensor.data_offset.to_le_bytes());
        hash.update(tensor.stored_length.to_le_bytes());
        hash.update((tensor.dimensions.len() as u64).to_le_bytes());
        for dimension in &tensor.dimensions {
            hash.update(dimension.to_le_bytes());
        }
    }
    format!("sha256:{}", hex::encode(hash.finalize()))
}

fn hash_len_prefixed(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

fn realize_plan(planner: &Planner, layer_start: u32, layer_end: u32) -> Result<Plan> {
    let mut plan = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_planner_realize_v1(
            planner.0,
            i32::try_from(layer_start).context("stage layer start exceeds i32")?,
            i32::try_from(layer_end).context("stage layer end exceeds i32")?,
            &mut plan,
            &mut error,
        )
    };
    if let Err(error) = ensure_ok(status, error).context("realize native stage plan") {
        if !plan.is_null() {
            unsafe { skippy_ffi::skippy_stage_plan_free(plan) };
        }
        return Err(error);
    }
    anyhow::ensure!(!plan.is_null(), "native stage planner returned a null plan");
    Ok(Plan(plan))
}

fn resident_tensor_names(plan: &Plan) -> Result<Vec<String>> {
    let descriptor = plan_descriptor(plan)?;
    let count = usize::try_from(descriptor.resident_tensor_count)
        .context("native resident tensor count exceeds usize")?;
    anyhow::ensure!(count > 0, "native stage plan has no resident tensors");
    let mut names = Vec::with_capacity(count);
    for index in 0..count {
        let mut value = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanValueDescV1>() };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_stage_plan_resident_tensor_at_v1(
                plan.0, index, &mut value, &mut error,
            )
        };
        ensure_ok(status, error).with_context(|| format!("read native resident tensor {index}"))?;
        anyhow::ensure!(
            value.abi_version == skippy_ffi::STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION
                && usize::try_from(value.struct_size).ok()
                    == Some(std::mem::size_of::<skippy_ffi::StagePlanValueDescV1>()),
            "native resident tensor descriptor ABI mismatch"
        );
        names.push(read_plan_string(plan.0, value.identity)?);
    }
    anyhow::ensure!(
        names.windows(2).all(|window| window[0] < window[1]),
        "native resident tensor names are not strictly sorted and unique"
    );
    Ok(names)
}

fn runtime_plan(plan: &Plan) -> Result<GgufStageRuntimePlan> {
    let descriptor = plan_descriptor(plan)?;
    let profile_count = usize::try_from(descriptor.profile_count)
        .context("native stage profile count exceeds usize")?;
    anyhow::ensure!(profile_count > 0, "native stage plan has no profiles");

    let mut frontier = None;
    for profile_index in 0..profile_count {
        let current = read_profile_frontier(plan, profile_index)?;
        if let Some(expected) = &frontier {
            anyhow::ensure!(
                expected == &current,
                "native stage execution profile {profile_index} disagrees on activation frontier identities and bindings: expected {expected:?}, got {current:?}"
            );
        } else {
            frontier = Some(current);
        }
    }
    let (
        activation_import_identities,
        activation_import_bindings,
        activation_export_identities,
        activation_export_bindings,
    ) = frontier.expect("validated nonempty native stage profile set");
    Ok(GgufStageRuntimePlan {
        resident_tensor_names: resident_tensor_names(plan)?,
        activation_import_identities,
        activation_import_bindings,
        activation_export_identities,
        activation_export_bindings,
    })
}

fn plan_descriptor(plan: &Plan) -> Result<skippy_ffi::StagePlanDescV1> {
    let mut descriptor = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanDescV1>() };
    let mut error = ptr::null_mut();
    let status =
        unsafe { skippy_ffi::skippy_stage_plan_describe_v1(plan.0, &mut descriptor, &mut error) };
    ensure_ok(status, error).context("describe native stage plan")?;
    anyhow::ensure!(
        descriptor.abi_version == skippy_ffi::STAGE_PLAN_DESC_V1_ABI_VERSION
            && usize::try_from(descriptor.struct_size).ok()
                == Some(std::mem::size_of::<skippy_ffi::StagePlanDescV1>()),
        "native stage plan descriptor ABI mismatch"
    );
    Ok(descriptor)
}

type ActivationFrontier = (Vec<String>, Vec<String>, Vec<String>, Vec<String>);

fn read_profile_frontier(plan: &Plan, profile_index: usize) -> Result<ActivationFrontier> {
    let mut descriptor = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanProfileDescV1>() };
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_plan_profile_at_v1(
            plan.0,
            profile_index,
            &mut descriptor,
            &mut error,
        )
    };
    ensure_ok(status, error)
        .with_context(|| format!("read native stage profile {profile_index}"))?;
    anyhow::ensure!(
        descriptor.abi_version == skippy_ffi::STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION
            && usize::try_from(descriptor.struct_size).ok()
                == Some(std::mem::size_of::<skippy_ffi::StagePlanProfileDescV1>()),
        "native stage plan profile descriptor ABI mismatch"
    );
    let (imports, import_bindings) = read_frontier_values(
        plan,
        profile_index,
        skippy_ffi::StagePlanValueKind::ActivationImport,
        descriptor.activation_import_count,
    )?;
    let (exports, export_bindings) = read_frontier_values(
        plan,
        profile_index,
        skippy_ffi::StagePlanValueKind::ActivationExport,
        descriptor.activation_export_count,
    )?;
    Ok((imports, import_bindings, exports, export_bindings))
}

fn read_frontier_values(
    plan: &Plan,
    profile_index: usize,
    kind: skippy_ffi::StagePlanValueKind,
    count: u64,
) -> Result<(Vec<String>, Vec<String>)> {
    let count = usize::try_from(count).context("native frontier value count exceeds usize")?;
    let mut identities = Vec::with_capacity(count);
    let mut bindings = Vec::with_capacity(count);
    for index in 0..count {
        let mut descriptor = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanValueDescV1>() };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_stage_plan_value_at_v1(
                plan.0,
                profile_index,
                kind,
                index,
                &mut descriptor,
                &mut error,
            )
        };
        ensure_ok(status, error).with_context(|| {
            format!("read native {kind:?} value {index} for profile {profile_index}")
        })?;
        anyhow::ensure!(
            descriptor.abi_version == skippy_ffi::STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION
                && usize::try_from(descriptor.struct_size).ok()
                    == Some(std::mem::size_of::<skippy_ffi::StagePlanValueDescV1>()),
            "native stage plan value descriptor ABI mismatch"
        );
        identities.push(read_plan_string(plan.0, descriptor.identity)?);
        bindings.push(read_plan_string(plan.0, descriptor.binding)?);
    }
    anyhow::ensure!(
        identities.iter().all(|identity| !identity.is_empty())
            && bindings.iter().all(|binding| !binding.is_empty()),
        "native activation frontier contains an empty identity or binding"
    );
    let mut unique_identities = identities.clone();
    unique_identities.sort();
    unique_identities.dedup();
    let mut unique_bindings = bindings.clone();
    unique_bindings.sort();
    unique_bindings.dedup();
    anyhow::ensure!(
        unique_identities.len() == identities.len() && unique_bindings.len() == bindings.len(),
        "native activation frontier contains duplicate identities or bindings"
    );
    Ok((identities, bindings))
}

fn read_plan_string(
    plan: *const skippy_ffi::StagePlan,
    reference: skippy_ffi::StagePlanStringRefV1,
) -> Result<String> {
    let mut data = ptr::null();
    let mut length = 0;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_plan_string_v1(plan, reference, &mut data, &mut length, &mut error)
    };
    ensure_ok(status, error).context("read native stage-plan string")?;
    anyhow::ensure!(
        length > 0 && !data.is_null(),
        "native stage-plan string is empty"
    );
    let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), length) };
    Ok(std::str::from_utf8(bytes)
        .context("native stage-plan string is not UTF-8")?
        .to_owned())
}
