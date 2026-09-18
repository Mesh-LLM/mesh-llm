//! Native stage-planner inputs, descriptor validation and FFI ownership.
use super::{
    RealizedStagePlan, RealizedStageProfile, RealizedStageStateEffect, StagePlannerProfile,
};
use anyhow::Context as _;
use skippy_package_format::{PackageManifest, Sidecar};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::{CStr, CString},
    path::{Component, Path, PathBuf},
    ptr,
};
/// A native realized plan kept alive until the complete chain has been
/// validated through the public ABI.
struct NativePlan {
    raw: *mut skippy_ffi::StagePlan,
    realized: RealizedStagePlan,
}

impl Drop for NativePlan {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { skippy_ffi::skippy_stage_plan_free(self.raw) };
        }
    }
}

struct NativePlanner(*mut skippy_ffi::StagePlanner);

impl Drop for NativePlanner {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { skippy_ffi::skippy_stage_planner_free(self.0) };
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum NativePlannerSource<'a> {
    Package(&'a Path),
    ExplicitShards(&'a [PathBuf]),
}

pub(super) fn realize_native_stage_chain_from_manifest(
    manifest: &PackageManifest,
    source: NativePlannerSource<'_>,
    ranges: &[(u32, u32)],
    profiles: &[StagePlannerProfile],
    graph_configuration_id: &str,
    backend_id: &str,
    sidecars_by_stage: &[Vec<Sidecar>],
) -> anyhow::Result<(PackageManifest, Vec<RealizedStagePlan>)> {
    anyhow::ensure!(!ranges.is_empty(), "stage plan chain is empty");
    anyhow::ensure!(
        ranges.len() == sidecars_by_stage.len(),
        "stage ranges and sidecar selections differ in length"
    );
    manifest
        .validate()
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("validate stage-planning manifest")?;
    let computed_package_id = manifest
        .computed_package_id()
        .context("compute stage-planning identity")?;
    anyhow::ensure!(
        manifest.package_id == computed_package_id,
        "stage-planning package_id does not match its content"
    );

    let (package_dir, explicit_shard_paths) = match source {
        NativePlannerSource::Package(path) => (Some(path), None),
        NativePlannerSource::ExplicitShards(paths) => (None, Some(paths)),
    };
    let inputs = PlannerInputs::new(
        package_dir,
        explicit_shard_paths,
        manifest,
        profiles,
        graph_configuration_id,
        backend_id,
    )?;
    let planner = inputs.create_native_planner()?;
    let mut plans = Vec::with_capacity(ranges.len());
    for ((layer_start, layer_end), sidecars) in ranges.iter().zip(sidecars_by_stage) {
        plans.push(realize_native_plan(
            &planner,
            *layer_start,
            *layer_end,
            sidecars.clone(),
        )?);
    }
    let raw_plans = plans
        .iter()
        .map(|plan| plan.raw.cast_const())
        .collect::<Vec<_>>();
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_plan_validate_chain_v1(
            raw_plans.as_ptr(),
            raw_plans.len(),
            &mut error,
        )
    };
    ffi_result(status, error).context("validate native stage-plan chain")?;
    Ok((
        manifest.clone(),
        plans
            .into_iter()
            .map(|plan| plan.realized.clone())
            .collect(),
    ))
}

struct PlannerInputs {
    package_id: CString,
    shard_paths: Vec<CString>,
    _tensor_ids: Vec<CString>,
    _tensor_names: Vec<CString>,
    tensors: Vec<skippy_ffi::StagePlannerTensorV1>,
    _profile_ids: Vec<CString>,
    profiles: Vec<skippy_ffi::StagePlannerProfileV1>,
    graph_configuration_id: CString,
    backend_id: CString,
}

fn planner_profiles(
    profiles: &[StagePlannerProfile],
) -> anyhow::Result<(Vec<CString>, Vec<skippy_ffi::StagePlannerProfileV1>)> {
    let mut previous_profile = None;
    let profile_ids = profiles
        .iter()
        .map(|profile| {
            if previous_profile
                .is_some_and(|previous: &str| previous >= profile.profile_id.as_str())
            {
                anyhow::bail!("stage planner profile IDs are not strictly sorted");
            }
            previous_profile = Some(profile.profile_id.as_str());
            cstring(&profile.profile_id, "profile ID")
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(
        !profile_ids.is_empty(),
        "stage planner profile set is empty"
    );
    let native_profiles = profiles
        .iter()
        .zip(&profile_ids)
        .map(|(profile, id)| skippy_ffi::StagePlannerProfileV1 {
            abi_version: skippy_ffi::STAGE_PLANNER_PROFILE_V1_ABI_VERSION,
            struct_size: u32::try_from(std::mem::size_of::<skippy_ffi::StagePlannerProfileV1>())
                .expect("stage planner profile descriptor size fits u32"),
            profile_id: id.as_ptr(),
            n_tokens: profile.n_tokens,
            n_sequences: profile.n_sequences,
            n_outputs: profile.n_outputs,
            n_recurrent_rollback_sequences: profile.n_recurrent_rollback_sequences,
        })
        .collect();
    Ok((profile_ids, native_profiles))
}

impl PlannerInputs {
    fn new(
        package_dir: Option<&Path>,
        explicit_shard_paths: Option<&[PathBuf]>,
        manifest: &PackageManifest,
        profiles: &[StagePlannerProfile],
        graph_configuration_id: &str,
        backend_id: &str,
    ) -> anyhow::Result<Self> {
        let artifact_by_id = manifest
            .artifact_catalog
            .entries
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let mut shard_index_by_artifact = BTreeMap::new();
        let mut carrier_tensors = None;
        let shard_paths = match (package_dir, explicit_shard_paths) {
            (Some(package_dir), None) => {
                let artifact = artifact_by_id
                    .get(manifest.source_model.metadata_artifact_id.as_str())
                    .with_context(|| {
                        format!(
                            "package-v2 metadata artifact {:?} is absent",
                            manifest.source_model.metadata_artifact_id
                        )
                    })?;
                let path = contained_package_path(package_dir, &artifact.path)?;
                anyhow::ensure!(
                    path.is_file(),
                    "stage-planning metadata carrier is missing: {}",
                    path.display()
                );
                let catalog = skippy_model::gguf_catalog::read_gguf_metadata_catalog(&path)
                    .with_context(|| {
                        format!("read stage-planning metadata carrier {}", path.display())
                    })?;
                let tensors = catalog
                    .tensors
                    .into_iter()
                    .map(|tensor| (tensor.name.clone(), tensor))
                    .collect::<BTreeMap<_, _>>();
                anyhow::ensure!(
                    tensors.len() == manifest.tensor_catalog.entries.len(),
                    "metadata carrier and package tensor inventory sizes differ"
                );
                carrier_tensors = Some(tensors);
                vec![path_cstring(&path, "stage-planning metadata carrier path")?]
            }
            (None, Some(paths)) => {
                let mut source_artifacts = artifact_by_id
                    .iter()
                    .filter_map(|(id, artifact)| {
                        source_artifact_index(id).map(|index| (index, *artifact))
                    })
                    .collect::<Vec<_>>();
                source_artifacts.sort_by_key(|(index, _)| *index);
                anyhow::ensure!(
                    !source_artifacts.is_empty()
                        && source_artifacts
                            .iter()
                            .enumerate()
                            .all(|(expected, (actual, _))| expected == *actual),
                    "direct GGUF source artifacts are not a contiguous source-00000 shard set"
                );
                anyhow::ensure!(
                    paths.len() == source_artifacts.len(),
                    "direct GGUF source shard count differs from planning manifest"
                );
                let mut shard_paths = Vec::with_capacity(source_artifacts.len());
                for (index, artifact) in &source_artifacts {
                    let path = paths
                        .get(*index)
                        .cloned()
                        .with_context(|| format!("direct GGUF source shard {index} is missing"))?;
                    anyhow::ensure!(
                        path.is_file(),
                        "stage-planning source artifact is missing: {}",
                        path.display()
                    );
                    shard_paths.push(path_cstring(&path, "stage-planning shard path")?);
                    shard_index_by_artifact.insert(artifact.id.as_str(), *index);
                }
                shard_paths
            }
            _ => anyhow::bail!(
                "stage planner requires exactly one package directory or explicit shard set"
            ),
        };

        let tensor_by_id = manifest
            .tensor_catalog
            .entries
            .iter()
            .map(|tensor| (tensor.id.as_str(), tensor))
            .collect::<BTreeMap<_, _>>();
        anyhow::ensure!(
            tensor_by_id.len() == manifest.tensor_catalog.entries.len(),
            "package-v2 tensor IDs are duplicated"
        );
        let ordered_tensors = tensor_by_id.values().copied().collect::<Vec<_>>();
        let tensor_ids = ordered_tensors
            .iter()
            .map(|tensor| cstring(&tensor.id, "tensor ID"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tensor_names = ordered_tensors
            .iter()
            .map(|tensor| cstring(&tensor.name, "native tensor name"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut tensors = Vec::with_capacity(ordered_tensors.len());
        for (index, tensor) in ordered_tensors.iter().enumerate() {
            let (artifact_id, data_offset, stored_length) =
                tensor_storage(manifest, &tensor_by_id, &tensor.id)?;
            let (split_no, data_offset) = if let Some(carrier_tensors) = &carrier_tensors {
                let carrier = carrier_tensors.get(&tensor.name).with_context(|| {
                    format!(
                        "tensor {:?} is absent from the metadata carrier",
                        tensor.name
                    )
                })?;
                anyhow::ensure!(
                    carrier.ggml_type == tensor.ggml_type
                        && carrier.dimensions == tensor.dimensions,
                    "tensor {:?} metadata differs from the metadata carrier",
                    tensor.name
                );
                (0, carrier.data_offset)
            } else {
                let split_no = *shard_index_by_artifact.get(artifact_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "tensor {:?} resolves to non-source artifact {:?}",
                        tensor.id,
                        artifact_id
                    )
                })?;
                (split_no, data_offset)
            };
            let mut dimensions = [0_i64; skippy_ffi::STAGE_PLAN_MAX_DIMS];
            anyhow::ensure!(
                !tensor.dimensions.is_empty()
                    && tensor.dimensions.len() <= skippy_ffi::STAGE_PLAN_MAX_DIMS,
                "tensor {:?} has unsupported rank {}",
                tensor.id,
                tensor.dimensions.len()
            );
            for (destination, source) in dimensions.iter_mut().zip(&tensor.dimensions) {
                *destination = i64::try_from(*source)
                    .with_context(|| format!("tensor {:?} dimension exceeds i64", tensor.id))?;
                anyhow::ensure!(
                    *destination > 0,
                    "tensor {:?} has an empty dimension",
                    tensor.id
                );
            }
            tensors.push(skippy_ffi::StagePlannerTensorV1 {
                abi_version: skippy_ffi::STAGE_PLANNER_TENSOR_V1_ABI_VERSION,
                struct_size: u32::try_from(std::mem::size_of::<skippy_ffi::StagePlannerTensorV1>())
                    .expect("stage planner tensor descriptor size fits u32"),
                tensor_id: tensor_ids[index].as_ptr(),
                native_name: tensor_names[index].as_ptr(),
                ggml_type: i32::try_from(tensor.ggml_type)
                    .context("GGML tensor type exceeds i32")?,
                dimension_count: u32::try_from(tensor.dimensions.len())
                    .expect("validated tensor rank fits u32"),
                dimensions,
                split_no: u32::try_from(split_no).context("shard index exceeds u32")?,
                reserved: 0,
                data_offset,
                stored_length,
            });
        }

        let (profile_ids, profiles) = planner_profiles(profiles)?;
        Ok(Self {
            package_id: cstring(&manifest.package_id, "package ID")?,
            shard_paths,
            _tensor_ids: tensor_ids,
            _tensor_names: tensor_names,
            tensors,
            _profile_ids: profile_ids,
            profiles,
            graph_configuration_id: cstring(graph_configuration_id, "graph configuration ID")?,
            backend_id: cstring(backend_id, "backend ID")?,
        })
    }

    fn create_native_planner(&self) -> anyhow::Result<NativePlanner> {
        let shard_path_ptrs = self
            .shard_paths
            .iter()
            .map(|path| path.as_ptr())
            .collect::<Vec<_>>();
        let config = skippy_ffi::StagePlannerConfigV1 {
            abi_version: skippy_ffi::STAGE_PLANNER_CONFIG_V1_ABI_VERSION,
            struct_size: u32::try_from(std::mem::size_of::<skippy_ffi::StagePlannerConfigV1>())
                .expect("stage planner config size fits u32"),
            package_id: self.package_id.as_ptr(),
            shard_paths: shard_path_ptrs.as_ptr(),
            shard_count: shard_path_ptrs.len(),
            tensors: self.tensors.as_ptr(),
            tensor_count: self.tensors.len(),
            profiles: self.profiles.as_ptr(),
            profile_count: self.profiles.len(),
            graph_configuration_id: self.graph_configuration_id.as_ptr(),
            backend_id: self.backend_id.as_ptr(),
        };
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_stage_planner_create_v1(&config, &mut raw, &mut error) };
        ffi_result(status, error).context("create native stage planner")?;
        anyhow::ensure!(
            !raw.is_null(),
            "native stage planner returned a null handle"
        );
        Ok(NativePlanner(raw))
    }
}

fn realize_native_plan(
    planner: &NativePlanner,
    layer_start: u32,
    layer_end: u32,
    sidecars: Vec<Sidecar>,
) -> anyhow::Result<NativePlan> {
    let layer_start_i32 = i32::try_from(layer_start).context("stage layer start exceeds i32")?;
    let layer_end_i32 = i32::try_from(layer_end).context("stage layer end exceeds i32")?;
    let mut raw = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_planner_realize_v1(
            planner.0,
            layer_start_i32,
            layer_end_i32,
            &mut raw,
            &mut error,
        )
    };
    if let Err(error) = ffi_result(status, error).context("realize native stage plan") {
        if !raw.is_null() {
            unsafe { skippy_ffi::skippy_stage_plan_free(raw) };
        }
        return Err(error);
    }
    anyhow::ensure!(
        !raw.is_null(),
        "native stage realization returned a null plan"
    );

    let realized = describe_native_plan(raw, sidecars).inspect_err(|_| unsafe {
        skippy_ffi::skippy_stage_plan_free(raw);
    })?;
    anyhow::ensure!(
        realized.layer_start == layer_start && realized.layer_end == layer_end,
        "native stage descriptor range {}..{} differs from requested {}..{}",
        realized.layer_start,
        realized.layer_end,
        layer_start,
        layer_end
    );
    Ok(NativePlan { raw, realized })
}

fn describe_native_plan(
    raw: *const skippy_ffi::StagePlan,
    sidecars: Vec<Sidecar>,
) -> anyhow::Result<RealizedStagePlan> {
    let mut descriptor = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanDescV1>() };
    let mut error = ptr::null_mut();
    let status =
        unsafe { skippy_ffi::skippy_stage_plan_describe_v1(raw, &mut descriptor, &mut error) };
    ffi_result(status, error).context("describe native stage plan")?;
    ensure_descriptor_abi(
        "stage plan",
        descriptor.abi_version,
        skippy_ffi::STAGE_PLAN_DESC_V1_ABI_VERSION,
        descriptor.struct_size,
        std::mem::size_of::<skippy_ffi::StagePlanDescV1>(),
    )?;
    anyhow::ensure!(
        descriptor.layer_count > 0,
        "native stage plan has no layers"
    );
    anyhow::ensure!(
        descriptor.layer_start >= 0
            && descriptor.layer_end > descriptor.layer_start
            && descriptor.layer_end <= descriptor.layer_count,
        "native stage plan has invalid layer range {}..{} for {} layers",
        descriptor.layer_start,
        descriptor.layer_end,
        descriptor.layer_count
    );

    let resident_count = usize::try_from(descriptor.resident_tensor_count)
        .context("native resident tensor count exceeds usize")?;
    let mut resident_tensor_ids = Vec::with_capacity(resident_count);
    for index in 0..resident_count {
        let mut value = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanValueDescV1>() };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_stage_plan_resident_tensor_at_v1(raw, index, &mut value, &mut error)
        };
        ffi_result(status, error)
            .with_context(|| format!("read native resident tensor {index}"))?;
        ensure_descriptor_abi(
            "stage plan resident tensor",
            value.abi_version,
            skippy_ffi::STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION,
            value.struct_size,
            std::mem::size_of::<skippy_ffi::StagePlanValueDescV1>(),
        )?;
        resident_tensor_ids.push(read_plan_string(raw, value.identity)?);
    }
    ensure_canonical_strings("native resident tensor IDs", &resident_tensor_ids)?;

    let profile_count =
        usize::try_from(descriptor.profile_count).context("native profile count exceeds usize")?;
    anyhow::ensure!(
        profile_count > 0,
        "native stage plan has no guarded profiles"
    );
    let mut profiles = Vec::with_capacity(profile_count);
    for profile_index in 0..profile_count {
        profiles.push(read_native_profile(raw, profile_index)?);
    }
    ensure_canonical_strings(
        "native profile IDs",
        &profiles
            .iter()
            .map(|profile| profile.profile_id.clone())
            .collect::<Vec<_>>(),
    )?;

    for (index, window) in sidecars.windows(2).enumerate() {
        anyhow::ensure!(
            window[0] < window[1],
            "native stage host sidecars are not strictly sorted at index {index}"
        );
    }
    let package_id = read_plan_string(raw, descriptor.package_id)?;
    let plan_id = read_plan_string(raw, descriptor.plan_id)?;
    anyhow::ensure!(
        package_id
            .strip_prefix("sha256:")
            .is_some_and(is_lower_hex_digest),
        "native stage package identity is not canonical"
    );
    anyhow::ensure!(
        plan_id
            .strip_prefix("skippy-plan:v1:")
            .is_some_and(is_lower_hex_digest),
        "native stage plan identity is not canonical"
    );
    Ok(RealizedStagePlan {
        package_id,
        plan_id,
        layer_start: u32::try_from(descriptor.layer_start)
            .expect("validated nonnegative native layer start"),
        layer_end: u32::try_from(descriptor.layer_end)
            .expect("validated positive native layer end"),
        resident_tensor_ids,
        sidecars,
        profiles,
    })
}

fn read_native_profile(
    raw: *const skippy_ffi::StagePlan,
    profile_index: usize,
) -> anyhow::Result<RealizedStageProfile> {
    let mut descriptor = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanProfileDescV1>() };
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_plan_profile_at_v1(raw, profile_index, &mut descriptor, &mut error)
    };
    ffi_result(status, error)
        .with_context(|| format!("read native stage profile {profile_index}"))?;
    ensure_descriptor_abi(
        "stage plan profile",
        descriptor.abi_version,
        skippy_ffi::STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION,
        descriptor.struct_size,
        std::mem::size_of::<skippy_ffi::StagePlanProfileDescV1>(),
    )?;
    anyhow::ensure!(
        descriptor.n_tokens > 0
            && descriptor.n_sequences > 0
            && descriptor.n_outputs > 0
            && descriptor.n_tokens % descriptor.n_sequences == 0
            && descriptor.n_outputs <= descriptor.n_tokens,
        "native stage profile {profile_index} has an invalid execution guard"
    );

    let (activation_imports, activation_import_bindings) = read_native_frontier_values(
        raw,
        profile_index,
        skippy_ffi::StagePlanValueKind::ActivationImport,
        descriptor.activation_import_count,
    )?;
    let (activation_exports, activation_export_bindings) = read_native_frontier_values(
        raw,
        profile_index,
        skippy_ffi::StagePlanValueKind::ActivationExport,
        descriptor.activation_export_count,
    )?;
    let request_inputs = read_native_values(
        raw,
        profile_index,
        skippy_ffi::StagePlanValueKind::RequestInput,
        descriptor.request_input_count,
    )?;
    let state_count = usize::try_from(descriptor.state_effect_count)
        .context("native state effect count exceeds usize")?;
    let mut state_effects = Vec::with_capacity(state_count);
    let mut state_identities = BTreeSet::new();
    for index in 0..state_count {
        let mut state = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanStateDescV1>() };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_stage_plan_state_at_v1(
                raw,
                profile_index,
                index,
                &mut state,
                &mut error,
            )
        };
        ffi_result(status, error).with_context(|| {
            format!("read native state effect {index} for profile {profile_index}")
        })?;
        ensure_descriptor_abi(
            "stage plan state effect",
            state.abi_version,
            skippy_ffi::STAGE_PLAN_STATE_DESC_V1_ABI_VERSION,
            state.struct_size,
            std::mem::size_of::<skippy_ffi::StagePlanStateDescV1>(),
        )?;
        anyhow::ensure!(
            state.reserved == 0,
            "native state effect reserved field is nonzero"
        );
        let identity = read_plan_string(raw, state.identity)?;
        anyhow::ensure!(
            state_identities.insert(identity.clone()),
            "native state effect identity {identity:?} is duplicated"
        );
        let kind = match state.kind {
            value if value == skippy_ffi::StagePlanStateKind::KvKey as i32 => {
                skippy_ffi::StagePlanStateKind::KvKey
            }
            value if value == skippy_ffi::StagePlanStateKind::KvValue as i32 => {
                skippy_ffi::StagePlanStateKind::KvValue
            }
            value if value == skippy_ffi::StagePlanStateKind::RecurrentConv as i32 => {
                skippy_ffi::StagePlanStateKind::RecurrentConv
            }
            value if value == skippy_ffi::StagePlanStateKind::RecurrentSsm as i32 => {
                skippy_ffi::StagePlanStateKind::RecurrentSsm
            }
            value if value == skippy_ffi::StagePlanStateKind::DerivedPersistent as i32 => {
                skippy_ffi::StagePlanStateKind::DerivedPersistent
            }
            unknown => anyhow::bail!("native state effect kind {unknown} is unsupported"),
        };
        let access = match state.access {
            value if value == skippy_ffi::StagePlanStateAccess::Read as i32 => {
                skippy_ffi::StagePlanStateAccess::Read
            }
            value if value == skippy_ffi::StagePlanStateAccess::Write as i32 => {
                skippy_ffi::StagePlanStateAccess::Write
            }
            unknown => anyhow::bail!("native state effect access {unknown} is unsupported"),
        };
        state_effects.push(RealizedStageStateEffect {
            identity,
            kind,
            access,
            layer: state.layer,
            write_ordinal: state.write_ordinal,
        });
    }

    Ok(RealizedStageProfile {
        profile_id: read_plan_string(raw, descriptor.profile_id)?,
        graph_identity: read_plan_string(raw, descriptor.graph_identity)?,
        profile_identity: read_plan_string(raw, descriptor.profile_identity)?,
        slice_identity: read_plan_string(raw, descriptor.slice_identity)?,
        source_snapshot_identity: read_plan_string(raw, descriptor.source_snapshot_identity)?,
        graph_configuration_id: read_plan_string(raw, descriptor.graph_configuration_id)?,
        backend_id: read_plan_string(raw, descriptor.backend_id)?,
        activation_imports,
        activation_exports,
        activation_import_bindings,
        activation_export_bindings,
        request_inputs,
        state_effects,
    })
}

fn read_native_frontier_values(
    raw: *const skippy_ffi::StagePlan,
    profile_index: usize,
    kind: skippy_ffi::StagePlanValueKind,
    count: u64,
) -> anyhow::Result<(Vec<String>, Vec<String>)> {
    let count = usize::try_from(count).context("native frontier value count exceeds usize")?;
    let mut identities = Vec::with_capacity(count);
    let mut bindings = Vec::with_capacity(count);
    for index in 0..count {
        let descriptor = read_native_value_descriptor(raw, profile_index, kind, index)?;
        identities.push(read_plan_string(raw, descriptor.identity)?);
        bindings.push(read_plan_string(raw, descriptor.binding)?);
    }
    ensure_unique_strings(&format!("native {kind:?} identities"), &identities)?;
    ensure_unique_strings(&format!("native {kind:?} bindings"), &bindings)?;
    Ok((identities, bindings))
}

fn read_native_values(
    raw: *const skippy_ffi::StagePlan,
    profile_index: usize,
    kind: skippy_ffi::StagePlanValueKind,
    count: u64,
) -> anyhow::Result<Vec<String>> {
    let count = usize::try_from(count).context("native stage value count exceeds usize")?;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let descriptor = read_native_value_descriptor(raw, profile_index, kind, index)?;
        values.push(read_plan_string(raw, descriptor.identity)?);
    }
    ensure_unique_strings(&format!("native {kind:?} identities"), &values)?;
    Ok(values)
}

fn read_native_value_descriptor(
    raw: *const skippy_ffi::StagePlan,
    profile_index: usize,
    kind: skippy_ffi::StagePlanValueKind,
    index: usize,
) -> anyhow::Result<skippy_ffi::StagePlanValueDescV1> {
    let mut descriptor = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanValueDescV1>() };
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_plan_value_at_v1(
            raw,
            profile_index,
            kind,
            index,
            &mut descriptor,
            &mut error,
        )
    };
    ffi_result(status, error).with_context(|| {
        format!("read native {kind:?} value {index} for profile {profile_index}")
    })?;
    ensure_descriptor_abi(
        "stage plan value",
        descriptor.abi_version,
        skippy_ffi::STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION,
        descriptor.struct_size,
        std::mem::size_of::<skippy_ffi::StagePlanValueDescV1>(),
    )?;
    Ok(descriptor)
}

fn read_plan_string(
    raw: *const skippy_ffi::StagePlan,
    reference: skippy_ffi::StagePlanStringRefV1,
) -> anyhow::Result<String> {
    let mut data = ptr::null();
    let mut length = 0;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_plan_string_v1(raw, reference, &mut data, &mut length, &mut error)
    };
    ffi_result(status, error).context("read native stage-plan string")?;
    anyhow::ensure!(length > 0, "native stage-plan string is empty");
    anyhow::ensure!(!data.is_null(), "native stage-plan string data is null");
    let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), length) };
    Ok(std::str::from_utf8(bytes)
        .context("native stage-plan string is not UTF-8")?
        .to_owned())
}

fn ensure_descriptor_abi(
    label: &str,
    actual_version: u32,
    expected_version: u32,
    actual_size: u32,
    expected_size: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        actual_version == expected_version,
        "native {label} ABI version {actual_version} differs from expected {expected_version}"
    );
    anyhow::ensure!(
        usize::try_from(actual_size).ok() == Some(expected_size),
        "native {label} size {actual_size} differs from expected {expected_size}"
    );
    Ok(())
}

fn ensure_canonical_strings(label: &str, values: &[String]) -> anyhow::Result<()> {
    for (index, window) in values.windows(2).enumerate() {
        anyhow::ensure!(
            window[0] < window[1],
            "{label} are not strictly sorted at index {index}"
        );
    }
    Ok(())
}

fn ensure_unique_strings(label: &str, values: &[String]) -> anyhow::Result<()> {
    let mut unique = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        anyhow::ensure!(
            unique.insert(value),
            "{label} contain a duplicate at index {index}: {value:?}"
        );
    }
    Ok(())
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn ffi_result(status: skippy_ffi::Status, error: *mut skippy_ffi::Error) -> anyhow::Result<()> {
    let message = if error.is_null() {
        String::new()
    } else {
        let message = unsafe { (*error).message };
        if message.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        }
    };
    if !error.is_null() {
        unsafe { skippy_ffi::skippy_error_free(error) };
    }
    anyhow::ensure!(
        status == skippy_ffi::Status::Ok,
        "native stage planner returned {status:?}: {message}"
    );
    Ok(())
}

fn source_artifact_index(id: &str) -> Option<usize> {
    id.strip_prefix("source-")?.parse().ok()
}

fn tensor_storage<'a>(
    manifest: &'a PackageManifest,
    tensors: &BTreeMap<&str, &'a skippy_package_format::Tensor>,
    tensor_id: &str,
) -> anyhow::Result<(&'a str, u64, u64)> {
    let mut current = tensor_id;
    let mut visited = BTreeSet::new();
    loop {
        anyhow::ensure!(visited.insert(current), "tensor alias cycle at {current:?}");
        let tensor = tensors
            .get(current)
            .with_context(|| format!("tensor alias target {current:?} is absent"))?;
        match &tensor.storage {
            skippy_package_format::TensorStorage::Owned {
                artifact_id,
                data_offset,
                stored_length,
                ..
            } => {
                anyhow::ensure!(
                    manifest
                        .artifact_catalog
                        .entries
                        .iter()
                        .any(|artifact| artifact.id == *artifact_id),
                    "tensor {tensor_id:?} references missing artifact {artifact_id:?}"
                );
                return Ok((artifact_id, *data_offset, *stored_length));
            }
            skippy_package_format::TensorStorage::Alias { target_tensor_id } => {
                current = target_tensor_id;
            }
        }
    }
}

fn contained_package_path(package_dir: &Path, relative: &str) -> anyhow::Result<PathBuf> {
    let relative = Path::new(relative);
    anyhow::ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "package-v2 artifact path is not a safe relative path: {relative:?}"
    );
    Ok(package_dir.join(relative))
}

fn cstring(value: &str, label: &str) -> anyhow::Result<CString> {
    anyhow::ensure!(!value.is_empty(), "{label} is empty");
    CString::new(value).with_context(|| format!("{label} contains an interior NUL byte"))
}

fn path_cstring(path: &Path, label: &str) -> anyhow::Result<CString> {
    let value = path
        .to_str()
        .with_context(|| format!("{label} is not valid UTF-8: {}", path.display()))?;
    cstring(value, label)
}
