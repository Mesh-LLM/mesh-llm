//! Skippy admission of realized stage plans against planned stages.
//!
//! Binds the realized native stage descriptor (returned through the public
//! stage-plan ABI, patch 0074) to the planned stage before topology
//! publication. Every binding below is exact and fail-closed: package
//! identity is recomputed from manifest content, plan IDs must be stable,
//! layer ranges must match the plan, resident tensor closures resolve
//! through the v2 manifest with no name/layer/role fallbacks, and guarded
//! profile identities must agree on both sides. Any mismatch produces a
//! structured [`StagePlanAdmissionError`] before a topology can be published.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use anyhow::Context as _;
use skippy_package_format::{PackageManifest, Sidecar};

mod native;
use native::{NativePlannerSource, realize_native_stage_chain_from_manifest};

/// The planned admission expectations for one stage.
///
/// This is carried by the planner on `RuntimeSliceStagePlan` from planning
/// time and mirrored into the generation-10 control protocol descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedStageAdmission {
    /// Content-derived package identity (`sha256:...`).
    pub package_id: String,
    /// Deterministic native semantic plan identity (`skippy-plan:v1:...`).
    pub plan_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    /// Exact sorted, unique resident tensor IDs.
    pub resident_tensor_ids: Vec<String>,
    /// Typed sidecar references, strictly sorted.
    pub sidecars: Vec<Sidecar>,
    /// Guarded per-profile identities, sorted by `profile_id`.
    pub profiles: Vec<PlannedStageProfile>,
}

/// One guarded execution profile's planned identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedStageProfile {
    pub profile_id: String,
    pub graph_identity: String,
    pub profile_identity: String,
    pub slice_identity: String,
    pub source_snapshot_identity: String,
    pub graph_configuration_id: String,
    pub backend_id: String,
    pub activation_imports: Vec<String>,
    pub activation_exports: Vec<String>,
    pub activation_import_bindings: Vec<String>,
    pub activation_export_bindings: Vec<String>,
}

/// The realized native stage descriptor as returned through the ABI.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RealizedStagePlan {
    pub package_id: String,
    pub plan_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    /// Exact sorted, unique resident tensor IDs realized by the native side.
    pub resident_tensor_ids: Vec<String>,
    /// Typed sidecar references realized by the native side, strictly sorted.
    pub sidecars: Vec<Sidecar>,
    pub profiles: Vec<RealizedStageProfile>,
}

/// One guarded execution profile's realized identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RealizedStageProfile {
    pub profile_id: String,
    pub graph_identity: String,
    pub profile_identity: String,
    pub slice_identity: String,
    pub source_snapshot_identity: String,
    pub graph_configuration_id: String,
    pub backend_id: String,
    pub activation_imports: Vec<String>,
    pub activation_exports: Vec<String>,
    pub activation_import_bindings: Vec<String>,
    pub activation_export_bindings: Vec<String>,
    pub request_inputs: Vec<String>,
    pub state_effects: Vec<RealizedStageStateEffect>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RealizedStageStateEffect {
    pub identity: String,
    pub kind: skippy_ffi::StagePlanStateKind,
    pub access: skippy_ffi::StagePlanStateAccess,
    pub layer: i32,
    pub write_ordinal: i64,
}

/// Exact native planning profiles. The IDs must be strictly sorted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagePlannerProfile {
    pub profile_id: String,
    pub n_tokens: u32,
    pub n_sequences: u32,
    pub n_outputs: u32,
    pub n_recurrent_rollback_sequences: u32,
}

fn planned_admission_from_discovery(
    manifest: &PackageManifest,
    range: (u32, u32),
    sidecars: &[Sidecar],
    discovered: &RealizedStagePlan,
) -> PlannedStageAdmission {
    PlannedStageAdmission {
        package_id: manifest.package_id.clone(),
        plan_id: discovered.plan_id.clone(),
        layer_start: range.0,
        layer_end: range.1,
        resident_tensor_ids: discovered.resident_tensor_ids.clone(),
        sidecars: sidecars.to_vec(),
        profiles: discovered
            .profiles
            .iter()
            .map(|profile| PlannedStageProfile {
                profile_id: profile.profile_id.clone(),
                graph_identity: profile.graph_identity.clone(),
                profile_identity: profile.profile_identity.clone(),
                slice_identity: profile.slice_identity.clone(),
                source_snapshot_identity: profile.source_snapshot_identity.clone(),
                graph_configuration_id: profile.graph_configuration_id.clone(),
                backend_id: profile.backend_id.clone(),
                activation_imports: profile.activation_imports.clone(),
                activation_exports: profile.activation_exports.clone(),
                activation_import_bindings: profile.activation_import_bindings.clone(),
                activation_export_bindings: profile.activation_export_bindings.clone(),
            })
            .collect(),
    }
}

impl From<&PlannedStageAdmission> for skippy_protocol::StageAdmissionDescriptor {
    fn from(planned: &PlannedStageAdmission) -> Self {
        Self {
            version: skippy_protocol::STAGE_ADMISSION_DESCRIPTOR_VERSION,
            package_id: planned.package_id.clone(),
            plan_id: planned.plan_id.clone(),
            layer_start: planned.layer_start,
            layer_end: planned.layer_end,
            resident_tensor_ids: planned.resident_tensor_ids.clone(),
            sidecars: planned
                .sidecars
                .iter()
                .map(|sidecar| skippy_protocol::StageAdmissionSidecar {
                    kind: match sidecar.kind {
                        skippy_package_format::SidecarKind::Mmproj => {
                            skippy_protocol::StageAdmissionSidecarKind::Mmproj
                        }
                    },
                    artifact_id: sidecar.artifact_id.clone(),
                    name: sidecar.name.clone(),
                })
                .collect(),
            profiles: planned
                .profiles
                .iter()
                .map(|profile| skippy_protocol::StageAdmissionProfile {
                    profile_id: profile.profile_id.clone(),
                    graph_identity: profile.graph_identity.clone(),
                    profile_identity: profile.profile_identity.clone(),
                    slice_identity: profile.slice_identity.clone(),
                    source_snapshot_identity: profile.source_snapshot_identity.clone(),
                    graph_configuration_id: profile.graph_configuration_id.clone(),
                    backend_id: profile.backend_id.clone(),
                    activation_imports: profile.activation_imports.clone(),
                    activation_exports: profile.activation_exports.clone(),
                    activation_import_bindings: profile.activation_import_bindings.clone(),
                    activation_export_bindings: profile.activation_export_bindings.clone(),
                })
                .collect(),
        }
    }
}

/// The admitted stage: exact identities that passed every check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedStage {
    pub package_id: String,
    pub plan_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub resident_tensor_ids: Vec<String>,
}

#[derive(Debug)]
pub enum StagePlanAdmissionError {
    PlannedTensorIdsNotStrictlySorted {
        index: usize,
        previous: String,
        current: String,
    },
    RealizedTensorIdsNotStrictlySorted {
        index: usize,
        previous: String,
        current: String,
    },
    PlannedSidecarsNotStrictlySorted {
        index: usize,
    },
    RealizedSidecarsNotStrictlySorted {
        index: usize,
    },
    PlannedProfilesNotSorted {
        index: usize,
    },
    RealizedProfilesNotSorted {
        index: usize,
    },
    PackageIdMismatch {
        planned: String,
        realized: String,
    },
    PlanIdMismatch {
        planned: String,
        realized: String,
    },
    LayerRangeMismatch {
        planned: (u32, u32),
        realized: (u32, u32),
    },
    TensorClosureMismatch {
        planned_only: Vec<String>,
        realized_only: Vec<String>,
    },
    SidecarMismatch {
        planned_only: Vec<Sidecar>,
        realized_only: Vec<Sidecar>,
    },
    ProfileSetMismatch {
        planned_only: Vec<String>,
        realized_only: Vec<String>,
    },
    ProfileIdentityMismatch {
        profile_id: String,
        field: &'static str,
        planned: String,
        realized: String,
    },
    ManifestResolution(skippy_package_format::stage_admission::StageAdmissionError),
}

impl fmt::Display for StagePlanAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlannedTensorIdsNotStrictlySorted {
                index,
                previous,
                current,
            } => write!(
                formatter,
                "planned resident tensor ids are not strictly sorted at index {index}: {previous:?} then {current:?}"
            ),
            Self::RealizedTensorIdsNotStrictlySorted {
                index,
                previous,
                current,
            } => write!(
                formatter,
                "realized resident tensor ids are not strictly sorted at index {index}: {previous:?} then {current:?}"
            ),
            Self::PlannedSidecarsNotStrictlySorted { index } => write!(
                formatter,
                "planned sidecars are not strictly sorted at index {index}"
            ),
            Self::RealizedSidecarsNotStrictlySorted { index } => write!(
                formatter,
                "realized sidecars are not strictly sorted at index {index}"
            ),
            Self::PlannedProfilesNotSorted { index } => write!(
                formatter,
                "planned profiles are not sorted by profile id at index {index}"
            ),
            Self::RealizedProfilesNotSorted { index } => write!(
                formatter,
                "realized profiles are not sorted by profile id at index {index}"
            ),
            Self::PackageIdMismatch { planned, realized } => write!(
                formatter,
                "realized stage package id {realized:?} does not match planned {planned:?}"
            ),
            Self::PlanIdMismatch { planned, realized } => write!(
                formatter,
                "realized stage plan id {realized:?} does not match planned {planned:?}"
            ),
            Self::LayerRangeMismatch { planned, realized } => write!(
                formatter,
                "realized stage layer range {:?} does not match planned {:?}",
                realized, planned
            ),
            Self::TensorClosureMismatch {
                planned_only,
                realized_only,
            } => write!(
                formatter,
                "realized resident tensor closure differs from plan: unrealized planned {planned_only:?}, unplanned realized {realized_only:?}"
            ),
            Self::SidecarMismatch {
                planned_only,
                realized_only,
            } => write!(
                formatter,
                "realized sidecars differ from plan: unrealized planned {planned_only:?}, unplanned realized {realized_only:?}"
            ),
            Self::ProfileSetMismatch {
                planned_only,
                realized_only,
            } => write!(
                formatter,
                "realized profile set differs from plan: unrealized planned {planned_only:?}, unplanned realized {realized_only:?}"
            ),
            Self::ProfileIdentityMismatch {
                profile_id,
                field,
                planned,
                realized,
            } => write!(
                formatter,
                "realized profile {profile_id:?} field {field} is {realized:?} but planned {planned:?}"
            ),
            Self::ManifestResolution(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for StagePlanAdmissionError {}

/// Open a package-v2 manifest and realize an exact native plan for every
/// requested stage. The native plan objects remain alive until the complete
/// chain passes `skippy_stage_plan_validate_chain_v1`.
pub fn realize_native_stage_chain(
    package_dir: &Path,
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
    let manifest_path = package_dir.join("model-package.json");
    let manifest: PackageManifest = serde_json::from_slice(
        &std::fs::read(&manifest_path)
            .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse package-v2 manifest {}", manifest_path.display()))?;
    let manifest =
        skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, package_dir)
            .context("resolve package-v2 metadata carrier")?;
    let computed_package_id = manifest
        .computed_package_id()
        .context("compute package-v2 identity")?;
    anyhow::ensure!(
        manifest.package_id == computed_package_id,
        "package-v2 manifest package_id does not match its content"
    );

    realize_native_stage_chain_from_manifest(
        &manifest,
        NativePlannerSource::Package(package_dir),
        ranges,
        profiles,
        graph_configuration_id,
        backend_id,
        sidecars_by_stage,
    )
}

/// Realize, package-resolve, and admit every stage before a topology can be
/// published. Returned descriptors are canonical generation-10 wire values.
pub fn realize_stage_admissions(
    package_dir: &Path,
    ranges: &[(u32, u32)],
    profiles: &[StagePlannerProfile],
    graph_configuration_id: &str,
    backend_id: &str,
) -> anyhow::Result<Vec<skippy_protocol::StageAdmissionDescriptor>> {
    let manifest_bytes = std::fs::read(package_dir.join("model-package.json"))
        .context("read package-v2 manifest for sidecar assignment")?;
    let manifest: PackageManifest =
        serde_json::from_slice(&manifest_bytes).context("parse package-v2 manifest")?;
    let sidecars_by_stage = ranges
        .iter()
        .enumerate()
        .map(|(index, _)| {
            if index == 0 {
                manifest.sidecars.clone()
            } else {
                Vec::new()
            }
        })
        .collect::<Vec<_>>();
    // Discover and freeze the planner-side expectations before production
    // realization. The second native pass below is intentionally independent:
    // admission compares two separately created graph plans instead of cloning
    // the realized descriptor and comparing it with itself.
    let (planned_manifest, discovered) = realize_native_stage_chain(
        package_dir,
        ranges,
        profiles,
        graph_configuration_id,
        backend_id,
        &sidecars_by_stage,
    )?;
    let planned = ranges
        .iter()
        .zip(&sidecars_by_stage)
        .zip(&discovered)
        .map(|((range, sidecars), discovered)| {
            let planned =
                planned_admission_from_discovery(&planned_manifest, *range, sidecars, discovered);
            admit_stage_plan(&planned, discovered, &planned_manifest)
                .context("validate discovered native stage plan")?;
            Ok(planned)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let (realized_manifest, realized) = realize_native_stage_chain(
        package_dir,
        ranges,
        profiles,
        graph_configuration_id,
        backend_id,
        &sidecars_by_stage,
    )?;
    anyhow::ensure!(
        planned_manifest.package_id == realized_manifest.package_id,
        "package identity changed between native planning and realization"
    );
    planned
        .iter()
        .zip(&realized)
        .map(|(planned, realized)| {
            admit_stage_plan(planned, realized, &realized_manifest)
                .context("admit independently realized native stage plan")?;
            Ok(skippy_protocol::StageAdmissionDescriptor::from(planned))
        })
        .collect()
}

/// Realize and admit a direct GGUF chain from an in-memory source-complete
/// planning manifest. The original shards are opened in place; no package or
/// layer artifacts are written.
pub fn realize_direct_gguf_stage_admissions(
    model_id: &str,
    identity: &crate::package::SkippyPackageIdentity,
    ranges: &[(u32, u32)],
    profiles: &[StagePlannerProfile],
    graph_configuration_id: &str,
    backend_id: &str,
) -> anyhow::Result<Vec<skippy_protocol::StageAdmissionDescriptor>> {
    let (manifest, shard_paths) =
        crate::source::planning::direct_gguf_planning_manifest_from_identity(model_id, identity)?;
    let sidecars_by_stage = vec![Vec::new(); ranges.len()];
    let (planned_manifest, discovered) = realize_native_stage_chain_from_manifest(
        &manifest,
        NativePlannerSource::ExplicitShards(&shard_paths),
        ranges,
        profiles,
        graph_configuration_id,
        backend_id,
        &sidecars_by_stage,
    )?;
    let planned = ranges
        .iter()
        .zip(&sidecars_by_stage)
        .zip(&discovered)
        .map(|((range, sidecars), discovered)| {
            let planned =
                planned_admission_from_discovery(&planned_manifest, *range, sidecars, discovered);
            admit_stage_plan(&planned, discovered, &planned_manifest)
                .context("validate discovered direct GGUF stage plan")?;
            Ok(planned)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let (realized_manifest, realized) = realize_native_stage_chain_from_manifest(
        &manifest,
        NativePlannerSource::ExplicitShards(&shard_paths),
        ranges,
        profiles,
        graph_configuration_id,
        backend_id,
        &sidecars_by_stage,
    )?;
    anyhow::ensure!(
        planned_manifest.package_id == realized_manifest.package_id,
        "direct GGUF planning identity changed between native passes"
    );
    planned
        .iter()
        .zip(&realized)
        .map(|(planned, realized)| {
            admit_stage_plan(planned, realized, &realized_manifest)
                .context("admit independently realized direct GGUF stage plan")?;
            Ok(skippy_protocol::StageAdmissionDescriptor::from(planned))
        })
        .collect()
}

fn ensure_strictly_sorted(
    ids: &[String],
    mut error: impl FnMut(usize, String, String) -> StagePlanAdmissionError,
) -> Result<(), StagePlanAdmissionError> {
    for (index, window) in ids.windows(2).enumerate() {
        if window[0] >= window[1] {
            return Err(error(index, window[0].clone(), window[1].clone()));
        }
    }
    Ok(())
}

fn ensure_sidecars_sorted(
    sidecars: &[Sidecar],
    planned: bool,
) -> Result<(), StagePlanAdmissionError> {
    for window in sidecars.windows(2) {
        if window[0] >= window[1] {
            let index = 0;
            return Err(if planned {
                StagePlanAdmissionError::PlannedSidecarsNotStrictlySorted { index }
            } else {
                StagePlanAdmissionError::RealizedSidecarsNotStrictlySorted { index }
            });
        }
    }
    Ok(())
}

fn diff_sorted<T>(planned: &[T], realized: &[T]) -> (Vec<T>, Vec<T>)
where
    T: Ord + Clone,
{
    let mut planned_only = Vec::new();
    let mut realized_only = Vec::new();
    let mut planned_iter = planned.iter().peekable();
    let mut realized_iter = realized.iter().peekable();
    loop {
        match (planned_iter.peek(), realized_iter.peek()) {
            (None, None) => break,
            (Some(_), None) => {
                planned_only.push(planned_iter.next().unwrap().clone());
            }
            (None, Some(_)) => {
                realized_only.push(realized_iter.next().unwrap().clone());
            }
            (Some(p), Some(r)) => match p.cmp(r) {
                std::cmp::Ordering::Less => {
                    planned_only.push(planned_iter.next().unwrap().clone());
                }
                std::cmp::Ordering::Greater => {
                    realized_only.push(realized_iter.next().unwrap().clone());
                }
                std::cmp::Ordering::Equal => {
                    planned_iter.next();
                    realized_iter.next();
                }
            },
        }
    }
    (planned_only, realized_only)
}

/// Bind a realized stage plan to its planned admission expectations.
///
/// Resolution order:
/// 1. structural checks on both sides (sortedness of tensor ids, sidecars,
///    profiles);
/// 2. exact package and plan identity;
/// 3. exact layer range;
/// 4. exact resident tensor closure and sidecar set;
/// 5. guarded profile identities per profile;
/// 6. package identity recomputation and tensor closure resolution through
///    the v2 manifest (`PackageManifest::resolve_stage_admission`).
///
/// The returned [`AdmittedStage`] is the only signal that a stage may be
/// published into the topology.
pub fn admit_stage_plan(
    planned: &PlannedStageAdmission,
    realized: &RealizedStagePlan,
    manifest: &PackageManifest,
) -> Result<AdmittedStage, StagePlanAdmissionError> {
    ensure_strictly_sorted(&planned.resident_tensor_ids, |index, previous, current| {
        StagePlanAdmissionError::PlannedTensorIdsNotStrictlySorted {
            index,
            previous,
            current,
        }
    })?;
    ensure_strictly_sorted(&realized.resident_tensor_ids, |index, previous, current| {
        StagePlanAdmissionError::RealizedTensorIdsNotStrictlySorted {
            index,
            previous,
            current,
        }
    })?;
    ensure_sidecars_sorted(&planned.sidecars, true)?;
    ensure_sidecars_sorted(&realized.sidecars, false)?;

    if planned.package_id != realized.package_id {
        return Err(StagePlanAdmissionError::PackageIdMismatch {
            planned: planned.package_id.clone(),
            realized: realized.package_id.clone(),
        });
    }
    if planned.plan_id != realized.plan_id {
        return Err(StagePlanAdmissionError::PlanIdMismatch {
            planned: planned.plan_id.clone(),
            realized: realized.plan_id.clone(),
        });
    }
    let planned_range = (planned.layer_start, planned.layer_end);
    let realized_range = (realized.layer_start, realized.layer_end);
    if planned_range != realized_range {
        return Err(StagePlanAdmissionError::LayerRangeMismatch {
            planned: planned_range,
            realized: realized_range,
        });
    }

    let (planned_only, realized_only) =
        diff_sorted(&planned.resident_tensor_ids, &realized.resident_tensor_ids);
    if !planned_only.is_empty() || !realized_only.is_empty() {
        return Err(StagePlanAdmissionError::TensorClosureMismatch {
            planned_only,
            realized_only,
        });
    }

    let (planned_only, realized_only) = diff_sorted(&planned.sidecars, &realized.sidecars);
    if !planned_only.is_empty() || !realized_only.is_empty() {
        return Err(StagePlanAdmissionError::SidecarMismatch {
            planned_only,
            realized_only,
        });
    }

    admit_profiles(planned, realized)?;

    // Full package admission resolution: recomputes the package id from
    // manifest content, binds every resident tensor through the v2 catalog,
    // and resolves typed sidecar references. Fails closed on any mismatch.
    manifest
        .resolve_stage_admission(
            &skippy_package_format::stage_admission::StageAdmissionDescriptor {
                package_id: realized.package_id.clone(),
                resident_tensor_ids: realized.resident_tensor_ids.clone(),
                sidecars: realized.sidecars.clone(),
            },
        )
        .map_err(StagePlanAdmissionError::ManifestResolution)?;

    Ok(AdmittedStage {
        package_id: realized.package_id.clone(),
        plan_id: realized.plan_id.clone(),
        layer_start: realized.layer_start,
        layer_end: realized.layer_end,
        resident_tensor_ids: realized.resident_tensor_ids.clone(),
    })
}

fn admit_profiles(
    planned: &PlannedStageAdmission,
    realized: &RealizedStagePlan,
) -> Result<(), StagePlanAdmissionError> {
    let planned_ids: Vec<&str> = planned
        .profiles
        .iter()
        .map(|p| p.profile_id.as_str())
        .collect();
    let realized_ids: Vec<&str> = realized
        .profiles
        .iter()
        .map(|p| p.profile_id.as_str())
        .collect();
    for window in planned_ids.windows(2) {
        if window[0] >= window[1] {
            return Err(StagePlanAdmissionError::PlannedProfilesNotSorted { index: 0 });
        }
    }
    for window in realized_ids.windows(2) {
        if window[0] >= window[1] {
            return Err(StagePlanAdmissionError::RealizedProfilesNotSorted { index: 0 });
        }
    }
    let (planned_only, realized_only) = diff_sorted(&planned_ids, &realized_ids);
    if !planned_only.is_empty() || !realized_only.is_empty() {
        return Err(StagePlanAdmissionError::ProfileSetMismatch {
            planned_only: planned_only.into_iter().map(str::to_string).collect(),
            realized_only: realized_only.into_iter().map(str::to_string).collect(),
        });
    }
    let planned_by_id: BTreeMap<&str, &PlannedStageProfile> = planned
        .profiles
        .iter()
        .map(|p| (p.profile_id.as_str(), p))
        .collect();
    for realized_profile in &realized.profiles {
        let planned_profile = planned_by_id
            .get(realized_profile.profile_id.as_str())
            .expect("profile set equality checked above");
        for (field, planned_value, realized_value) in [
            (
                "graph_identity",
                &planned_profile.graph_identity,
                &realized_profile.graph_identity,
            ),
            (
                "profile_identity",
                &planned_profile.profile_identity,
                &realized_profile.profile_identity,
            ),
            (
                "slice_identity",
                &planned_profile.slice_identity,
                &realized_profile.slice_identity,
            ),
            (
                "source_snapshot_identity",
                &planned_profile.source_snapshot_identity,
                &realized_profile.source_snapshot_identity,
            ),
            (
                "graph_configuration_id",
                &planned_profile.graph_configuration_id,
                &realized_profile.graph_configuration_id,
            ),
            (
                "backend_id",
                &planned_profile.backend_id,
                &realized_profile.backend_id,
            ),
        ] {
            if planned_value != realized_value {
                return Err(StagePlanAdmissionError::ProfileIdentityMismatch {
                    profile_id: realized_profile.profile_id.clone(),
                    field,
                    planned: planned_value.clone(),
                    realized: realized_value.clone(),
                });
            }
        }
        for (field, planned_values, realized_values) in [
            (
                "activation_imports",
                &planned_profile.activation_imports,
                &realized_profile.activation_imports,
            ),
            (
                "activation_exports",
                &planned_profile.activation_exports,
                &realized_profile.activation_exports,
            ),
            (
                "activation_import_bindings",
                &planned_profile.activation_import_bindings,
                &realized_profile.activation_import_bindings,
            ),
            (
                "activation_export_bindings",
                &planned_profile.activation_export_bindings,
                &realized_profile.activation_export_bindings,
            ),
        ] {
            if planned_values != realized_values {
                return Err(StagePlanAdmissionError::ProfileIdentityMismatch {
                    profile_id: realized_profile.profile_id.clone(),
                    field,
                    planned: format!("{planned_values:?}"),
                    realized: format!("{realized_values:?}"),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
