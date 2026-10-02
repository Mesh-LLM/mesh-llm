//! Controller scheduling projection. Canonical source plan bytes stay unchanged.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const GIB: u64 = 1024 * 1024 * 1024;
const TIERS: [u64; 2] = [128, 256];

#[derive(Deserialize)]
struct Plan {
    selected_models: Vec<Model>,
    github_matrix: InputMatrix,
}
#[derive(Deserialize)]
struct Model {
    family: String,
    #[serde(rename = "class")]
    workload: Workload,
    artifact: Artifact,
    draft_artifact: Option<Artifact>,
    mmproj_artifact: Option<Artifact>,
    resources: Resources,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Workload {
    CausalGeneration,
    Embedding,
    Rerank,
    EncoderDecoder,
    Ocr,
    SpeechSynthesis,
    SpeechRecognition,
}
#[derive(Deserialize)]
struct Artifact {
    files: Vec<String>,
    file_integrity: BTreeMap<String, Integrity>,
}
#[derive(Deserialize)]
struct Integrity {
    size_bytes: u64,
}
#[derive(Deserialize)]
struct Resources {
    estimated_model_bytes: u64,
    minimum_runner_memory_gib: Option<u64>,
}
#[derive(Deserialize)]
struct InputMatrix {
    include: Vec<Row>,
}
#[derive(Deserialize, Serialize)]
struct Row {
    id: String,
    shard_index: u64,
    families: String,
    estimated_work_bytes: u64,
}
#[derive(Serialize)]
pub(crate) struct Matrix {
    pub(crate) include: Vec<ScheduledRow>,
}
#[derive(Serialize)]
pub(crate) struct ScheduledRow {
    #[serde(flatten)]
    row: Row,
    #[serde(flatten)]
    placement: Placement,
}
#[derive(Serialize)]
struct Placement {
    resident_model_bytes: u64,
    runtime_allowance_bytes: u64,
    estimated_peak_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    minimum_runner_memory_gib: Option<u64>,
    memory_tier: String,
}

pub(crate) fn project(bytes: &[u8]) -> DynResult<Matrix> {
    super::SourceFamilyPlan::parse(bytes)?;
    let plan: Plan = serde_json::from_slice(bytes)?;
    let placements = plan
        .selected_models
        .into_iter()
        .map(|model| {
            let placement =
                estimate(&model).map_err(|error| format!("{}: {error}", model.family))?;
            Ok((model.family, placement))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let mut placements = placements;
    let mut include = Vec::new();
    for row in plan.github_matrix.include {
        let placement = placements
            .remove(&row.families)
            .ok_or("unplanned scheduling family")?;
        include.push(ScheduledRow { row, placement });
    }
    include.sort_by(|left, right| {
        (left.row.estimated_work_bytes, &left.row.families)
            .cmp(&(right.row.estimated_work_bytes, &right.row.families))
    });
    Ok(Matrix { include })
}

fn estimate(model: &Model) -> DynResult<Placement> {
    positive(model.resources.estimated_model_bytes)?;
    let weights = artifact_bytes(&model.artifact)?.max(model.resources.estimated_model_bytes);
    let mut auxiliaries = 0_u64;
    for artifact in [&model.draft_artifact, &model.mmproj_artifact]
        .into_iter()
        .flatten()
    {
        auxiliaries = add(auxiliaries, artifact_bytes(artifact)?)?;
    }
    let (copies, processes) = match model.workload {
        Workload::CausalGeneration => (1, 3),
        Workload::Embedding
        | Workload::Rerank
        | Workload::EncoderDecoder
        | Workload::Ocr
        | Workload::SpeechSynthesis
        | Workload::SpeechRecognition => (2, 2),
    };
    let resident_model_bytes = add(weights, auxiliaries)?
        .checked_mul(copies)
        .ok_or("memory estimate overflow")?;
    let runtime_allowance_bytes = add(add(resident_model_bytes, 3)? / 4, processes * 2 * GIB)?;
    let estimated_peak_bytes = add(resident_model_bytes, runtime_allowance_bytes)?;
    let tier = tier_for(
        estimated_peak_bytes,
        model.resources.minimum_runner_memory_gib,
    )?;
    Ok(Placement {
        resident_model_bytes,
        runtime_allowance_bytes,
        estimated_peak_bytes,
        minimum_runner_memory_gib: model.resources.minimum_runner_memory_gib,
        memory_tier: format!("accelerator-memory-{tier}plus"),
    })
}

fn artifact_bytes(artifact: &Artifact) -> DynResult<u64> {
    if artifact.files.is_empty()
        || artifact.files.iter().collect::<BTreeSet<_>>().len() != artifact.files.len()
    {
        return Err("missing or duplicate pinned artifact files".into());
    }
    artifact.files.iter().try_fold(0_u64, |total, name| {
        let size = artifact
            .file_integrity
            .get(name)
            .ok_or("missing pinned artifact size")?
            .size_bytes;
        positive(size)?;
        add(total, size)
    })
}

fn tier_for(peak: u64, minimum: Option<u64>) -> DynResult<u64> {
    positive(peak)?;
    if minimum.is_some_and(|tier| !TIERS.contains(&tier)) {
        return Err("minimum runner memory must be 128 or 256 GiB".into());
    }
    let estimated = TIERS
        .into_iter()
        .find(|tier| peak <= tier * GIB * 90 / 100)
        .ok_or("estimated peak exceeds largest runner budget (230.4 GiB)")?;
    Ok(estimated.max(minimum.unwrap_or(estimated)))
}

fn positive(bytes: u64) -> DynResult<()> {
    if bytes == 0 {
        return Err("memory estimate requires positive pinned sizes".into());
    }
    Ok(())
}
fn add(left: u64, right: u64) -> DynResult<u64> {
    left.checked_add(right)
        .ok_or_else(|| "memory estimate overflow".into())
}
