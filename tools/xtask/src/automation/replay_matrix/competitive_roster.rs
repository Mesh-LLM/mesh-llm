//! Complete configured roster, explicit optional exclusions, and alternating trace order.
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Model {
    pub key: String,
    pub model: super::competitive_launch::Artifact,
    pub backends: BTreeMap<String, super::competitive_launch::Backend>,
}
pub(super) struct Roster {
    pub cells: Vec<Value>,
    pub availability: Value,
}
pub(super) fn select(
    config: &Value,
    bytes: &[u8],
    input: &super::competitive_matrix::Input,
) -> DynResult<Roster> {
    select_on(config, bytes, input, cfg!(target_os = "linux"))
}
pub(super) fn select_on(
    config: &Value,
    bytes: &[u8],
    input: &super::competitive_matrix::Input,
    linux: bool,
) -> DynResult<Roster> {
    admit_selection(config, input)?;
    let configured = config["models"].as_array().ok_or("models")?;
    let mut cells = Vec::new();
    let mut availability = Vec::new();
    for source in configured {
        let Some(model) = input.models.iter().find(|model| source["key"] == model.key) else {
            continue;
        };
        let arms = arms(source, model, input, &mut availability, linux)?;
        append_cells(&mut cells, config, bytes, input, model, &arms)?;
    }
    if cells.is_empty() || cells.len() > 100000 {
        return Err("empty or oversized competitive matrix".into());
    }
    Ok(Roster {
        cells,
        availability: json!({"scope":"declared_prepared_backend_inventory","models":availability}),
    })
}
fn arms(
    source: &Value,
    model: &Model,
    input: &super::competitive_matrix::Input,
    availability: &mut Vec<Value>,
    linux: bool,
) -> DynResult<Vec<String>> {
    let mut arms = vec!["llama".into(), "mesh".into()];
    if !model.backends.contains_key("llama") || !model.backends.contains_key("mesh") {
        return Err("selected model requires prepared raw and mesh backends".into());
    }
    if input.adaptive {
        if !model.backends.contains_key("mesh-adaptive") {
            return Err("adaptive selection requires prepared backend".into());
        }
        arms.push("mesh-adaptive".into());
    }
    for name in &input.optional_arms {
        let exclusion = if source["comparison_support"][name]["available"] == false {
            Some(
                source["comparison_support"][name]["reason"]
                    .as_str()
                    .ok_or("source exclusion reason")?
                    .to_owned(),
            )
        } else if !linux || input.platform != "cuda" {
            Some("optional engine requires Linux CUDA".into())
        } else if !model.backends.contains_key(name) {
            Some("no prepared backend supplied".into())
        } else {
            None
        };
        let pinned = source["comparison_support"][name]["available"] == false;
        if let Some(reason) = &exclusion
            && input.required_comparisons.contains(name)
            && !pinned
        {
            return Err(format!("required {name} unavailable: {reason}").into());
        }
        availability.push(json!({"model":model.key,"arm":name,"selected":exclusion.is_none(),"source_pinned_exclusion":pinned,"reason":exclusion}));
        if exclusion.is_none() {
            arms.push(name.clone());
        }
    }
    Ok(arms)
}

fn admit_selection(config: &Value, input: &super::competitive_matrix::Input) -> DynResult<()> {
    if !["cuda", "metal", "rocm"].contains(&input.platform.as_str())
        || input.models.is_empty()
        || input.workloads.is_empty()
    {
        return Err(
            "matrix requires one known platform, nonempty selected models/workloads".into(),
        );
    }
    let mut unique = BTreeSet::new();
    for workload in &input.workloads {
        if !["synthetic", "thoughtworks"].contains(&workload.as_str()) || !unique.insert(workload) {
            return Err("unknown or duplicate workload".into());
        }
    }
    unique.clear();
    for name in &input.optional_arms {
        if !["vllm", "sglang"].contains(&name.as_str()) || !unique.insert(name) {
            return Err("unknown or duplicate optional arm".into());
        }
    }
    if input
        .required_comparisons
        .iter()
        .any(|name| !input.optional_arms.contains(name))
    {
        return Err("required comparisons must be explicitly selected".into());
    }
    let configured = config["models"].as_array().ok_or("models")?;
    let mut seen = BTreeSet::new();
    for model in &input.models {
        if !seen.insert(&model.key) || !configured.iter().any(|source| source["key"] == model.key) {
            return Err("unknown or duplicate selected model".into());
        }
    }
    Ok(())
}
fn append_cells(
    cells: &mut Vec<Value>,
    config: &Value,
    bytes: &[u8],
    input: &super::competitive_matrix::Input,
    model: &Model,
    arms: &[String],
) -> DynResult<()> {
    let workload_refs: Vec<_> = input.workloads.iter().map(String::as_str).collect();
    let base = super::competitive_plan::build(
        config,
        bytes,
        &[&input.platform],
        &[&model.key],
        &workload_refs,
    )?;
    let base = base["cells"].as_array().ok_or("planned cells")?;
    for arm in arms {
        for cell in base
            .iter()
            .filter(|cell| cell["workload"] == "synthetic" && cell["arm"] == "llama")
        {
            let mut cell = cell.clone();
            cell["arm"] = arm.clone().into();
            cells.push(cell);
        }
    }
    for (trace_level, pair) in base
        .iter()
        .filter(|cell| cell["workload"] == "thoughtworks")
        .collect::<Vec<_>>()
        .chunks(2)
        .enumerate()
    {
        let reference = pair.first().ok_or("trace reference")?;
        let order: Vec<_> = if trace_level.is_multiple_of(2) {
            arms.iter().collect()
        } else {
            arms.iter().rev().collect()
        };
        for arm in order {
            let mut cell = (**reference).clone();
            cell["arm"] = arm.clone().into();
            cells.push(cell);
        }
    }
    Ok(())
}
