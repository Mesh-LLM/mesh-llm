use super::{
    Error,
    boundary::{array, identity, integer, object, text},
    types::{Stage, Status, Topology},
};
use crate::automation::codepoint_json::value::Value;

fn stage(value: &Value, label: &str, nested: bool) -> Result<Stage, Error> {
    let value = object(value, label)?;
    let null = Value::Null;
    let endpoint = if nested {
        Some(object(
            value.get("endpoint").unwrap_or(&null),
            &format!("{label}.endpoint"),
        )?)
    } else {
        None
    };
    Ok(Stage {
        stage_id: text(value.get("stage_id"), &format!("{label}.stage_id"))?,
        stage_index: integer(value, "stage_index", label)?,
        node_id: text(value.get("node_id"), &format!("{label}.node_id"))?,
        layer_start: integer(value, "layer_start", label)?,
        layer_end: integer(value, "layer_end", label)?,
        bind_addr: match endpoint {
            Some(endpoint) => text(
                endpoint.get("bind_addr"),
                &format!("{label}.endpoint.bind_addr"),
            )?,
            None => text(value.get("bind_addr"), &format!("{label}.bind_addr"))?,
        },
    })
}

pub(super) fn topology(value: &Value, label: &str) -> Result<Topology, Error> {
    let items = array(value.get("topologies"), &format!("{label}.topologies"))?;
    let [value] = items else {
        return Err(Error::Contract(format!(
            "{label}.topologies must contain exactly one topology, got {}",
            items.len()
        )));
    };
    let item_label = format!("{label}.topologies[0]");
    let value = object(value, &item_label)?;
    let items = array(value.get("stages"), &format!("{item_label}.stages"))?;
    if items.len() != 2 {
        return Err(Error::Contract(format!(
            "{label} topology must contain exactly two stages, got {}",
            items.len()
        )));
    }
    let mut stages = items
        .iter()
        .enumerate()
        .map(|(index, value)| stage(value, &format!("{item_label}.stages[{index}]"), true))
        .collect::<Result<Vec<_>, Error>>()?;
    stages.sort_by(|left, right| left.stage_index.cmp(&right.stage_index));
    let stages: [Stage; 2] = stages
        .try_into()
        .map_err(|_| Error::Contract("expected two stages".into()))?;
    let [first, second] = &stages;
    let failure = if !first.stage_index.equals(0) || !second.stage_index.equals(1) {
        Some("stage indexes must be exactly [0, 1]")
    } else if !first.layer_start.equals(0) {
        Some("must start at layer 0")
    } else if first.layer_start >= first.layer_end || second.layer_start >= second.layer_end {
        Some("stages must have non-empty layer ranges")
    } else if first.layer_end != second.layer_start {
        Some("layer ranges must be contiguous")
    } else if first.stage_id == second.stage_id {
        Some("stage IDs must be distinct")
    } else if first.node_id == second.node_id {
        Some("stage nodes must be distinct")
    } else {
        None
    };
    if let Some(reason) = failure {
        return Err(Error::Contract(format!("{label} topology {reason}")));
    }
    Ok(Topology {
        identity: identity(value, &item_label)?,
        stages,
    })
}

pub(super) fn statuses(value: &Value, label: &str) -> Result<[Status; 2], Error> {
    let items = array(value.get("statuses"), &format!("{label}.statuses"))?;
    if items.len() != 2 {
        return Err(Error::Contract(format!(
            "{label}.statuses must contain exactly two stage statuses, got {}",
            items.len()
        )));
    }
    let mut statuses = items
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let label = format!("{label}.statuses[{index}]");
            let value = object(value, &label)?;
            Ok(Status {
                identity: identity(value, &label)?,
                stage: stage(value, &label, false)?,
                state: text(value.get("state"), &format!("{label}.state"))?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    statuses.sort_by(|left, right| left.stage.stage_index.cmp(&right.stage.stage_index));
    let statuses: [Status; 2] = statuses
        .try_into()
        .map_err(|_| Error::Contract("expected two statuses".into()))?;
    let [first, second] = &statuses;
    if !first.stage.stage_index.equals(0) || !second.stage.stage_index.equals(1) {
        return Err(Error::Contract(format!(
            "{label} status indexes must be exactly [0, 1]"
        )));
    }
    if first.state.0 != *"ready" || second.state.0 != *"ready" {
        let states = [&first.state, &second.state]
            .map(|state| crate::repository::python_text::repr(&state.display()));
        return Err(Error::Contract(format!(
            "{label} stage statuses must both be ready, got [{}, {}]",
            states[0], states[1]
        )));
    }
    Ok(statuses)
}
