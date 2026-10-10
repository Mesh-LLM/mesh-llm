use super::document::Json;
use super::fields::{PlanResult, choice, exact, object};
use super::integer::Integer;
use super::projection::{ToJson, object as json_object};

pub(super) struct Execution {
    trunk_layers: Integer,
    pub(super) mtp_layers: Integer,
    activation_width: Integer,
    layer_end: Integer,
    speculative_policy: String,
}

impl ToJson for Execution {
    fn to_json(&self) -> Json {
        json_object([
            ("trunk_layers", self.trunk_layers.to_json()),
            ("mtp_layers", self.mtp_layers.to_json()),
            ("activation_width", self.activation_width.to_json()),
            ("layer_end", self.layer_end.to_json()),
            ("speculative_policy", self.speculative_policy.to_json()),
        ])
    }
}

pub(super) fn parse(value: Option<&Json>, field: &str, model_class: &str) -> PlanResult<Execution> {
    let execution_field = format!("{field}.execution");
    let row = object(value, &execution_field)?;
    exact(
        row,
        &[
            "trunk_layers",
            "mtp_layers",
            "activation_width",
            "speculative_policy",
        ],
        &execution_field,
    )?;
    let trunk_layers = Integer::parse(
        row.get("trunk_layers"),
        &format!("{execution_field}.trunk_layers"),
        1,
    )?;
    let mtp_layers = Integer::parse(
        row.get("mtp_layers"),
        &format!("{execution_field}.mtp_layers"),
        0,
    )?;
    let activation_width = Integer::parse(
        row.get("activation_width"),
        &format!("{execution_field}.activation_width"),
        1,
    )?;
    let layer_end = trunk_layers.sum(&mtp_layers);
    let speculative_policy = choice(
        row.get("speculative_policy"),
        &format!("{execution_field}.speculative_policy"),
        &["mtp-if-present", "disabled"],
    )?;
    if model_class != "causal_generation" && !mtp_layers.is_zero() {
        return Err(format!(
            "{field}.class {model_class} must not request split or MTP certification"
        ));
    }
    if model_class != "causal_generation" && speculative_policy != "disabled" {
        return Err(format!(
            "{field}.class {model_class} must disable speculative decoding"
        ));
    }
    Ok(Execution {
        trunk_layers,
        mtp_layers,
        activation_width,
        layer_end,
        speculative_policy,
    })
}
