use std::{env, io::Write};

use serde_json::json;
use skippy_topology::{
    BoundaryDecision, NodeSpec, PlannerPolicy, TopologyPlanRequest, dense_attention_layers,
    infer_family_capability, plan_balanced_accepted_contiguous, plan_contiguous_with_splits,
};

fn usage() -> Box<dyn std::error::Error> {
    "usage: skippy-topology-plan MODEL_ID LAYER_COUNT ACTIVATION_WIDTH".into()
}

fn parse_u32(value: Option<String>, name: &str) -> Result<u32, Box<dyn std::error::Error>> {
    value
        .ok_or_else(usage)?
        .parse::<u32>()
        .map_err(|_| format!("{name} must be a positive integer").into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model_id = args.next().ok_or_else(usage)?;
    let layer_count = parse_u32(args.next(), "LAYER_COUNT")?;
    let activation_width = parse_u32(args.next(), "ACTIVATION_WIDTH")?;
    if layer_count < 3 || activation_width == 0 || args.next().is_some() {
        return Err(usage());
    }

    // Product callers treat family capability as optional: known families add
    // reviewed constraints, while an unknown identity uses the generic
    // activation-only boundary contract. Keep canary planning identical.
    let family = infer_family_capability(&model_id, layer_count, activation_width);
    let request = TopologyPlanRequest {
        topology_id: "family-canary-preflight".to_string(),
        model_id: model_id.clone(),
        layers: dense_attention_layers(layer_count, 1),
        nodes: (0..3)
            .map(|index| NodeSpec {
                node_id: format!("canary-stage-{index}"),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect(),
        family: family.clone(),
        policy: PlannerPolicy::default(),
    };

    let boundaries = (1..layer_count)
        .map(|cut| {
            let plan = plan_contiguous_with_splits(&request, &[cut])?;
            let boundary = &plan.boundaries[0];
            Ok(json!({
                "layer": cut,
                "decision": if boundary.decision == BoundaryDecision::Accepted { "accepted" } else { "rejected" },
                "reason_codes": &boundary.reason_codes,
                "messages": &boundary.messages,
            }))
        })
        .collect::<Result<Vec<_>, skippy_topology::PlanError>>()?;
    let two_stage = plan_balanced_accepted_contiguous(&request, 2)?;
    let three_stage = plan_balanced_accepted_contiguous(&request, 3)?;
    let selected = |plan: &skippy_topology::TopologyPlan| {
        plan.boundaries
            .iter()
            .map(|boundary| boundary.layer_boundary)
            .collect::<Vec<_>>()
    };

    let output = json!({
        "schema_version": 1,
        "model_id": model_id,
        "family_id": family.as_ref().map(|record| &record.family_id),
        "capability_source": if family.is_some() { "reviewed-or-inferred" } else { "generic" },
        "layer_count": layer_count,
        "activation_width": activation_width,
        "boundaries": boundaries,
        "two_stage_splits": selected(&two_stage),
        "three_stage_splits": selected(&three_stage),
    });
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    serde_json::to_writer_pretty(&mut stdout, &output)?;
    stdout.write_all(b"\n")?;
    Ok(())
}
