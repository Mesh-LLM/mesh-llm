use super::{
    args::NAMES,
    types::{Identity, Observer, Ready, Snapshot, Stage},
};
use crate::automation::codepoint_json::value::Value;

pub(super) const KIND: &str = "mesh-llm-two-node-split-readiness";

pub(super) fn object<const COUNT: usize>(entries: [(&str, Value); COUNT]) -> Value {
    Value::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
    )
}

pub(super) fn string(text: &str) -> Value {
    Value::Str(text.into())
}

fn identity(identity: &Identity) -> Vec<(&str, Value)> {
    vec![
        ("topology_id", identity.topology_id.value()),
        ("run_id", identity.run_id.value()),
        ("model_id", identity.model_id.value()),
        ("package_ref", identity.package_ref.value()),
        ("manifest_sha256", identity.manifest_sha256.value()),
    ]
}

fn stage(stage: &Stage) -> Value {
    object([
        ("stage_id", stage.stage_id.value()),
        ("stage_index", stage.stage_index.value()),
        ("node_id", stage.node_id.value()),
        ("layer_start", stage.layer_start.value()),
        ("layer_end", stage.layer_end.value()),
        ("endpoint", object([("bind_addr", stage.bind_addr.value())])),
        ("state", string("ready")),
    ])
}

fn observer(observer: &Observer) -> Value {
    object([
        ("node_id", observer.node_id.value()),
        (
            "peer_node_ids",
            Value::Array(vec![observer.peer_node_id.value()]),
        ),
    ])
}

pub(super) fn ready(ready: &Ready, snapshots: &[Snapshot; 6], label: &str) -> Value {
    let [first, second] = &ready.topology.stages;
    let mut topology = identity(&ready.topology.identity);
    topology.extend([
        ("layer_start", first.layer_start.value()),
        ("layer_end", second.layer_end.value()),
        (
            "stages",
            Value::Array(ready.topology.stages.iter().map(stage).collect()),
        ),
    ]);
    object([
        ("schema_version", Value::Int(1)),
        ("kind", string(KIND)),
        ("status", string("ready")),
        ("model_label", string(label)),
        ("model_id", ready.model.value()),
        (
            "topology",
            Value::Object(
                topology
                    .into_iter()
                    .map(|(key, value)| (key.into(), value))
                    .collect(),
            ),
        ),
        (
            "observers",
            object([
                ("mesh_id", ready.seed.mesh_id.value()),
                ("seed", observer(&ready.seed)),
                ("worker", observer(&ready.worker)),
            ]),
        ),
        (
            "snapshots",
            Value::Object(
                NAMES
                    .into_iter()
                    .zip(snapshots)
                    .map(|(name, snapshot)| {
                        (
                            name.into(),
                            object([
                                ("path", snapshot.basename.value()),
                                ("sha256", string(&snapshot.sha256)),
                            ]),
                        )
                    })
                    .collect(),
            ),
        ),
        ("errors", Value::Array(Vec::new())),
    ])
}

pub(super) fn failed(label: &str, reason: &str) -> Value {
    object([
        ("schema_version", Value::Int(1)),
        ("kind", string(KIND)),
        ("status", string("failed")),
        ("model_label", string(label)),
        ("errors", Value::Array(vec![string(reason)])),
    ])
}
