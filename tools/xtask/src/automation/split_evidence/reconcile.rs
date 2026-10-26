use super::{
    Error, boundary, topology,
    types::{Ready, Snapshot, Stage, Status},
};

pub(super) fn reconcile(snapshots: &[Snapshot; 6]) -> Result<Ready, Error> {
    let [
        seed_status,
        seed_stages,
        seed_models,
        worker_status,
        worker_stages,
        worker_models,
    ] = snapshots;
    let seed = boundary::observer_identity(&seed_status.payload, "seed_status")?;
    let worker = boundary::observer_identity(&worker_status.payload, "worker_status")?;
    if seed.0 == worker.0 {
        return Err(Error::Contract(
            "seed and worker observers must have distinct node IDs".into(),
        ));
    }
    if seed.1 != worker.1 {
        return Err(Error::Contract(
            "seed and worker observers must share the same mesh ID".into(),
        ));
    }
    let seed = boundary::observer(&seed_status.payload, seed, "seed_status")?;
    let worker = boundary::observer(&worker_status.payload, worker, "worker_status")?;
    if seed.peer_node_id != worker.node_id || worker.peer_node_id != seed.node_id {
        return Err(Error::Contract(
            "seed and worker status snapshots must identify each other as their sole peer".into(),
        ));
    }
    let topology = topology::topology(&seed_stages.payload, "seed_stages")?;
    let worker_topology = topology::topology(&worker_stages.payload, "worker_stages")?;
    if topology != worker_topology {
        return Err(Error::Contract(
            "seed and worker topology snapshots do not match exactly".into(),
        ));
    }
    let statuses = topology::statuses(&seed_stages.payload, "seed_stages")?;
    let worker_statuses = topology::statuses(&worker_stages.payload, "worker_stages")?;
    if statuses != worker_statuses {
        return Err(Error::Contract(
            "seed and worker stage status snapshots do not match exactly".into(),
        ));
    }
    for (stage, status) in topology.stages.iter().zip(&statuses) {
        if status.identity != topology.identity {
            return Err(Error::Contract(format!(
                "stage status {} does not match the common topology/run/model/package/manifest",
                status.stage.stage_id.display()
            )));
        }
        match_stage(stage, status)?;
    }
    for (label, observer) in [("seed", &seed), ("worker", &worker)] {
        if topology
            .stages
            .iter()
            .filter(|stage| stage.node_id.starts_with(&observer.node_id))
            .count()
            != 1
        {
            return Err(Error::Contract(format!(
                "{label} observer node ID must match exactly one topology stage node"
            )));
        }
    }
    let model = boundary::model(&seed_models.payload, "seed_models")?;
    let worker_model = boundary::model(&worker_models.payload, "worker_models")?;
    if model != worker_model || model != topology.identity.model_id {
        return Err(Error::Contract(
            "seed, worker, and topology model IDs must match exactly".into(),
        ));
    }
    Ok(Ready {
        topology,
        seed,
        worker,
        model,
    })
}

fn match_stage(stage: &Stage, status: &Status) -> Result<(), Error> {
    for (field, matches) in [
        ("stage_id", stage.stage_id == status.stage.stage_id),
        ("stage_index", stage.stage_index == status.stage.stage_index),
        ("node_id", stage.node_id == status.stage.node_id),
        ("layer_start", stage.layer_start == status.stage.layer_start),
        ("layer_end", stage.layer_end == status.stage.layer_end),
    ] {
        if !matches {
            return Err(Error::Contract(format!(
                "stage status {} does not match topology field {field}",
                status.stage.stage_id.display()
            )));
        }
    }
    if stage.bind_addr != status.stage.bind_addr {
        return Err(Error::Contract(format!(
            "stage status {} does not match topology bind_addr",
            status.stage.stage_id.display()
        )));
    }
    Ok(())
}
