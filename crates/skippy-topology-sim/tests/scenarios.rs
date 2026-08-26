//! Corpus scenarios run end-to-end through the real planner. Each asserts
//! the behavioral property the scenario exists to guard (see the design
//! doc's scenario corpus section).

use skippy_topology_sim::Scenario;

fn load(name: &str) -> Scenario {
    let path = format!("{}/scenarios/{name}", env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    Scenario::from_toml(&raw).unwrap_or_else(|error| panic!("parse {path}: {error}"))
}

#[test]
fn heterogeneous_pair_splits_proportionally() {
    let scenario = load("heterogeneous_pair.toml");
    let plan = scenario.plan().expect("plan");
    assert_eq!(plan.stages.len(), 2);
    let span = |id: &str| {
        plan.stages
            .iter()
            .find(|stage| stage.node_id == id)
            .map(|stage| stage.layer_end - stage.layer_start)
            .unwrap_or(0)
    };
    let (alpha, beta) = (span("alpha"), span("beta"));
    assert_eq!(alpha + beta, 40);
    assert!(
        u64::from(alpha) >= 2 * u64::from(beta),
        "alpha (2.7x bandwidth) should earn >= 2x beta's layers: {alpha}/{beta}"
    );
}

#[test]
fn straggler_triplet_limits_the_laptop() {
    let scenario = load("straggler_triplet.toml");
    let plan = scenario.plan().expect("plan");
    let laptop = plan
        .stages
        .iter()
        .find(|stage| stage.node_id == "laptop")
        .expect("laptop participates");
    let laptop_layers = laptop.layer_end - laptop.layer_start;
    assert!(
        laptop_layers <= 4,
        "laptop (80 GB/s) must receive at most a few layers: {laptop_layers}"
    );
}

#[test]
fn cross_continent_chain_fails_tpot_target_honestly() {
    let scenario = load("cross_continent_chain.toml");
    let plan = scenario.plan().expect("plan");
    let estimate = plan
        .estimated_decode_network_ms_per_token
        .expect("latency-aware plan carries an estimate");
    assert!(
        estimate > 33,
        "70-140 ms hops must yield an estimate above the 33 ms target: {estimate}"
    );
    assert_eq!(
        plan.decode_tpot_target_met,
        Some(false),
        "the plan must not claim the TPOT target is met"
    );
}
