//! Execution-model calibration against the measured anchors in
//! `docs/BENCHMARKS.md` (GLM-4.7-Flash-Q4_K_M on M4 Max + Mac mini over
//! Wi-Fi: solo 68 tok/s, 2-way split 21, 3-way split 12-13).
//!
//! Tolerance is +/-15% per anchor: BENCHMARKS.md itself calls the numbers
//! "a quick reality check". If these tests drift, either the model or the
//! calibration coefficients in the anchor scenario need updating — and a
//! model that cannot reproduce the anchors must not drive placement
//! decisions.
//!
//! `planner_model_matches_execution_sim` additionally locks the
//! *coordinator planner's* modeled decode TPOT to the calibrated execution
//! sim on the same scenario: the two cost models must agree, or the
//! planner will rank candidates against numbers nobody has validated.

use skippy_topology_sim::Scenario;

fn load(name: &str) -> Scenario {
    let path = format!("{}/scenarios/{name}", env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    Scenario::from_toml(&raw).unwrap_or_else(|error| panic!("parse {path}: {error}"))
}

fn assert_within(anchor: f64, modeled: f64, label: &str) {
    let tolerance = anchor * 0.15;
    assert!(
        (modeled - anchor).abs() <= tolerance,
        "{label}: modeled {modeled:.1} tok/s vs anchor {anchor} (±{tolerance:.1})"
    );
}

#[test]
fn reproduces_solo_anchor() {
    let scenario = load("benchmarks_anchor_pair.toml");
    let estimate = scenario
        .estimate_execution(&[("m4max", 40)])
        .expect("execution estimate");
    assert_within(68.0, estimate.serial_tok_s, "solo M4 Max");
}

#[test]
fn reproduces_two_way_split_anchor() {
    let scenario = load("benchmarks_anchor_pair.toml");
    // BENCHMARKS.md 2-way split is 85/15: 34/6 of 40 layers.
    let estimate = scenario
        .estimate_execution(&[("m4max", 34), ("mini", 6)])
        .expect("execution estimate");
    assert_within(21.0, estimate.serial_tok_s, "2-way split");
    // The serial regime must be strictly slower than the pipelined
    // estimate — that difference is why splits cost single-stream decode.
    assert!(estimate.serial_tok_s < estimate.pipelined_tok_s_per_lane.unwrap());
}

#[test]
fn reproduces_three_way_split_anchor() {
    let scenario = load("benchmarks_anchor_pair.toml");
    // BENCHMARKS.md 3-way split is 62/31/8: 25/12/3 of 40 layers.
    let estimate = scenario
        .estimate_execution(&[("m4max", 25), ("mini", 12), ("mini2", 3)])
        .expect("execution estimate");
    assert_within(13.0, estimate.serial_tok_s, "3-way split (12-13 anchor)");
}

#[test]
fn monotonically_worse_with_more_hops() {
    // Sanity property: adding hops must never improve single-stream decode.
    let scenario = load("benchmarks_anchor_pair.toml");
    let solo = scenario.estimate_execution(&[("m4max", 40)]).unwrap();
    let two = scenario
        .estimate_execution(&[("m4max", 34), ("mini", 6)])
        .unwrap();
    let three = scenario
        .estimate_execution(&[("m4max", 25), ("mini", 12), ("mini2", 3)])
        .unwrap();
    assert!(solo.serial_tok_s > two.serial_tok_s);
    assert!(two.serial_tok_s > three.serial_tok_s);
}

#[test]
fn planner_model_matches_execution_sim() {
    // The coordinator planner's modeled decode TPOT must equal the
    // calibrated execution sim's serial estimate for the same stage
    // assignment. The planner previously used a different formula
    // (bottleneck-stage + network) with no calibrated overhead terms and
    // no calibration test at all — this locks the two cost models together
    // on the anchor scenario so a future divergence fails CI instead of
    // silently mis-ranking candidates.
    let scenario = load("benchmarks_anchor_pair.toml");
    let plan = scenario.plan().expect("planner plan");
    let chosen: Vec<(&str, u32)> = plan
        .stages
        .iter()
        .map(|stage| (stage.node_id.as_str(), stage.layer_end - stage.layer_start))
        .collect();
    let sim = scenario
        .estimate_execution(&chosen)
        .expect("sim estimate for the planner-chosen assignment");
    let planner_us = plan
        .modeled_decode_tpot_us
        .expect("planner models TPOT for the anchor scenario (all nodes signaled)");
    let sim_us = u128::from(sim.serial_token_us);
    // Both models must agree within 1% — they use the same terms; any
    // larger gap is a formula divergence, not calibration noise.
    let tolerance = sim_us / 100;
    assert!(
        planner_us.abs_diff(sim_us) <= tolerance,
        "planner TPOT {planner_us} µs vs sim {sim_us} µs for {chosen:?} (divergence beyond 1%)"
    );
}
