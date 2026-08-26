//! Execution-model calibration against the measured anchors in
//! `docs/BENCHMARKS.md` (GLM-4.7-Flash-Q4_K_M on M4 Max + Mac mini over
//! Wi-Fi: solo 68 tok/s, 2-way split 21, 3-way split 12-13).
//!
//! Tolerance is +/-15% per anchor: BENCHMARKS.md itself calls the numbers
//! "a quick reality check". If these tests drift, either the model or the
//! calibration coefficients in the anchor scenario need updating — and a
//! model that cannot reproduce the anchors must not drive placement
//! decisions.

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
