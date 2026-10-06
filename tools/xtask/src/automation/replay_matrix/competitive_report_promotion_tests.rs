use super::*;
use serde_json::json;
fn fixture(arms: &[(&str, f64, f64)]) -> (Value, Vec<Value>, Vec<Value>, Vec<Value>) {
    let source = json!({"concurrency":[1,2],"synthetic":{"output_tokens":[8,64]}});
    let mut plan = Vec::new();
    let mut rows = Vec::new();
    let mut gates = Vec::new();
    for &(arm, synthetic, trace) in arms {
        for workload in ["synthetic", "thoughtworks"] {
            for concurrency in [1, 2] {
                let outputs = if workload == "synthetic" {
                    vec![8, 64]
                } else {
                    vec![256]
                };
                for output in outputs {
                    let cell = json!({"platform":"metal","model":"dense","workload":workload,"arm":arm,"concurrency":concurrency,"output_tokens":output});
                    plan.push(cell.clone());
                    rows.push(json!({"cell":cell,"throughput":100.0*if workload=="synthetic"{synthetic}else{trace},"complete":true,"capacity_policy":{"mode":"declared-shared-context","comparison_kv_matched":arm=="mesh"}}));
                    if workload == "synthetic"
                        && concurrency == 1
                        && arm != "mesh"
                        && arm != "llama"
                    {
                        gates.push(json!({"cell":cell,"passed":true}));
                    }
                }
            }
        }
    }
    (source, plan, rows, gates)
}
fn report(f: &(Value, Vec<Value>, Vec<Value>, Vec<Value>)) -> Value {
    serde_json::to_value(evaluate(&f.0, &f.1, &f.2, &f.3).unwrap()).unwrap()
}
#[test]
fn promotion_selects_highest_combined_gain_and_lexical_ties_without_cross_policy_matching() {
    let f = fixture(&[
        ("mesh", 1.0, 1.0),
        ("mesh-adaptive", 1.1, 1.2),
        ("sglang", 1.2, 1.2),
    ]);
    let r = report(&f);
    assert_eq!(r[0]["winner"], "sglang");
    assert!(
        r[0]["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v["eligible"] == true)
    );
    assert_ne!(
        r[0]["candidates"][0]["capacity_policy"],
        r[0]["candidates"][0]["baseline_capacity_policy"]
    );
    let ties = fixture(&[
        ("mesh", 1.0, 1.0),
        ("sglang", 1.2, 1.2),
        ("mesh-adaptive", 1.2, 1.2),
    ]);
    assert_eq!(report(&ties)[0]["winner"], "mesh-adaptive");
}
#[test]
fn promotion_missing_candidate_baseline_and_partial_full_roster_hold() {
    let base = fixture(&[("mesh", 1.0, 1.0), ("mesh-adaptive", 1.2, 1.2)]);
    for (arm, workload, partial) in [
        ("mesh", "synthetic", false),
        ("mesh-adaptive", "thoughtworks", false),
        ("mesh-adaptive", "synthetic", true),
    ] {
        let mut f = base.clone();
        let index =
            f.2.iter()
                .position(|r| r["cell"]["arm"] == arm && r["cell"]["workload"] == workload)
                .unwrap();
        if partial {
            f.2[index]["complete"] = json!(false);
        } else {
            f.2.remove(index);
        }
        let r = report(&f);
        assert!(r[0]["winner"].is_null());
        assert_eq!(r[0]["candidates"][0]["complete"], false);
        assert_eq!(r[0]["candidates"][0]["eligible"], false);
    }
    let mut f = base;
    f.1.retain(|c| c["workload"] != "thoughtworks");
    f.2.retain(|r| r["cell"]["workload"] != "thoughtworks");
    assert!(report(&f)[0]["winner"].is_null());
}
#[test]
fn promotion_c1_failed_missing_or_duplicate_paired_gates_hold() {
    for mode in ["failed", "missing", "duplicate"] {
        let mut f = fixture(&[("mesh", 1.0, 1.0), ("mesh-adaptive", 1.2, 1.2)]);
        match mode {
            "failed" => f.3[0]["passed"] = json!(false),
            "missing" => {
                f.3.pop();
            }
            _ => f.3.push(f.3[0].clone()),
        }
        let r = report(&f);
        assert!(r[0]["winner"].is_null());
        assert_eq!(r[0]["candidates"][0]["complete"], true);
        assert_eq!(r[0]["candidates"][0]["c1_parity"], false);
    }
}
#[test]
fn promotion_requires_strict_positive_mean_in_both_workloads() {
    for (synthetic, trace) in [(1.4, 0.9), (0.9, 1.4), (1.0, 1.4), (1.4, 1.0)] {
        let f = fixture(&[("mesh", 1.0, 1.0), ("mesh-adaptive", synthetic, trace)]);
        let r = report(&f);
        assert!(r[0]["winner"].is_null());
        assert_eq!(r[0]["candidates"][0]["eligible"], false);
    }
}
#[test]
fn promotion_rejects_foreign_duplicate_nonfinite_arithmetic_and_mixed_policy_cohorts() {
    let base = fixture(&[("mesh", 1.0, 1.0), ("mesh-adaptive", 1.2, 1.2)]);
    let mut duplicate = base.clone();
    duplicate.2.push(duplicate.2[0].clone());
    assert!(evaluate(&duplicate.0, &duplicate.1, &duplicate.2, &duplicate.3).is_err());
    let mut foreign = base.clone();
    foreign.2[0]["cell"]["platform"] = json!("cuda");
    assert!(evaluate(&foreign.0, &foreign.1, &foreign.2, &foreign.3).is_err());
    let mut overflow = base.clone();
    for row in &mut overflow.2 {
        row["throughput"] = json!(if row["cell"]["arm"] == "mesh" {
            1e-300
        } else {
            1e308
        });
    }
    assert!(evaluate(&overflow.0, &overflow.1, &overflow.2, &overflow.3).is_err());
    let mut mixed = base;
    mixed.2.last_mut().unwrap()["capacity_policy"]["comparison_kv_matched"] = json!(true);
    let r = report(&mixed);
    assert!(r[0]["winner"].is_null());
    assert_eq!(r[0]["candidates"][0]["capacity_consistent"], false);
}
#[test]
fn promotion_json_and_markdown_preserve_per_platform_decisions_and_explicit_hold() {
    let (source, mut plan, mut rows, mut gates) =
        fixture(&[("mesh", 1.0, 1.0), ("mesh-adaptive", 1.2, 1.2)]);
    let (_, mut other_plan, mut other_rows, mut other_gates) =
        fixture(&[("mesh", 1.0, 1.0), ("mesh-adaptive", 0.9, 1.2)]);
    for c in &mut other_plan {
        c["platform"] = json!("cuda");
    }
    for r in &mut other_rows {
        r["cell"]["platform"] = json!("cuda");
    }
    for g in &mut other_gates {
        g["cell"]["platform"] = json!("cuda");
    }
    plan.extend(other_plan);
    rows.extend(other_rows);
    gates.extend(other_gates);
    let groups = evaluate(&source, &plan, &rows, &gates).unwrap();
    let r = serde_json::to_value(&groups).unwrap();
    assert!(r[0]["winner"].is_null());
    assert_eq!(r[1]["winner"], "mesh-adaptive");
    let text = markdown(&groups);
    assert!(text.contains("PROMOTION CANDIDATE"));
    assert!(text.contains("hold"));
    assert!(text.contains("no cross-platform/model aggregation"));
}
