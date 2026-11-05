use super::*;

fn statement(overrides: serde_json::Value) -> String {
    let mut p = serde_json::json!({
        "seed": 7,
        "samplers": ["penalties", "dry", "top_n_sigma", "top_k", "typical_p", "top_p", "min_p", "xtc", "temperature"],
        "temperature": 0.8,
        "dynatemp_range": 0.0,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.05,
        "typical_p": 1.0,
        "top_nsigma": -1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_penalty": 1.0,
        "dry": { "multiplier": 0.0 },
        "xtc": { "probability": 0.0, "threshold": 0.1 },
        "mirostat": { "mode": 0 },
        "logit_bias": [],
        "grammar": null,
    });
    for (k, v) in overrides.as_object().unwrap() {
        p[k] = v.clone();
    }
    turn::sampled(&p)
}

fn keeps(rule: &TokenRule, logits: &[f64], sampled: usize) -> bool {
    let TokenRule::Filtered { filters, biases } = rule else {
        panic!("not a filtered rule: {rule:?}");
    };
    TokenRule::keeps(filters, biases, logits, sampled, 1e-6)
}

#[test]
fn the_default_chain_keeps_likely_tokens_and_drops_the_tail() {
    let rule = TokenRule::from_sampler(&statement(serde_json::json!({})));
    // two likely tokens and a long unlikely tail
    let mut logits = vec![-10.0; 100];
    logits[0] = 5.0;
    logits[1] = 4.5;
    assert!(keeps(&rule, &logits, 0));
    assert!(keeps(&rule, &logits, 1));
    assert!(!keeps(&rule, &logits, 50));
}

#[test]
fn top_k_counts_better_tokens() {
    let rule = TokenRule::from_sampler(&statement(
        serde_json::json!({"top_k": 2, "top_p": 1.0, "min_p": 0.0}),
    ));
    let logits = [3.0, 2.0, 1.0, 0.0];
    assert!(keeps(&rule, &logits, 1));
    assert!(!keeps(&rule, &logits, 2));
}

#[test]
fn a_bias_moves_a_token_into_the_candidates() {
    let rule = TokenRule::from_sampler(&statement(
        serde_json::json!({"top_k": 1, "logit_bias": [[2, 10.0]]}),
    ));
    let logits = [3.0, 2.0, 1.0];
    assert!(keeps(&rule, &logits, 2));
    assert!(!keeps(&rule, &logits, 0));
}

#[test]
fn a_zero_temperature_keeps_only_the_best() {
    let rule = TokenRule::from_sampler(&statement(
        serde_json::json!({"temperature": 0.0, "logit_bias": [[1, 0.5]]}),
    ));
    let logits = [3.0, 2.9, 1.0];
    assert!(keeps(&rule, &logits, 1));
    assert!(!keeps(&rule, &logits, 0));
}

#[test]
fn min_p_after_temperature_uses_the_scaled_logits() {
    let samplers = serde_json::json!(["temperature", "min_p"]);
    let rule = TokenRule::from_sampler(&statement(
        serde_json::json!({"samplers": samplers, "temperature": 0.5, "min_p": 0.1}),
    ));
    // ln(0.1) = -2.3: unscaled, 2.0 below the best is kept; at half the
    // temperature it is 4.0 below
    let logits = [0.0, -2.0];
    assert!(!keeps(&rule, &logits, 1));
}

#[test]
fn penalties_grammar_and_unknown_samplers_are_not_judged() {
    for overrides in [
        serde_json::json!({"repeat_penalty": 1.1}),
        serde_json::json!({"grammar": "abc"}),
        serde_json::json!({"mirostat": {"mode": 2}}),
        serde_json::json!({"samplers": ["top_k", "mystery"]}),
        serde_json::json!({"reasoning_budget": 128}),
        serde_json::json!({"dynatemp_range": 0.5, "samplers": ["temperature", "top_p"]}),
    ] {
        let rule = TokenRule::from_sampler(&statement(overrides.clone()));
        assert!(
            matches!(rule, TokenRule::Unjudged(_)),
            "{overrides}: {rule:?}"
        );
    }
    // a dynamic temperature at the end only shapes the draw
    let rule = TokenRule::from_sampler(&statement(serde_json::json!({"dynatemp_range": 0.5})));
    assert!(matches!(rule, TokenRule::Filtered { .. }), "{rule:?}");
    assert_eq!(TokenRule::from_sampler(turn::GREEDY), TokenRule::Greedy);
}
