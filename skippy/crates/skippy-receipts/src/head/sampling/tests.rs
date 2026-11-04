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
    let TokenRule::Filtered {
        filters, biases, ..
    } = rule
    else {
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
        // a reasoning budget whose sampler the statement names badly
        serde_json::json!({"reasoning_budget": 128, "reasoning": {"start": ""}}),
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
    // a reasoning budget forces only the end of the reasoning, which the
    // head check replays: the rule holds, with or without a sampler
    for overrides in [
        serde_json::json!({"reasoning_budget": 128}),
        serde_json::json!({
            "reasoning_budget": 128,
            "reasoning": {"start": "<think>", "ends": ["</think>"], "prefill": ""},
        }),
    ] {
        let rule = TokenRule::from_sampler(&statement(overrides.clone()));
        assert!(
            matches!(rule, TokenRule::Filtered { .. }),
            "{overrides}: {rule:?}"
        );
    }
}

fn draw_of(rule: &TokenRule) -> &DrawRule {
    let TokenRule::Filtered { draw, .. } = rule else {
        panic!("not a filtered rule: {rule:?}");
    };
    draw
}

#[test]
fn a_statement_that_names_the_indexed_draw_has_it_replayed() {
    let named = serde_json::json!({ "draw": turn::INDEXED_DRAW });
    let rule = TokenRule::from_sampler(&statement(named.clone()));
    assert_eq!(
        draw_of(&rule),
        &DrawRule::Indexed(IndexedDraw {
            seed: 7,
            temperature: 0.8
        })
    );
    assert!(rule.replayed_draw().is_some());
    assert!(rule.describe().ends_with("picked by the draw of seed 7"));
    // a statement without the rule, as an older node writes it
    let rule = TokenRule::from_sampler(&statement(serde_json::json!({})));
    assert!(matches!(draw_of(&rule), DrawRule::Unjudged(_)));
    assert!(rule.replayed_draw().is_none());
    assert!(
        rule.describe().contains("draw not replayed"),
        "{}",
        rule.describe()
    );
    // the kept-token rule holds either way
    let mut logits = vec![-10.0; 100];
    logits[0] = 5.0;
    assert!(keeps(&rule, &logits, 0));
    assert!(!keeps(&rule, &logits, 50));
}

#[test]
fn a_draw_the_check_cannot_follow_is_not_replayed() {
    let with = |overrides: serde_json::Value| {
        let mut all = serde_json::json!({ "draw": turn::INDEXED_DRAW });
        for (k, v) in overrides.as_object().unwrap() {
            all[k] = v.clone();
        }
        TokenRule::from_sampler(&statement(all))
    };
    for (overrides, reason) in [
        (serde_json::json!({ "seed": null }), "seed"),
        (serde_json::json!({ "seed": 4_294_967_296u64 }), "seed"),
        (serde_json::json!({ "dynatemp_range": 0.5 }), "dynamic"),
        (serde_json::json!({ "ignore_eos": true }), "end token"),
        (serde_json::json!({ "draw": "another-rule" }), "no draw"),
    ] {
        let rule = with(overrides.clone());
        let DrawRule::Unjudged(why) = draw_of(&rule) else {
            panic!("{overrides}: replayed");
        };
        assert!(why.contains(reason), "{overrides}: {why}");
    }
    // a model that suppresses tokens the check does not read
    let rule = with(serde_json::json!({})).on_suppressed_tokens(true);
    assert!(matches!(draw_of(&rule), DrawRule::Unjudged(_)));
    assert!(
        with(serde_json::json!({}))
            .on_suppressed_tokens(false)
            .replayed_draw()
            .is_some()
    );
    // a turn that is not sampled from the chain has no draw to replay
    assert!(
        with(serde_json::json!({ "grammar": "abc" }))
            .replayed_draw()
            .is_none()
    );
}

#[test]
fn the_draw_sees_every_temperature_of_the_chain() {
    let chain = |samplers: serde_json::Value, temperature: f64| {
        let rule = TokenRule::from_sampler(&statement(serde_json::json!({
            "draw": turn::INDEXED_DRAW, "samplers": samplers, "temperature": temperature,
        })));
        rule.replayed_draw()
            .map(|replayed| replayed.draw.temperature)
    };
    let close = |a: Option<f64>, b: f64| a.is_some_and(|a| (a - b).abs() < 1e-12);
    assert!(close(
        chain(serde_json::json!(["top_k", "temperature"]), 0.5),
        0.5
    ));
    // a chain without the temperature sampler does not divide the logits
    assert!(close(
        chain(serde_json::json!(["top_k", "top_p"]), 0.5),
        1.0
    ));
    // a temperature before a filter still scales what the draw sees
    assert!(close(
        chain(serde_json::json!(["temperature", "top_k"]), 0.5),
        0.5
    ));
    // a temperature of 0 keeps the best logit alone: the turn draws nothing
    assert_eq!(
        chain(serde_json::json!(["top_k", "temperature"]), 0.0),
        None
    );
}

#[test]
fn a_turn_that_keeps_the_best_logit_alone_has_no_draw() {
    // a greedy request with a reasoning budget states a sampled turn at a
    // temperature of 0, and may name a draw its chain never runs
    let rule = TokenRule::from_sampler(&statement(serde_json::json!({
        "temperature": 0.0,
        "seed": 7,
        "draw": turn::INDEXED_DRAW,
        "reasoning_budget": 64,
    })));
    assert_eq!(draw_of(&rule), &DrawRule::Single);
    assert!(rule.replayed_draw().is_none());
    assert!(!rule.describe().contains("draw"), "{}", rule.describe());
    let logits = [3.0, 2.9, 1.0];
    assert!(keeps(&rule, &logits, 0));
    assert!(!keeps(&rule, &logits, 1));
}
