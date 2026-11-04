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
        (serde_json::json!({ "dynatemp_range": 0.5 }), "dynamic"),
        (serde_json::json!({ "draw": "another-rule" }), "no draw"),
    ] {
        let rule = with(overrides.clone());
        let DrawRule::Unjudged(why) = draw_of(&rule) else {
            panic!("{overrides}: replayed");
        };
        assert!(why.contains(reason), "{overrides}: {why}");
    }
    // a turn that ignores its end draws among the other tokens: the draw
    // is replayed without them
    let rule = with(serde_json::json!({ "ignore_eos": true })).without(Ok(&[3]));
    assert!(
        rule.replayed_draw()
            .is_some_and(|r| r.biases == [(3, f64::NEG_INFINITY)])
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

#[test]
fn a_statement_with_the_wrong_types_is_the_nodes_fault() {
    use crate::node_fault::NodeFault;
    // an honest node writes each field with its type; a node that writes
    // another opts out of the token judgement, so it is held to it
    for overrides in [
        serde_json::json!({ "reasoning": 1 }),
        serde_json::json!({ "reasoning_budget": 8, "reasoning": {"start": "<think>"} }),
        serde_json::json!({ "seed": null }),
        serde_json::json!({ "seed": 4_294_967_296u64 }),
        serde_json::json!({ "top_k": 5_000_000_000u64 }),
        serde_json::json!({ "top_k": "40" }),
        serde_json::json!({ "top_p": null }),
        serde_json::json!({ "temperature": "hot" }),
        serde_json::json!({ "draw": 1 }),
        serde_json::json!({ "ignore_eos": "yes" }),
        serde_json::json!({ "logit_bias": [[1]] }),
        serde_json::json!({ "samplers": "top_k" }),
        serde_json::json!({ "samplers": ["top_k", 3] }),
        serde_json::json!({ "repeat_penalty": null }),
        serde_json::json!({ "mirostat": 2 }),
    ] {
        let statement = statement(overrides.clone());
        let error = check_statement(&statement).expect_err(&overrides.to_string());
        assert!(NodeFault::find(&error).is_some(), "{overrides}: {error:#}");
        let rule = TokenRule::from_sampler(&statement);
        assert!(matches!(rule, TokenRule::Unjudged(_)), "{overrides}");
    }
    for text in ["sampled", "sampled [1]", "anything"] {
        assert!(check_statement(text).is_err(), "{text}");
    }
    // a well-formed statement of something the check cannot judge is not
    for overrides in [
        serde_json::json!({}),
        serde_json::json!({ "grammar": "abc", "repeat_penalty": 1.1 }),
        serde_json::json!({ "samplers": ["top_k", "mystery"] }),
        serde_json::json!({ "top_p": 7.0 }),
        serde_json::json!({ "reasoning_budget": 8, "reasoning": {"start": "", "ends": []} }),
    ] {
        assert!(
            check_statement(&statement(overrides.clone())).is_ok(),
            "{overrides}"
        );
    }
    assert!(check_statement(turn::GREEDY).is_ok());
}

#[test]
fn what_leaves_a_turn_unjudged_is_named_as_a_gap() {
    let gaps_of = |overrides: serde_json::Value| {
        let mut all = serde_json::json!({ "draw": turn::INDEXED_DRAW });
        for (k, v) in overrides.as_object().unwrap() {
            all[k] = v.clone();
        }
        let statement = statement(all);
        gaps(&turn::sampler_parameters(&statement).unwrap()).unwrap()
    };
    assert!(gaps_of(serde_json::json!({})).is_empty());
    assert_eq!(
        gaps_of(serde_json::json!({ "grammar": "abc" })),
        [Gap::Grammar]
    );
    assert_eq!(gaps_of(serde_json::json!({ "draw": null })), [Gap::NoDraw]);
    assert_eq!(
        gaps_of(serde_json::json!({ "repeat_penalty": 1.1 })),
        [Gap::Sampler("penalties".into())]
    );
    assert_eq!(
        gaps_of(serde_json::json!({ "dynatemp_range": 0.5 })),
        [Gap::DynamicTemperature]
    );
    assert_eq!(
        gaps_of(serde_json::json!({ "samplers": ["mystery", "temperature"] })),
        [Gap::Unknown("mystery".into())]
    );
    // mirostat picks its token itself: no draw is missing
    assert_eq!(
        gaps_of(serde_json::json!({ "mirostat": {"mode": 2}, "draw": null })),
        [Gap::Mirostat]
    );
    // a turn at a temperature of 0 draws nothing
    assert!(gaps_of(serde_json::json!({ "temperature": 0.0, "draw": null })).is_empty());
}

#[test]
fn an_empty_chain_is_the_runtimes_default_chain() {
    // `sampling_params` of `src/skippy/sampling_config.cpp` runs its default
    // chain when a request names none: the filters hold
    let rule = TokenRule::from_sampler(&statement(
        serde_json::json!({"samplers": [], "top_k": 2, "top_p": 1.0, "min_p": 0.0}),
    ));
    let logits = [3.0, 2.0, 1.0, 0.0];
    assert!(keeps(&rule, &logits, 1));
    assert!(!keeps(&rule, &logits, 2), "{rule:?}");
    // and the default chain leaves a top-p of 1 out
    assert_eq!(
        default_chain(1.0, 1.0),
        [
            "penalties",
            "dry",
            "top_n_sigma",
            "top_k",
            "min_p",
            "xtc",
            "temperature"
        ]
    );
    assert!(default_chain(0.5, 0.9).contains(&"typical_p"));
}

#[test]
fn filters_at_their_bounds_keep_what_llama_keeps() {
    let logits = [3.0, 2.0, 1.0, 0.0];
    let rule = |overrides: serde_json::Value| TokenRule::from_sampler(&statement(overrides));
    // `llama_sampler_top_p_apply` keeps the best token at a top-p of 0
    let zero_p = rule(serde_json::json!({"top_p": 0.0, "top_k": 0, "min_p": 0.0}));
    assert!(keeps(&zero_p, &logits, 0));
    assert!(!keeps(&zero_p, &logits, 1));
    // a top-p of 1 keeps every token
    let all_p = rule(serde_json::json!({"top_p": 1.0, "top_k": 0, "min_p": 0.0}));
    assert!(keeps(&all_p, &logits, 3));
    // min-p 0 keeps every token, min-p 1 the best and its equals
    let no_min = rule(serde_json::json!({"top_p": 1.0, "top_k": 0, "min_p": 0.0}));
    assert!(keeps(&no_min, &logits, 3));
    let full_min = rule(serde_json::json!({"top_p": 1.0, "top_k": 0, "min_p": 1.0}));
    assert!(keeps(&full_min, &logits, 0));
    assert!(!keeps(&full_min, &logits, 1));
    // a top-k below 0 is off, and one above the vocabulary keeps every token
    for k in [-5, 1000] {
        let k_rule = rule(serde_json::json!({"top_p": 1.0, "top_k": k, "min_p": 0.0}));
        assert!(keeps(&k_rule, &logits, 3), "{k}");
    }
}

#[test]
fn a_dynamic_temperature_at_zero_still_samples() {
    // `llama_sampler_temp_ext_apply` samples with a range above 0 at any
    // base temperature; the chain does not keep the best logit alone
    let rule = TokenRule::from_sampler(&statement(serde_json::json!({
        "temperature": 0.0,
        "dynatemp_range": 0.5,
        "logit_bias": [[1, 0.5]],
    })));
    let logits = [3.0, 2.0, 1.0, 0.0];
    assert!(keeps(&rule, &logits, 1), "{rule:?}");
    assert!(matches!(draw_of(&rule), DrawRule::Unjudged(_)));
}

#[test]
fn tokens_the_runtime_takes_out_are_never_kept() {
    let logits = [3.0, 2.9, 1.0];
    // the best token is suppressed: the greedy choice is the next one
    let greedy = TokenRule::Greedy.without(Ok(&[0]));
    assert!(keeps(&greedy, &logits, 1), "{greedy:?}");
    assert!(!keeps(&greedy, &logits, 0));
    assert_eq!(TokenRule::Greedy.without(Ok(&[])), TokenRule::Greedy);
    // a sampled rule loses them too, whatever bias the statement gives them
    let sampled = TokenRule::from_sampler(&statement(
        serde_json::json!({"top_k": 0, "top_p": 1.0, "min_p": 0.0, "logit_bias": [[0, 5.0]]}),
    ))
    .without(Ok(&[0]));
    assert!(!keeps(&sampled, &logits, 0));
    assert!(keeps(&sampled, &logits, 2));
    // a client that cannot read them judges nothing
    let unread = TokenRule::Greedy.without(Err("no vocabulary"));
    assert_eq!(unread, TokenRule::Unjudged("no vocabulary".into()));
}
