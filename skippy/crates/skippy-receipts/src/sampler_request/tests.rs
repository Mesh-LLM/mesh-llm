use super::*;

/// A statement of the node's default chain, with `overrides`.
fn statement(overrides: serde_json::Value) -> String {
    let mut p = serde_json::json!({
        "seed": 0,
        "draw": "indexed-v1",
        "samplers": ["penalties", "dry", "top_n_sigma", "top_k", "typical_p", "top_p", "min_p", "xtc", "temperature"],
        "temperature": 0.800000011920929,
        "dynatemp_range": 0.0,
        "top_k": 40,
        "top_p": 0.949999988079071,
        "min_p": 0.05000000074505806,
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
        "reasoning_budget": null,
    });
    for (k, v) in overrides.as_object().unwrap() {
        p[k] = v.clone();
    }
    turn::sampled(&p)
}

/// A request that set these parameters and sent no constraint.
fn plain(temperature: f64, top_p: f64, top_k: i64, min_p: f64) -> RequestedSampling {
    RequestedSampling {
        temperature: Some(temperature),
        top_p: Some(top_p),
        top_k: Some(top_k),
        min_p: Some(min_p),
        grammar: Some(false),
        penalties: Some(false),
        logit_bias: Some(false),
        sampler_chain: Some(false),
        ..RequestedSampling::default()
    }
}

#[test]
fn a_statement_of_what_the_request_set_is_consistent() {
    // the node holds the values as f32
    let request = plain(0.8, 0.95, 40, 0.05);
    assert_eq!(
        request.contradiction(&statement(serde_json::json!({}))),
        None
    );
    let seeded = RequestedSampling {
        seed: Some(7),
        ..request
    };
    assert_eq!(
        seeded.contradiction(&statement(serde_json::json!({"seed": 7}))),
        None
    );
    // a request that set nothing is contradicted by nothing it could know
    let unknown = RequestedSampling::default();
    for overrides in [
        serde_json::json!({}),
        serde_json::json!({"temperature": 1.5, "grammar": "abc", "logit_bias": [[1, 9.0]]}),
    ] {
        assert_eq!(unknown.contradiction(&statement(overrides)), None);
    }
    assert_eq!(unknown.contradiction(turn::GREEDY), None);
}

#[test]
fn a_parameter_the_request_set_must_be_stated_with_its_value() {
    let request = RequestedSampling {
        seed: Some(7),
        ..plain(0.8, 0.95, 40, 0.05)
    };
    for (overrides, named) in [
        (
            serde_json::json!({"seed": 7, "temperature": 1.5}),
            "temperature",
        ),
        (serde_json::json!({"seed": 7, "top_p": 1.0}), "top_p"),
        (serde_json::json!({"seed": 7, "top_k": 0}), "top_k"),
        (serde_json::json!({"seed": 7, "min_p": 0.0}), "min_p"),
        (serde_json::json!({"seed": 8}), "seed"),
        (serde_json::json!({"seed": 7, "top_k": null}), "top_k"),
    ] {
        let found = request.contradiction(&statement(overrides.clone()));
        assert!(
            found.as_deref().is_some_and(|f| f.contains(named)),
            "{overrides}: {found:?}"
        );
    }
}

#[test]
fn a_constraint_the_request_did_not_send_must_not_be_claimed() {
    // each of these leaves the tokens unjudged, or moves any token into
    // the candidates
    let request = plain(0.8, 0.95, 40, 0.05);
    for overrides in [
        serde_json::json!({"grammar": "9f2c"}),
        serde_json::json!({"logit_bias": [[5, 100.0]]}),
        serde_json::json!({"repeat_penalty": 1.1}),
        serde_json::json!({"presence_penalty": 0.5}),
        serde_json::json!({"dry": {"multiplier": 0.8}}),
        serde_json::json!({"xtc": {"probability": 0.5, "threshold": 0.1}}),
        serde_json::json!({"typical_p": 0.5}),
        serde_json::json!({"top_nsigma": 2.0}),
        serde_json::json!({"mirostat": {"mode": 2}}),
        serde_json::json!({"dynatemp_range": 0.5}),
        serde_json::json!({"samplers": ["top_k", "top_p", "min_p", "mystery", "temperature"]}),
    ] {
        let found = request.contradiction(&statement(overrides.clone()));
        assert!(found.is_some(), "{overrides} was accepted");
    }
    // a request that did send them is not contradicted
    let constrained = RequestedSampling {
        grammar: Some(true),
        penalties: Some(true),
        logit_bias: Some(true),
        ..request
    };
    let overrides = serde_json::json!({
        "grammar": "9f2c", "logit_bias": [[5, 1.0]], "repeat_penalty": 1.1,
    });
    assert_eq!(constrained.contradiction(&statement(overrides)), None);
}

#[test]
fn a_chain_that_leaves_out_a_filter_the_request_set_is_a_contradiction() {
    // without its filters, a chain keeps every token
    let request = plain(0.8, 0.95, 40, 0.05);
    for chain in [
        serde_json::json!(["temperature"]),
        serde_json::json!(["top_k", "top_p", "temperature"]),
        serde_json::json!(["top_k", "top_p", "min_p"]),
    ] {
        let found = request.contradiction(&statement(serde_json::json!({"samplers": chain})));
        assert!(
            found
                .as_deref()
                .is_some_and(|f| f.contains("leaves it out")),
            "{chain}: {found:?}"
        );
    }
    // an empty chain is the runtime's default chain, which holds them
    assert_eq!(
        request.contradiction(&statement(serde_json::json!({"samplers": []}))),
        None
    );
    // a request that named its own chain decides what is in it
    let own = RequestedSampling {
        sampler_chain: Some(true),
        ..request
    };
    let chain = serde_json::json!({"samplers": ["top_k", "temperature"]});
    assert_eq!(own.contradiction(&statement(chain)), None);
}

#[test]
fn a_greedy_statement_fits_a_request_that_asked_for_the_best_token() {
    assert_eq!(
        RequestedSampling::greedy().contradiction(turn::GREEDY),
        None
    );
    // a request that turned every filter off at a temperature of 1 is
    // decoded greedily too
    assert_eq!(plain(1.0, 1.0, 0, 0.0).contradiction(turn::GREEDY), None);
    let found = plain(0.8, 0.95, 40, 0.05).contradiction(turn::GREEDY);
    assert!(found.is_some_and(|f| f.contains("greedily")));
    assert!(
        plain(1.0, 0.95, 0, 0.0)
            .contradiction(turn::GREEDY)
            .is_some()
    );
}

#[test]
fn the_audit_request_accepts_only_an_unconstrained_best_token() {
    let asked = RequestedSampling::greedy();
    // a node whose model thinks within a budget says so; the audit asked
    // for none
    let budget = statement(serde_json::json!({"temperature": 0.0, "reasoning_budget": 512}));
    assert!(asked.contradiction(&budget).is_some());
    for overrides in [
        serde_json::json!({"temperature": 0.7}),
        serde_json::json!({"temperature": 0.0, "repeat_penalty": 1.1}),
        serde_json::json!({"temperature": 0.0, "grammar": "9f2c"}),
        serde_json::json!({"temperature": 0.0, "logit_bias": [[1, 5.0]]}),
        serde_json::json!({"temperature": 0.0, "samplers": ["top_k"]}),
    ] {
        let found = asked.contradiction(&statement(overrides.clone()));
        assert!(found.is_some(), "{overrides} was accepted");
    }
    assert_eq!(
        asked.contradiction(&statement(serde_json::json!({"temperature": 0.0}))),
        None
    );
}

#[test]
fn a_default_reasoning_budget_does_not_contradict_a_request_that_named_none() {
    // The known honest case: a thinking model gets a default budget from
    // the server, so a request at temperature 0 without `reasoning_budget`
    // is not greedy, and its statement names the budget.
    let request = RequestedSampling {
        temperature: Some(0.0),
        grammar: Some(false),
        penalties: Some(false),
        logit_bias: Some(false),
        sampler_chain: Some(false),
        ..RequestedSampling::default()
    };
    let honest = statement(serde_json::json!({"temperature": 0.0, "reasoning_budget": 2048}));
    assert_eq!(request.contradiction(&honest), None);
    // a request that named a budget is held to it
    let limited = RequestedSampling {
        reasoning_budget: Some(1024),
        ..request.clone()
    };
    assert!(limited.contradiction(&honest).is_some());
    let within = statement(serde_json::json!({"temperature": 0.0, "reasoning_budget": 512}));
    assert_eq!(limited.contradiction(&within), None);
    let unlimited = statement(serde_json::json!({"temperature": 0.0}));
    assert!(limited.contradiction(&unlimited).is_some());
}

#[test]
fn a_statement_that_names_nothing_is_a_contradiction() {
    let request = RequestedSampling::default();
    for text in ["sampled", "sampled {", "whatever the node likes", ""] {
        assert!(request.contradiction(text).is_some(), "{text:?}");
    }
}

#[test]
fn a_node_may_not_opt_out_of_the_judgement_unasked() {
    let request = RequestedSampling {
        ignore_eos: Some(false),
        ..plain(0.8, 0.95, 40, 0.05)
    };
    for (overrides, named) in [
        // no draw a client can replay, on a turn that draws
        (serde_json::json!({"draw": null}), "no draw"),
        (serde_json::json!({"draw": "another-rule"}), "no draw"),
        // end tokens the request did not ask to ignore
        (serde_json::json!({"ignore_eos": true}), "ignored its end"),
        // a reasoning sampler with no budget that could force a token
        (
            serde_json::json!({"reasoning": {"start": "<think>", "ends": ["</think>"]}}),
            "no reasoning budget",
        ),
        // a statement malformed in itself
        (serde_json::json!({"reasoning": 1}), "malformed"),
        (serde_json::json!({"top_k": 5_000_000_000u64}), "malformed"),
        // a value no request can set
        (serde_json::json!({"temperature": 150.0}), "out of range"),
        (serde_json::json!({"top_p": -0.5}), "out of range"),
    ] {
        let found = request.contradiction(&statement(overrides.clone()));
        assert!(
            found.as_deref().is_some_and(|f| f.contains(named)),
            "{overrides}: {found:?}"
        );
    }
    // a request that asked for them is not contradicted
    let asked = RequestedSampling {
        ignore_eos: Some(true),
        grammar: Some(true),
        ..request
    };
    let overrides = serde_json::json!({
        "ignore_eos": true, "grammar": "9f2c", "reasoning_budget": 8,
        "reasoning": {"start": "<think>", "ends": ["</think>"]},
    });
    assert_eq!(asked.contradiction(&statement(overrides)), None);
}

// The runtime builds the reasoning sampler for every turn with a finite
// budget, with a grammar or without. A thinking model's node sets a
// default budget, so a request that named none can get one.
#[test]
fn a_reasoning_sampler_needs_a_budget_and_no_grammar() {
    let sampler = serde_json::json!({"start": "<think>", "ends": ["</think>"]});
    let stated = |budget: serde_json::Value| {
        statement(serde_json::json!({"reasoning_budget": budget, "reasoning": sampler}))
    };
    // the request named no budget: the node's default is consistent
    let unset = plain(0.8, 0.95, 40, 0.05);
    assert_eq!(unset.reasoning_budget, None);
    assert_eq!(unset.contradiction(&stated(8.into())), None);
    assert_eq!(unset.contradiction(&stated(0.into())), None);
    // the request named one: the node may not allow more
    let asked = RequestedSampling {
        reasoning_budget: Some(8),
        ..plain(0.8, 0.95, 40, 0.05)
    };
    assert_eq!(asked.contradiction(&stated(8.into())), None);
    assert!(asked.contradiction(&stated(64.into())).is_some());
    // the request allowed any number: no sampler can force a token
    let unlimited = RequestedSampling {
        reasoning_budget: Some(-1),
        ..plain(0.8, 0.95, 40, 0.05)
    };
    assert!(unlimited.contradiction(&stated(8.into())).is_some());
    // a sampler with no budget, or a negative one, forces nothing
    for budget in [serde_json::Value::Null, (-1).into()] {
        let found = unset.contradiction(&stated(budget.clone()));
        assert!(
            found
                .as_deref()
                .is_some_and(|f| f.contains("no reasoning budget")),
            "{budget}: {found:?}"
        );
    }
}

#[test]
fn numbers_are_compared_exactly() {
    // a seed near 2^31 and its neighbour are two seeds
    let request = RequestedSampling {
        seed: Some(2_147_483_648),
        ..plain(0.8, 0.95, 40, 0.05)
    };
    let found = request.contradiction(&statement(serde_json::json!({"seed": 2_147_483_649u64})));
    assert!(found.is_some_and(|f| f.contains("seed")));
    assert_eq!(
        request.contradiction(&statement(serde_json::json!({"seed": 2_147_483_648u64}))),
        None
    );
    // the node holds a temperature as an f32: the same f32 is the same
    // value, the next one is not
    let warm = plain(0.7, 0.95, 40, 0.05);
    let as_f32 = f64::from(0.7f32);
    assert_eq!(
        warm.contradiction(&statement(serde_json::json!({"temperature": as_f32}))),
        None
    );
    let next = f64::from(f32::from_bits(0.7f32.to_bits() + 1));
    assert!(
        warm.contradiction(&statement(serde_json::json!({"temperature": next})))
            .is_some()
    );
}

#[test]
fn a_filter_left_at_a_default_that_is_on_is_not_stated_off() {
    let request = RequestedSampling {
        temperature: Some(0.8),
        grammar: Some(false),
        penalties: Some(false),
        logit_bias: Some(false),
        sampler_chain: Some(false),
        unset_filters_on: Some(true),
        ..RequestedSampling::default()
    };
    assert_eq!(
        request.contradiction(&statement(serde_json::json!({}))),
        None
    );
    for overrides in [
        serde_json::json!({"top_k": 0}),
        serde_json::json!({"top_p": 1.0}),
        serde_json::json!({"min_p": 0.0}),
        serde_json::json!({"samplers": ["top_k", "temperature"]}),
    ] {
        let found = request.contradiction(&statement(overrides.clone()));
        assert!(found.is_some(), "{overrides}");
    }
    // a caller that does not know the defaults holds the node to nothing
    let unknown = RequestedSampling {
        unset_filters_on: None,
        ..request
    };
    assert_eq!(
        unknown.contradiction(&statement(serde_json::json!({"top_k": 0}))),
        None
    );
}

#[test]
fn a_request_with_sampling_off_may_be_stated_at_a_temperature_of_zero() {
    // Skippy turns sampling off for a request at temperature 1 with every
    // filter off, and the runtime then keeps the best logit
    let off = plain(1.0, 1.0, 0, 0.0);
    let budgeted = statement(serde_json::json!({
        "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
        "draw": null, "reasoning_budget": 16,
    }));
    assert_eq!(off.contradiction(&budgeted), None);
    assert!(plain(0.9, 1.0, 0, 0.0).contradiction(&budgeted).is_some());
}
