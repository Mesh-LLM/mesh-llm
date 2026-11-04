use super::*;

/// A statement of the node's default chain, with `overrides`.
fn statement(overrides: serde_json::Value) -> String {
    let mut p = serde_json::json!({
        "seed": 0,
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
        serde_json::json!([]),
    ] {
        let found = request.contradiction(&statement(serde_json::json!({"samplers": chain})));
        assert!(
            found
                .as_deref()
                .is_some_and(|f| f.contains("leaves it out")),
            "{chain}: {found:?}"
        );
    }
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
