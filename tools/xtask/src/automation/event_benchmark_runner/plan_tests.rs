use super::*;
fn spec() -> Spec {
    Spec {
        seed: 42,
        pairs_primary: 20,
        pairs_scenario: 10,
        scenarios: vec!["smoke".into(), "pressure".into()],
    }
}
fn defaults() -> [Side; 2] {
    sides(
        "candidate".into(),
        None,
        &[Mode::Production, Mode::EventDisabled],
    )
    .unwrap()
}

#[test]
fn modes_are_closed_and_the_trial_selector_preserves_every_wire_value() {
    for (mode, label) in [
        (Mode::Production, "production"),
        (Mode::EventDisabled, "event-disabled"),
        (Mode::Off, "off"),
    ] {
        assert_eq!(mode.label(), label);
        assert_eq!(
            serde_json::from_value::<Mode>(serde_json::json!(label)).unwrap(),
            mode
        );
    }
    assert!(serde_json::from_value::<Mode>(serde_json::json!("unknown")).is_err());
}
#[test]
fn comparison_a_accepts_exactly_two_distinct_modes_on_one_binary() {
    for a in [Mode::Production, Mode::EventDisabled, Mode::Off] {
        for b in [Mode::Production, Mode::EventDisabled, Mode::Off] {
            let result = sides("candidate".into(), None, &[a, b]);
            assert_eq!(result.is_ok(), a != b);
            if let Ok(result) = result {
                assert_eq!(result[0].binary, result[1].binary);
                assert_eq!(result[0].side_id, a.label());
                assert_eq!(result[1].side_id, b.label());
            }
        }
    }
    for modes in [
        vec![],
        vec![Mode::Production],
        vec![Mode::Production, Mode::EventDisabled, Mode::Off],
    ] {
        assert!(sides("candidate".into(), None, &modes).is_err());
    }
}
#[test]
fn comparison_b_varies_binary_and_preserves_exactly_one_trial_mode() {
    let result = sides("candidate".into(), Some("baseline".into()), &[Mode::Off]).unwrap();
    assert_eq!(result[0].mode, result[1].mode);
    assert_ne!(result[0].binary, result[1].binary);
    assert_eq!(
        [result[0].side_id.as_str(), result[1].side_id.as_str()],
        ["current", "baseline"]
    );
    for modes in [vec![], vec![Mode::Production, Mode::EventDisabled]] {
        assert!(sides("candidate".into(), Some("baseline".into()), &modes).is_err());
    }
}
#[test]
fn same_seed_plan_and_prompts_are_reproducible_and_groups_preserve_declared_order() {
    let specification = spec();
    let first = build(&specification, &defaults()).unwrap();
    assert_eq!(first, build(&specification, &defaults()).unwrap());
    assert_eq!(first.len(), 40);
    assert!(first[..20].iter().all(|row| row.scenario == PRIMARY));
    assert!(first[20..30].iter().all(|row| row.scenario == "smoke"));
    assert!(first[30..].iter().all(|row| row.scenario == "pressure"));
    assert_eq!(first[20].pair_index, 0);
    assert!(
        first[0]
            .prompt()
            .ends_with(&format!("{:016x}", first[0].prompt_seed))
    );
}
#[test]
fn prompt_identity_is_independent_of_comparison_side_labels_and_order() {
    let a = build(&spec(), &defaults()).unwrap();
    let b = build(
        &spec(),
        &sides(
            "candidate".into(),
            Some("baseline".into()),
            &[Mode::Production],
        )
        .unwrap(),
    )
    .unwrap();
    for (a, b) in a.iter().zip(&b) {
        assert_eq!(a.prompt_seed, b.prompt_seed);
        assert_eq!(a.prompt(), b.prompt());
        assert!(matches!(
            b.side_order_first.as_str(),
            "current" | "baseline"
        ));
    }
    let mut changed = spec();
    changed.seed += 1;
    assert_ne!(
        a.iter().map(|entry| entry.prompt_seed).collect::<Vec<_>>(),
        build(&changed, &defaults())
            .unwrap()
            .iter()
            .map(|entry| entry.prompt_seed)
            .collect::<Vec<_>>()
    );
}
#[test]
fn seeded_launch_order_uses_both_sides_without_fabricating_observed_order() {
    let entries = build(&spec(), &defaults()).unwrap();
    let ids = entries
        .iter()
        .map(|entry| entry.side_order_first.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(ids, BTreeSet::from(["production", "event-disabled"]));
}
#[test]
fn invalid_counts_duplicate_or_reserved_scenarios_refuse_before_planning() {
    for (primary, scenario, names) in [
        (0, 10, vec!["smoke"]),
        (20, 0, vec!["smoke"]),
        (20, 10, vec![]),
        (20, 10, vec!["smoke", "smoke"]),
        (20, 10, vec![PRIMARY]),
        (10000, 1, vec!["smoke"]),
        (usize::MAX, usize::MAX, vec!["smoke"]),
    ] {
        let spec = Spec {
            seed: 0,
            pairs_primary: primary,
            pairs_scenario: scenario,
            scenarios: names.into_iter().map(String::from).collect(),
        };
        assert!(build(&spec, &defaults()).is_err());
    }
}
#[test]
fn typed_seed_accepts_u64_boundaries_and_refuses_signed_or_overflowing_input() {
    for seed in [0, u64::MAX] {
        let mut input = serde_json::to_value(spec()).unwrap();
        input["seed"] = serde_json::json!(seed);
        assert!(serde_json::from_value::<Spec>(input).is_ok());
    }
    let text = serde_json::to_string(&spec()).unwrap();
    for seed in ["-1", "18446744073709551616"] {
        assert!(
            serde_json::from_str::<Spec>(&text.replace("\"seed\":42", &format!("\"seed\":{seed}")))
                .is_err()
        );
    }
}

#[test]
fn scenario_labels_remain_metadata_and_log_paths_never_use_their_raw_bytes() {
    let mut spec = spec();
    spec.scenarios = vec!["../escape / 雪".into()];
    let entries = build(&spec, &defaults()).unwrap();
    let entry = entries.last().unwrap();
    assert_eq!(entry.scenario, "../escape / 雪");
    let stem = entry.log_stem("production");
    assert!(!stem.contains("/"));
    assert!(!stem.contains(".."));
    assert_eq!(entry.prompt_sha256().len(), 64);
}
