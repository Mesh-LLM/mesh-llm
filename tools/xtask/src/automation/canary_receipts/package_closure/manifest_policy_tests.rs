use super::*;
#[test]
fn only_positive_model_size_may_change_while_roster_policy_and_artifacts_stay_fixed() {
    let before = json!({"policy":{"fixed":true},"models":[{"family":"fixture","resources":{"estimated_model_bytes":1},"artifact":{"repo":"trusted"}}]});
    let mut after = before.clone();
    after["models"][0]["resources"]["estimated_model_bytes"] = json!(2);
    assert!(family(&before, &after).is_ok());
    for value in [json!(0), json!(-1), json!(true), json!("2")] {
        after["models"][0]["resources"]["estimated_model_bytes"] = value;
        assert!(family(&before, &after).is_err());
    }
    after = before.clone();
    after["models"][0]["artifact"]["repo"] = json!("agent");
    assert!(family(&before, &after).is_err());
    after = before.clone();
    after["policy"]["fixed"] = json!(false);
    assert!(family(&before, &after).is_err());
}
#[test]
fn new_source_classifications_are_exact_append_only_and_cannot_select_artifacts() {
    let before = json!({"policy":"fixed","candidates":[{"llama_model":"old","family":"variant-a"},{"llama_model":"old","family":"variant-b"}]});
    let mut after = before.clone();
    after["candidates"].as_array_mut().unwrap().push(json!({"llama_model":"new","family":"new-family","status":"candidate","notes":"paired boundary registered"}));
    let sources = BTreeSet::from(["old".into(), "new".into()]);
    let boundaries = BTreeSet::from(["new".into()]);
    assert!(parity(&before, &after, &sources, &boundaries).is_ok());
    assert!(parity(&before, &before, &sources, &boundaries).is_err());
    assert!(parity(&before, &after, &sources, &BTreeSet::new()).is_err());
    after["candidates"][2]["repo"] = json!("agent/repo");
    assert!(parity(&before, &after, &sources, &boundaries).is_err());
    after["candidates"][2]
        .as_object_mut()
        .unwrap()
        .remove("repo");
    after["candidates"][2]["status"] = json!("needs_candidate");
    assert!(parity(&before, &after, &sources, &boundaries).is_err());
    after["candidates"][2]["status"] = json!("candidate");
    after["candidates"][0]["family"] = json!("reordered");
    assert!(parity(&before, &after, &sources, &boundaries).is_err());
}
#[test]
fn actual_native_files_require_real_paired_calls_outside_comments_strings_and_raw_literals() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("src/models");
    fs::create_dir_all(&models).unwrap();
    fs::write(
        models.join("paired.cpp"),
        b"begin_block (layer); end_block(layer);\n",
    )
    .unwrap();
    fs::write(
        models.join("one.cpp"),
        b"begin_block(layer); // end_block(layer)\n",
    )
    .unwrap();
    fs::write(models.join("fake.cpp"),b"// begin_block(x); end_block(x)\nconst char *a=\"begin_block(x); end_block(x)\"; auto b=u8R\"delim(begin_block(x); end_block(x))delim\"; /* begin_block(x); end_block(x) */\n").unwrap();
    let (sources, boundaries) = model_boundaries::inventory(directory.path()).unwrap();
    assert_eq!(
        sources,
        BTreeSet::from(["paired".into(), "one".into(), "fake".into()])
    );
    assert_eq!(boundaries, BTreeSet::from(["paired".into()]));
}
