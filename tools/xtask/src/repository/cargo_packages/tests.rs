use super::*;

const LEGACY: &str =
    include_str!("../../../../../scripts/tests/fixtures/ci-source-layout/legacy-packages.json");
const EXTRACTED: &str =
    include_str!("../../../../../scripts/tests/fixtures/ci-source-layout/extracted-packages.json");

fn available(json: &str) -> BTreeSet<PackageName> {
    PackageList::parse(json)
        .expect("frozen census")
        .names()
        .iter()
        .cloned()
        .collect()
}

fn resolve(crates: &str, generation: Generation) -> Result<PackageList, Error> {
    TranslationRequest::parse(crates, None, generation)?.resolve(&available(EXTRACTED))
}

#[test]
fn extracted_members_have_one_owner_when_legacy_batches_are_separate() {
    let legacy = PackageList::parse(LEGACY).expect("legacy census");
    let workspace = available(EXTRACTED);
    let mut translated = Vec::new();
    for name in legacy.names() {
        let json = format!("[\"{}\"]", name.as_str());
        let request = TranslationRequest::parse(&json, None, Generation::Legacy).expect("batch");
        translated.extend(
            request
                .resolve(&workspace)
                .expect("complete owners")
                .names()
                .to_vec(),
        );
    }
    assert_eq!(translated.len(), workspace.len());
    assert_eq!(translated.into_iter().collect::<BTreeSet<_>>(), workspace);
}

#[test]
fn batches_are_preserved_when_source_and_plan_generation_match() {
    for (json, generation) in [
        (LEGACY, Generation::Legacy),
        (EXTRACTED, Generation::Current),
    ] {
        let input = PackageList::parse(json).expect("census");
        let request = TranslationRequest::parse(json, None, generation).expect("plan");
        let output = request.resolve(&available(json)).expect("unchanged");
        assert_eq!(output.names(), input.names());
    }
}

#[test]
fn reused_names_resolve_by_declared_generation_when_plan_is_partial() {
    for (crates, generation, expected) in [
        (
            "[\"model-package\"]",
            Generation::Legacy,
            "[\"skippy-model-package\"]\n",
        ),
        (
            "[\"skippy-model-package\"]",
            Generation::Legacy,
            "[\"skippy-package-builder\"]\n",
        ),
        (
            "[\"skippy-model-package\"]",
            Generation::Current,
            "[\"skippy-model-package\"]\n",
        ),
    ] {
        let output = resolve(crates, generation).expect("explicit generation");
        assert_eq!(output.python_json_line(), expected);
    }
}

#[test]
fn mapped_missing_owner_fails_when_extraction_is_incomplete() {
    let mut workspace = available(EXTRACTED);
    workspace.remove(&PackageName::try_from("mesh-llm-membership".to_owned()).expect("name"));
    let request =
        TranslationRequest::parse("[\"mesh-llm-host-runtime\"]", None, Generation::Legacy)
            .expect("request");
    let error = request
        .resolve(&workspace)
        .expect_err("must not skip missing owner");
    assert!(matches!(error, Error::MissingOwners { .. }));
}

#[test]
fn unmapped_absent_names_pass_through_when_membership_filter_runs_later() {
    let output =
        resolve("[\"unknown\", \"mesh-llm-analytics\"]", Generation::Legacy).expect("unmapped");
    assert_eq!(
        output.python_json_line(),
        "[\"unknown\", \"mesh-llm-analytics\"]\n"
    );
}

#[test]
fn migration_stays_inactive_when_builder_is_absent() {
    let workspace = BTreeSet::new();
    let request =
        TranslationRequest::parse("[\"model-hf\"]", None, Generation::Legacy).expect("request");
    let output = request.resolve(&workspace).expect("legacy source");
    assert_eq!(output.python_json_line(), "[\"model-hf\"]\n");
}

#[test]
fn translation_keeps_request_and_successor_order_when_owner_expands() {
    let output = resolve(
        "[\"unknown\", \"model-hf\", \"skippy-server\"]",
        Generation::Legacy,
    )
    .expect("owners");
    assert_eq!(
        output.python_json_line(),
        "[\"unknown\", \"skippy-model-hf\", \"skippy-hf-hub\", \"skippy-serving\", \"skippy-api\", \"skippy-cli\", \"skippy-commands\", \"skippy-config\", \"skippy-events\"]\n"
    );
}

#[test]
fn duplicate_translation_fails_when_distinct_inputs_share_an_owner() {
    let error =
        resolve("[\"model-hf\", \"skippy-model-hf\"]", Generation::Legacy).expect_err("overlap");
    assert!(matches!(error, Error::DuplicateTranslation(_)));
}

#[test]
fn request_fails_when_name_is_outside_complete_plan() {
    let error = TranslationRequest::parse(
        "[\"skippy-cli\"]",
        Some("[{\"crates\":[\"mesh-llm\"]}]"),
        Generation::Current,
    )
    .err()
    .expect("outside plan");
    assert!(matches!(error, Error::OutsidePlan));
}

#[test]
fn complete_batches_flatten_in_order_when_request_is_a_subset() {
    let request = TranslationRequest::parse(
        "[\"model-hf\"]",
        Some("[{\"id\":\"one\",\"crates\":[\"unknown\"]},{\"crates\":[\"model-hf\"]}]"),
        Generation::Legacy,
    )
    .expect("valid subset");
    let output = request
        .resolve(&available(EXTRACTED))
        .expect("translated subset");
    assert_eq!(
        output.python_json_line(),
        "[\"skippy-model-hf\", \"skippy-hf-hub\"]\n"
    );
}

#[test]
fn package_boundary_rejects_invalid_names_shapes_and_duplicates() {
    for json in [
        "[]",
        "null",
        "{}",
        "[1]",
        "[true]",
        "[\"\"]",
        "[\"a\",\"a\"]",
        "[\"--workspace\"]",
        "[\"$(touch /tmp/unwanted)\"]",
        "[\"x\\ny\"]",
        "[\"a.b\"]",
        "[\"é\"]",
    ] {
        assert!(PackageList::parse(json).is_err(), "accepted {json}");
    }
}

#[test]
fn package_boundary_accepts_ascii_grammar_when_names_are_ordered() {
    let names = PackageList::parse("[\"9abc\",\"A_B-c\"]").expect("grammar");
    assert_eq!(names.python_json_line(), "[\"9abc\", \"A_B-c\"]\n");
}

#[test]
fn complete_matrix_rejects_bad_rows_and_cross_batch_duplicates() {
    for batches in [
        "[]",
        "{}",
        "null",
        "[{}]",
        "[{\"crates\":[]}]",
        "[{\"crates\":[\"a\"]},{\"crates\":[\"a\"]}]",
        "[{\"crates\":[\"a\"]},{\"crates\":[\"--workspace\"]}]",
    ] {
        assert!(
            TranslationRequest::parse("[\"a\"]", Some(batches), Generation::Current).is_err(),
            "accepted {batches}"
        );
    }
}

#[test]
fn generation_fails_when_selector_is_not_explicitly_supported() {
    assert!(Generation::parse("auto").is_err());
}
