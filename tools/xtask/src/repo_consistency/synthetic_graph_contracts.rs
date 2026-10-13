//! Sparse native graph fixtures must cover the current certification registry.
//! These checks qualify source contracts, not actual CTest/native execution.
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::Path};

const PATCH: &str =
    "skippy/llama_cpp/patches/0024-test-skippy-cover-the-complete-canary-graph-registry.patch";
const REGISTRY: &str = "ci/llama-canary/family-certified.json";

#[derive(Clone, Deserialize)]
struct Model {
    family: String,
    architecture: String,
    class: String,
    execution: Dimensions,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct Dimensions {
    trunk_layers: u64,
    activation_width: u64,
    mtp_layers: u64,
}
#[derive(Clone, Debug)]
struct Fixture {
    family: String,
    architecture: String,
    dimensions: Dimensions,
    mode: String,
}
#[derive(Debug, Eq, PartialEq)]
enum Failure {
    Missing(String),
    Unregistered(String),
    Duplicate(String),
    Dimensions(String),
    Mode(String),
}
fn source(path: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::read_to_string(root.join(path)).expect("owned native graph contract source")
}
fn models() -> Vec<Model> {
    #[derive(Deserialize)]
    struct Registry {
        models: Vec<Model>,
    }
    serde_json::from_str::<Registry>(&source(REGISTRY))
        .expect("typed family registry")
        .models
}
fn fixtures(text: &str) -> Result<Vec<Fixture>, String> {
    text.lines()
        .filter_map(|line| line.strip_prefix("+skippy_contract_case("))
        .map(|line| {
            let row = line.strip_suffix(')').ok_or("unterminated graph fixture")?;
            let fields: Vec<_> = row.split_whitespace().collect();
            let [family, architecture, layers, width, mtp, mode] = fields.as_slice() else {
                return Err(
                    "graph fixture must declare family, architecture, layers, width, MTP and mode"
                        .to_owned(),
                );
            };
            let number = |value: &str| {
                value
                    .parse::<u64>()
                    .map_err(|_| "graph fixture dimension must be an unsigned integer".to_owned())
            };
            Ok(Fixture {
                family: (*family).to_owned(),
                architecture: (*architecture).to_owned(),
                dimensions: Dimensions {
                    trunk_layers: number(layers)?,
                    activation_width: number(width)?,
                    mtp_layers: number(mtp)?,
                },
                mode: (*mode).to_owned(),
            })
        })
        .collect()
}
fn expected_mode(model: &Model) -> &'static str {
    match model.architecture.as_str() {
        "gemma3n" => "shared-kv-18",
        "gemma4" => "shared-kv-22",
        "graniteswitch" => "causal-tokens",
        _ => match model.class.as_str() {
            "embedding" | "rerank" => "stateless",
            "encoder_decoder" => "encoder-decoder-rejection",
            _ => "causal",
        },
    }
}
fn coverage(models: &[Model], fixtures: &[Fixture]) -> Vec<Failure> {
    let expected: BTreeMap<_, _> = models
        .iter()
        .map(|model| (model.family.as_str(), model))
        .collect();
    let mut actual = BTreeMap::new();
    let mut failures = Vec::new();
    for fixture in fixtures {
        if actual.insert(fixture.family.as_str(), fixture).is_some() {
            failures.push(Failure::Duplicate(fixture.family.clone()));
        }
    }
    for (family, model) in &expected {
        let Some(fixture) = actual.get(family) else {
            failures.push(Failure::Missing((*family).to_owned()));
            continue;
        };
        if fixture.architecture != model.architecture || fixture.dimensions != model.execution {
            failures.push(Failure::Dimensions((*family).to_owned()));
        }
        if fixture.mode != expected_mode(model) {
            failures.push(Failure::Mode((*family).to_owned()));
        }
    }
    for family in actual
        .keys()
        .filter(|family| !expected.contains_key(**family))
    {
        failures.push(Failure::Unregistered((*family).to_owned()));
    }
    failures
}
fn current() -> (Vec<Model>, Vec<Fixture>) {
    (
        models(),
        fixtures(&source(PATCH)).expect("complete typed patch fixtures"),
    )
}

#[test]
fn synthetic_graph_every_registered_family_has_a_matching_fixture() {
    let (models, fixtures) = current();
    assert!(!models.is_empty());
    assert_eq!(coverage(&models, &fixtures), []);
}
#[test]
fn synthetic_graph_new_family_requires_a_fixture() {
    let (mut models, fixtures) = current();
    let mut added = models[0].clone();
    added.family = "new-canary-family".to_owned();
    models.push(added);
    assert!(
        coverage(&models, &fixtures).contains(&Failure::Missing("new-canary-family".to_owned()))
    );
}
#[test]
fn synthetic_graph_removed_fixture_fails_coverage() {
    let (models, mut fixtures) = current();
    let removed = fixtures.remove(0);
    assert!(coverage(&models, &fixtures).contains(&Failure::Missing(removed.family)));
}
#[test]
fn synthetic_graph_duplicate_fixture_fails_coverage() {
    let (models, mut fixtures) = current();
    let duplicate = fixtures[0].clone();
    let family = duplicate.family.clone();
    fixtures.push(duplicate);
    assert!(coverage(&models, &fixtures).contains(&Failure::Duplicate(family)));
}
#[test]
fn synthetic_graph_dimension_and_mtp_drift_fail_coverage() {
    let (models, fixtures) = current();
    for field in ["trunk_layers", "activation_width", "mtp_layers"] {
        let mut changed = models.clone();
        match field {
            "trunk_layers" => changed[0].execution.trunk_layers += 1,
            "activation_width" => changed[0].execution.activation_width += 1,
            _ => changed[0].execution.mtp_layers += 1,
        }
        assert!(
            coverage(&changed, &fixtures).contains(&Failure::Dimensions(changed[0].family.clone())),
            "{field}"
        );
    }
}
#[test]
fn synthetic_graph_cases_remain_registered_with_required_ctest_fixtures() {
    let patch = source(PATCH);
    for term in [
        "+    include(skippy/tests/graph_contract_cases.cmake)",
        "+    add_test(NAME skippy_graph_contract_${family} COMMAND skippy-graph-contract-models",
        "FIXTURES_REQUIRED contract-${family}",
    ] {
        assert!(patch.contains(term), "missing graph CTest contract {term}");
    }
    for forbidden in ["DISABLED TRUE", "WILL_FAIL"] {
        assert!(
            !patch.contains(forbidden),
            "graph contracts must execute successfully: {forbidden}"
        );
    }
}
#[test]
fn synthetic_graph_fixture_generator_forwards_declared_mtp_depth() {
    let patch = source(PATCH);
    for term in [
        "strcmp(argv[i], \"--contract-mtp\") == 0",
        "contract_mtp_layers = std::stoul(argv[++i])",
        "contract_width, contract_mtp_layers)",
    ] {
        assert!(patch.contains(term), "missing MTP graph contract {term}");
    }
}
#[test]
fn synthetic_graph_stateless_contract_rejects_mutable_state() {
    assert!(source(PATCH).contains("mode == \"stateless\" && !all_states.empty()"));
}
#[test]
fn synthetic_graph_malformed_unregistered_or_wrong_mode_fixture_fails() {
    for row in [
        "+skippy_contract_case(family arch 1 2 0)",
        "+skippy_contract_case(family arch 1 invalid 0 causal)",
        "+skippy_contract_case(family arch 1 2 0 causal",
    ] {
        assert!(fixtures(row).is_err(), "malformed fixture accepted: {row}");
    }
    let (models, mut rows) = current();
    let mut extra = rows[0].clone();
    extra.family = "unregistered-family".to_owned();
    rows.push(extra);
    assert!(
        coverage(&models, &rows).contains(&Failure::Unregistered("unregistered-family".to_owned()))
    );
    let family = rows[0].family.clone();
    rows[0].mode = "wrong-mode".to_owned();
    assert!(coverage(&models, &rows).contains(&Failure::Mode(family)));
}
