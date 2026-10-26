use super::*;
use serde_json::json;
fn encoded(value: serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap()
}
fn embedding(index: serde_json::Value, values: serde_json::Value) -> Vec<u8> {
    encoded(json!({"data":[{"index":index,"embedding":values}]}))
}
#[test]
fn embedding_comparison_requires_typed_indexes_dimensions_nonzero_and_both_numeric_gates() {
    let reference = embedding(json!(0), json!([1.0, 0.0]));
    assert!(numeric::embeddings(&reference, &reference, 1).is_ok());
    for index in [
        json!(false),
        json!(0.0),
        json!("0"),
        json!(null),
        json!(-1),
        json!(1),
    ] {
        assert!(numeric::embeddings(&embedding(index, json!([1, 0])), &reference, 1).is_err());
    }
    for vector in [
        json!([]),
        json!([0, 0]),
        json!([true, 0]),
        json!([1]),
        json!([0.9, 0.1]),
    ] {
        assert!(numeric::embeddings(&embedding(json!(0), vector), &reference, 1).is_err());
    }
    // Small component deltas alone cannot admit orthogonal tiny vectors.
    let tiny_a = embedding(json!(0), json!([1e-6, 0]));
    let tiny_b = embedding(json!(0), json!([0, 1e-6]));
    assert!(numeric::embeddings(&tiny_a, &tiny_b, 1).is_err());
    // Finite large values must not overflow intermediate norms to a false verdict.
    let large = embedding(json!(0), json!([1e300, 1e300]));
    assert!(numeric::embeddings(&large, &large, 1).is_ok());
    assert!(numeric::embeddings(&reference, &reference, 3).is_err());
    assert!(
        numeric::embeddings(
            b"{\"data\":[{\"index\":0,\"embedding\":[1e999]}]}",
            &reference,
            1
        )
        .is_err()
    );
}
#[test]
fn rerank_comparison_preserves_wire_order_and_document_score_identity() {
    let reference =
        json!({"results":[{"index":0,"relevance_score":2},{"index":1,"relevance_score":-1}]});
    let bytes = encoded(reference.clone());
    assert!(numeric::rerank(&bytes, &bytes).is_ok());
    let mut changed = reference.clone();
    changed["results"].as_array_mut().unwrap().reverse();
    assert!(numeric::rerank(&encoded(changed), &bytes).is_err());
    let mut changed = reference.clone();
    changed["results"][0]["relevance_score"] = json!(2.001);
    assert!(numeric::rerank(&encoded(changed), &bytes).is_err());
    for index in [json!(false), json!(0.0), json!(1), json!(2)] {
        let mut changed = reference.clone();
        changed["results"][0]["index"] = index;
        assert!(numeric::rerank(&encoded(changed), &bytes).is_err());
    }
}
#[test]
fn translation_compares_only_whitespace_normalized_generated_text() {
    let candidate = encoded(json!({"choices":[{"text":"  Das\tHaus ist wunderbar.\n"}]}));
    assert!(native::compare(&candidate, "Das Haus ist wunderbar.").is_ok());
    assert!(native::compare(&candidate, "Das Auto ist wunderbar.").is_err());
    for response in [
        json!({"choices":[]}),
        json!({"choices":[{"text":" "}]}),
        json!({"choices":[{"text":true}]}),
        json!({"choices":[{"text":"a"},{"text":"b"}]}),
    ] {
        assert!(native::compare(&encoded(response), "a").is_err());
    }
}
#[test]
fn options_reject_cross_class_or_missing_reference_before_execution() {
    let args = |words: &[&str]| {
        words
            .iter()
            .map(|word| (*word).to_owned())
            .collect::<Vec<_>>()
    };
    let common = [
        "--candidate-url",
        "http://127.0.0.1/v1",
        "--model",
        "fixture",
        "--class",
        "embedding",
    ];
    assert!(Options::parse(&args(&common)).is_err());
    let mut supplied = common.to_vec();
    supplied.extend(["--oracle-url", "http://127.0.0.1/v1"]);
    assert!(Options::parse(&args(&supplied)).is_ok());
    supplied.extend(["--oracle-completion", "/fake"]);
    assert!(Options::parse(&args(&supplied)).is_err());
}
