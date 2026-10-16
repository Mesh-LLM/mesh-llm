use super::*;
use serde_json::{Value, json};
use std::collections::VecDeque;

const COMMIT: &str = "1111111111111111111111111111111111111111";
const TAG: &str = "2222222222222222222222222222222222222222";

fn release() -> Value {
    json!({"id":123,"tag_name":"v0.7.0","draft":false,"prerelease":false,
        "created_at":"2026-10-15T17:13:19Z","published_at":"2026-10-15T17:13:19Z",
        "target_commitish":"master"})
}
fn reference(kind: &str, sha: &str) -> Value {
    json!({"ref":"refs/tags/v0.7.0","object":{"type":kind,"sha":sha}})
}
fn fixture(requested: &str, rows: Vec<(&str, Value)>) -> DynResult<Selection> {
    let mut rows: VecDeque<_> = rows.into();
    let result = select(requested, |endpoint| {
        let (expected, response) = rows.pop_front().expect("unexpected API request");
        if expected == "release-page" {
            assert_eq!(*endpoint, github::Request::ReleasePage(None));
            Ok(json!({"data":{"repository":{"releases":{
                "nodes":[response], "pageInfo":{"hasNextPage":false,"endCursor":null}
            }}}}))
        } else {
            assert_eq!(
                *endpoint,
                github::Request::Rest(format!("{REPOSITORY}/{expected}"))
            );
            Ok(response)
        }
    });
    assert!(rows.is_empty(), "fixture left unconsumed responses");
    result
}

#[test]
fn upstream_release_peels_annotated_tag_instead_of_target_commitish() {
    let selected = fixture(
        "",
        vec![
            ("release-page", release()),
            ("git/ref/tags/v0.7.0", reference("tag", TAG)),
            (
                "git/tags/2222222222222222222222222222222222222222",
                json!({"sha":TAG,"object":{"type":"commit","sha":COMMIT}}),
            ),
            ("git/ref/tags/v0.7.0", reference("tag", TAG)),
        ],
    )
    .unwrap();
    assert_eq!(selected.sha, COMMIT);
    assert_eq!(selected.release_tag.as_deref(), Some("v0.7.0"));
    assert!(selected.summary().contains("releases/tag/v0.7.0"));
    assert!(selected.summary().contains(COMMIT));
}

#[test]
fn upstream_release_latest_alias_and_lightweight_tag() {
    let selected = fixture(
        "latest",
        vec![
            ("release-page", release()),
            ("git/ref/tags/v0.7.0", reference("commit", COMMIT)),
            ("git/ref/tags/v0.7.0", reference("commit", COMMIT)),
        ],
    )
    .unwrap();
    assert_eq!(selected.sha, COMMIT);
}

#[test]
fn upstream_release_explicit_commit_needs_no_network() {
    let selected = fixture(COMMIT, vec![]).unwrap();
    assert_eq!(selected.sha, COMMIT);
    assert!(selected.release_tag.is_none());
    assert!(selected.summary().contains("override"));
    for invalid in [
        "master",
        "v0.7.0",
        "b11669",
        " abc",
        "ABCDEF",
        "$(bad)",
        &"A".repeat(40),
    ] {
        assert!(fixture(invalid, vec![]).is_err());
    }
}

#[test]
fn upstream_release_rejects_malformed_or_unpublished_metadata() {
    for (key, value) in [
        ("draft", json!(true)),
        ("published_at", Value::Null),
        ("published_at", json!("")),
        ("id", json!(0)),
        ("tag_name", json!("v0.7.0/../master")),
    ] {
        let mut input = release();
        input[key] = value;
        assert!(fixture("", vec![("release-page", input)]).is_err());
    }
    assert!(fixture("", vec![("release-page", json!({"tag_name":"v0.7.0"}))]).is_err());
}

#[test]
fn upstream_release_does_not_fallback_on_api_failure() {
    let mut calls = 0;
    let error = select("", |_| {
        calls += 1;
        Err("HTTP 404, 403, 429 or transport failure".into())
    })
    .unwrap_err();
    assert_eq!(calls, 1);
    assert!(error.to_string().contains("transport failure"));
}

#[test]
fn upstream_release_rejects_moved_tag() {
    assert!(
        fixture(
            "",
            vec![
                ("release-page", release()),
                ("git/ref/tags/v0.7.0", reference("commit", COMMIT)),
                ("git/ref/tags/v0.7.0", reference("commit", TAG)),
            ]
        )
        .unwrap_err()
        .to_string()
        .contains("moved")
    );
}

#[test]
fn upstream_release_rejects_noncommit_and_wrong_ref() {
    for input in [
        reference("tree", COMMIT),
        reference("commit", "bad"),
        json!({"ref":"refs/heads/master","object":{"type":"commit","sha":COMMIT}}),
    ] {
        assert!(
            fixture(
                "",
                vec![("release-page", release()), ("git/ref/tags/v0.7.0", input)]
            )
            .is_err()
        );
    }
}

#[test]
fn upstream_release_rejects_tag_cycle_and_wrong_object_identity() {
    for input in [
        json!({"sha":TAG,"object":{"type":"tag","sha":TAG}}),
        json!({"sha":COMMIT,"object":{"type":"commit","sha":COMMIT}}),
    ] {
        assert!(
            fixture(
                "",
                vec![
                    ("release-page", release()),
                    ("git/ref/tags/v0.7.0", reference("tag", TAG)),
                    ("git/tags/2222222222222222222222222222222222222222", input),
                ]
            )
            .is_err()
        );
    }
}

#[test]
fn upstream_release_nested_tags_are_bounded() {
    let mut count = 0;
    let error = peel(
        Object {
            sha: COMMIT.into(),
            kind: "tag".into(),
        },
        &mut |endpoint| {
            let github::Request::Rest(endpoint) = endpoint else {
                panic!("expected tag request")
            };
            let old = endpoint.rsplit('/').next().unwrap();
            count += 1;
            Ok(json!({"sha":old,"object":{"type":"tag","sha":format!("{count:040x}")}}))
        },
    )
    .unwrap_err();
    assert_eq!(count, 8);
    assert!(error.to_string().contains("nesting"));
}
