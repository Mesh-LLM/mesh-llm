use super::*;
use serde_json::{Value, json};

fn release(id: u64, prerelease: bool, published: &str) -> Value {
    json!({"id":id,"tag_name":format!("b{id}"),"draft":false,
        "prerelease":prerelease,"published_at":published})
}
fn page(nodes: Vec<Value>, cursor: Option<&str>) -> Value {
    json!({"data":{"repository":{"releases":{
        "nodes":nodes,"pageInfo":{"hasNextPage":cursor.is_some(),"endCursor":cursor}
    }}}})
}

#[test]
fn publication_order_beats_pinned_stable_feed_position_and_release_kind() {
    for prerelease in [false, true] {
        let rows = vec![
            release(10, !prerelease, "2026-10-15T17:13:19Z"),
            release(11, prerelease, "2026-10-16T00:18:22Z"),
            release(12, !prerelease, "2026-10-14T00:00:00Z"),
        ];
        let selected = newest(&mut |_| Ok(page(rows.clone(), None))).unwrap();
        assert_eq!(selected.id, 11);
        assert_eq!(selected.prerelease, prerelease);
    }
}

#[test]
fn later_pages_are_compared_and_drafts_are_excluded() {
    let mut draft = release(30, false, "2027-01-01T00:00:00Z");
    draft["draft"] = json!(true);
    draft["published_at"] = Value::Null;
    let mut calls = 0;
    let selected = newest(&mut |request| {
        calls += 1;
        match calls {
            1 => {
                assert_eq!(*request, Request::ReleasePage(None));
                Ok(page(
                    vec![draft.clone(), release(20, false, "2026-10-15T00:00:00Z")],
                    Some("next"),
                ))
            }
            2 => {
                assert_eq!(*request, Request::ReleasePage(Some("next".into())));
                Ok(page(vec![release(10, true, "2026-10-16T00:00:00Z")], None))
            }
            _ => panic!("unexpected page"),
        }
    })
    .unwrap();
    assert_eq!(selected.id, 10);
    assert_eq!(calls, 2);
}

#[test]
fn empty_feed_and_graphql_errors_never_select_a_partial_result() {
    for response in [
        page(vec![], None),
        json!({}),
        json!({"errors":[{"message":"no access"}],"data":{"repository":null}}),
    ] {
        assert!(newest(&mut |_| Ok(response.clone())).is_err());
    }
    let mut calls = 0;
    assert!(
        newest(&mut |_| {
            calls += 1;
            if calls == 1 {
                Ok(page(
                    vec![release(1, false, "2026-10-16T00:00:00Z")],
                    Some("next"),
                ))
            } else {
                Err("API unavailable".into())
            }
        })
        .is_err()
    );
    assert_eq!(calls, 2);
}

#[test]
fn publication_ties_use_release_identity_deterministically() {
    let rows = vec![
        release(2, true, "2026-10-16T00:00:00Z"),
        release(1, false, "2026-10-16T00:00:00Z"),
    ];
    assert_eq!(newest(&mut |_| Ok(page(rows.clone(), None))).unwrap().id, 2);
}

#[test]
fn duplicate_release_and_cursor_or_incomplete_pages_are_rejected() {
    let row = release(1, true, "2026-10-16T00:00:00Z");
    for response in [
        page(vec![row.clone(), row.clone()], None),
        page(vec![], Some("next")),
        page(vec![row.clone()], Some("")),
        page(vec![row.clone(); 101], None),
    ] {
        assert!(newest(&mut |_| Ok(response.clone())).is_err());
    }
    let mut id = 0;
    let error = newest(&mut |_| {
        id += 1;
        Ok(page(
            vec![release(id, false, "2026-10-16T00:00:00Z")],
            Some("same"),
        ))
    })
    .err()
    .unwrap();
    assert!(error.to_string().contains("cursor"));
}

#[test]
fn invalid_publication_dates_fail_instead_of_guessing_order() {
    for timestamp in [
        "",
        "2026-10-16",
        "not-a-date",
        "2026-02-30T00:00:00Z",
        "2026-10-16T25:00:00Z",
        "2026-10-16T00:00:00+00:00",
        "0000-10-16T00:00:00Z",
    ] {
        assert!(validate_timestamp(timestamp).is_err());
    }
    assert!(validate_timestamp("2024-02-29T23:59:59Z").is_ok());
    assert!(validate_timestamp("2025-02-29T23:59:59Z").is_err());
}

#[test]
fn pagination_limit_never_returns_a_partial_selection() {
    let mut id = 0;
    let error = newest(&mut |_| {
        id += 1;
        Ok(page(
            vec![release(id, true, "2026-10-16T00:00:00Z")],
            Some(&id.to_string()),
        ))
    })
    .err()
    .unwrap();
    assert_eq!(id, 1000);
    assert!(error.to_string().contains("pagination"));
}
