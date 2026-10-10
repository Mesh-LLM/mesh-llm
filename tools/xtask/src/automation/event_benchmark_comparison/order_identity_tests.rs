use super::*;

fn records() -> Vec<Order> {
    (0..2)
        .map(|pair_index| Order {
            scenario: "__primary__".into(),
            pair_index,
            order: if pair_index == 0 {
                ["production".into(), "event-disabled".into()]
            } else {
                ["event-disabled".into(), "production".into()]
            },
        })
        .collect()
}
fn keys() -> Keys {
    (0..2).map(|index| ("__primary__".into(), index)).collect()
}
fn check(a: Option<&[Order]>, b: Option<&[Order]>) -> Vec<String> {
    compare(a, b, ["production", "event-disabled"], &keys())
}

#[test]
fn complete_matching_observed_order_preserves_both_side_orders() {
    let records = records();
    assert!(check(Some(&records), Some(&records)).is_empty());
}

#[test]
fn missing_order_names_each_actual_missing_manifest() {
    let records = records();
    assert_eq!(check(None, None).len(), 2);
    assert!(check(None, Some(&records))[0].starts_with("production manifest"));
    assert!(check(Some(&records), Some(&[]))[0].starts_with("event_disabled manifest"));
}

#[test]
fn different_or_constant_observed_order_blocks() {
    let records = records();
    let mut changed = records.clone();
    changed[1].order.swap(0, 1);
    assert!(check(Some(&records), Some(&changed))[0].contains("disagrees"));
    assert!(check(Some(&changed), Some(&changed))[0].contains("constant"));
}

#[test]
fn duplicate_missing_or_wrong_side_identity_cannot_fake_observed_order() {
    let records = records();
    let duplicate = [records[0].clone(), records[0].clone()];
    assert!(
        check(Some(&duplicate), Some(&duplicate))
            .iter()
            .any(|v| v.contains("duplicate"))
    );
    assert!(
        check(Some(&records[..1]), Some(&records[..1]))
            .iter()
            .any(|v| v.contains("census"))
    );
    let mut changed = records;
    changed[0].order[0] = "unrelated".into();
    assert!(
        check(Some(&changed), Some(&changed))
            .iter()
            .any(|v| v.contains("side identity"))
    );
}
