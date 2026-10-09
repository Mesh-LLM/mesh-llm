//! Deterministic bounds when several policies or dispatches contribute metadata.
use std::collections::BTreeMap;
pub(super) fn response_header_prefix(name: &str) -> String {
    format!("x-plugin-{}-", hex::encode(name.as_bytes()))
}

pub(super) fn merge_headers(
    target: &mut Vec<(String, String)>,
    incoming: Vec<(String, String)>,
) -> bool {
    let mut merged: BTreeMap<_, _> = target
        .drain(..)
        .chain(incoming)
        .map(|(name, value)| (name.to_ascii_lowercase(), value))
        .collect();
    let complete = merged.len() <= 16;
    while merged.len() > 16 {
        merged.pop_last();
    }
    target.extend(merged);
    complete
}

pub(super) fn merge_annotations(
    target: &mut BTreeMap<String, String>,
    incoming: BTreeMap<String, String>,
) -> bool {
    target.extend(incoming);
    let mut budgets = BTreeMap::<String, usize>::new();
    let mut complete = true;
    target.retain(|key, value| {
        let author = key.split('.').next().unwrap_or_default().to_owned();
        let budget = budgets.entry(author).or_default();
        let size = key.len().saturating_add(value.len());
        if budget.saturating_add(size) > 4096 {
            complete = false;
            false
        } else {
            *budget += size;
            true
        }
    });
    complete
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn response_header_namespaces_do_not_alias_punctuation_or_case() {
        let names: std::collections::BTreeSet<_> = ["foo.bar", "foo_bar", "foo-bar", "Foo.bar"]
            .map(response_header_prefix)
            .into();
        assert_eq!(names.len(), 4);
        assert_eq!(
            response_header_prefix("foo.bar"),
            "x-plugin-666f6f2e626172-"
        );
    }
    #[test]
    fn aggregate_headers_are_bounded_and_repeated_phases_replace_values() {
        let first: Vec<_> = (0..16)
            .map(|id| (format!("x-plugin-a-{id:02}"), "original".into()))
            .collect();
        let second: Vec<_> = (0..16)
            .map(|id| (format!("x-plugin-b-{id:02}"), "selected".into()))
            .collect();
        let mut forward = first.clone();
        assert!(!merge_headers(&mut forward, second.clone()));
        let mut reverse = second;
        assert!(!merge_headers(&mut reverse, first));
        assert_eq!(forward, reverse);
        assert_eq!(forward.len(), 16);
        assert!(merge_headers(
            &mut forward,
            vec![("x-plugin-a-00".into(), "selected".into())]
        ));
        assert_eq!(forward[0].1, "selected");
    }
    #[test]
    fn annotation_budget_is_per_author_and_bounded_across_dispatches() {
        let mut annotations = BTreeMap::new();
        assert!(!merge_annotations(
            &mut annotations,
            (0..8)
                .map(|id| (format!("observer.{id}"), "x".repeat(1024)))
                .collect()
        ));
        assert!(
            annotations
                .iter()
                .map(|(key, value)| key.len() + value.len())
                .sum::<usize>()
                <= 4096
        );
        assert!(merge_annotations(
            &mut annotations,
            [("another.receipt".into(), "safe".into())].into()
        ));
    }
}
