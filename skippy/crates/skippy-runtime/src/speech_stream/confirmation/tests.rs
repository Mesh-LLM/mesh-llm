use super::*;

mod occurrences;

/// Splits text into the word units the stream confirms.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

/// Builds a confirmed prefix from a space-separated word list.
fn prefix(text: &str) -> ConfirmedPrefix {
    let mut prefix = ConfirmedPrefix::default();
    prefix.append(&words(text));
    prefix
}

/// Counts the anchors that own a canonical word.
fn ownership(prefix: &ConfirmedPrefix) -> usize {
    prefix
        .anchors
        .iter()
        .filter(|anchor| anchor.canonical.is_some())
        .count()
}

/// Canonical and observed spellings can return after an omission.
#[test]
fn canonical_and_observed_spellings_can_return_after_an_omission() {
    for restored in ["asked", "ask"] {
        let mut prefix = prefix("Hello asked");
        assert_eq!(prefix.align(&words("Hello ask")), Some(2));
        assert_eq!(prefix.align(&words("Hello")), Some(1));
        assert_eq!(
            prefix.align(&words(&format!("Hello {restored} today"))),
            Some(2)
        );
        assert_eq!(ownership(&prefix), 2);
        assert_eq!(
            prefix.anchors.last().unwrap().canonical.as_deref(),
            Some("asked")
        );
    }
}

/// A known canonical spelling does not consume a genuine new repetition.
#[test]
fn a_known_canonical_spelling_does_not_consume_a_genuine_new_repetition() {
    let mut prefix = prefix("Hello asked");
    assert_eq!(prefix.align(&words("Hello ask")), Some(2));
    assert_eq!(prefix.align(&words("Hello ask asked")), Some(2));
    assert_eq!(ownership(&prefix), 2);
}

/// A changed prefix does not match a confirmed word to a later repetition.
#[test]
fn a_changed_prefix_does_not_match_a_confirmed_word_to_a_later_repetition() {
    // The prefix is revised (`hello` -> `hi`) and the speaker repeats the
    // confirmed word: the confirmed `ask` keeps its earlier occurrence, so
    // the new repetition stays outside the consumed prefix for the delta.
    let mut prefix = prefix("hello ask not");
    assert_eq!(prefix.align(&words("hi ask not ask")), Some(3));
    assert_eq!(prefix.align(&words("hi ask not ask ask")), Some(3));
}

/// Substitutions and temporary omissions preserve ownership.
#[test]
fn substitutions_and_temporary_omissions_preserve_ownership() {
    let mut prefix = prefix("hello wrong ready ask not");
    assert_eq!(prefix.align(&words("hello right ready")), Some(3));
    assert_eq!(ownership(&prefix), 5);
    assert_eq!(prefix.observed_len(), 3);
    assert_eq!(
        prefix.align(&words("hello right ready ask not today")),
        Some(5)
    );
    assert_eq!(ownership(&prefix), 5);
    assert_eq!(prefix.trim(5), 5);
    assert!(prefix.anchors.is_empty());
}

/// An omitted word cannot consume a genuine new tail as a substitution.
#[test]
fn an_omitted_word_cannot_consume_a_genuine_new_tail_as_a_substitution() {
    let mut prefix = prefix("A B C D");
    assert_eq!(prefix.align(&words("A B D")), Some(3));
    assert_eq!(prefix.align(&words("A B D E")), Some(3));
    prefix.append(&words("E"));
    assert_eq!(prefix.align(&words("A B C D E F")), Some(5));
    assert_eq!(ownership(&prefix), 5);
}

/// Missing suffix ownership survives an early segment cut.
#[test]
fn missing_suffix_ownership_survives_an_early_segment_cut() {
    let mut prefix = prefix("A B C D");
    assert_eq!(prefix.align(&words("A B")), Some(2));
    assert_eq!(prefix.trim(2), 2);
    assert_eq!(ownership(&prefix), 2);
    assert_eq!(prefix.align(&words("C D E")), Some(2));
    assert_eq!(prefix.trim(2), 2);
}

/// Missing prefix ownership is removed with a later segment cut.
#[test]
fn missing_prefix_ownership_is_removed_with_a_later_segment_cut() {
    let mut prefix = prefix("A B C D");
    assert_eq!(prefix.align(&words("C D E")), Some(2));
    assert_eq!(prefix.trim(2), 4);
    assert!(prefix.anchors.is_empty());
}

/// An insertion inside consumed words owns no output word.
#[test]
fn an_insertion_inside_consumed_words_owns_no_output_word() {
    let mut prefix = prefix("A C D");
    assert_eq!(prefix.align(&words("A B C D E")), Some(4));
    assert_eq!(ownership(&prefix), 3);
    assert_eq!(prefix.trim(2), 1);
    assert_eq!(ownership(&prefix), 2);
    assert_eq!(prefix.align(&words("C D E")), Some(2));
}

/// A genuine repetition after restored words remains available.
#[test]
fn genuine_repetition_after_restored_words_remains_available() {
    let mut prefix = prefix("Go now Go now");
    assert_eq!(prefix.align(&words("Go now")), Some(2));
    assert_eq!(prefix.align(&words("Go now Go now Go now")), Some(4));
    prefix.append(&words("Go now"));
    assert_eq!(ownership(&prefix), 6);
    assert_eq!(prefix.align(&words("Go now Go now Go now Go now")), Some(6));
}

/// An omitted start does not make aligned confirmed words pending.
#[test]
fn an_omitted_start_does_not_make_aligned_confirmed_words_pending() {
    let mut prefix = prefix("Hello world ask not");
    assert_eq!(
        prefix.align_with_pending(
            &words("ask not today"),
            &words("ask"),
            &words("Hello world ask not"),
        ),
        Some(2),
    );
    assert_eq!(ownership(&prefix), 4);
    assert_eq!(prefix.observed_len(), 2);
}

/// An observed substitution is not weaker than matching pending words.
#[test]
fn an_observed_substitution_is_not_weaker_than_matching_pending_words() {
    let mut prefix = prefix("Hello Alice");
    assert_eq!(prefix.align(&words("Hello Bob Hello Bob")), Some(2));
    assert_eq!(
        prefix.align_with_pending(
            &words("Hello Bob today"),
            &words("Hello Bob"),
            &words("Hello Alice"),
        ),
        Some(2),
    );
    assert_eq!(ownership(&prefix), 2);
    assert_eq!(prefix.observed_len(), 2);
}

/// Pending evidence keeps omitted ownership for a restored prefix.
#[test]
fn pending_evidence_keeps_omitted_ownership_for_a_restored_prefix() {
    let mut prefix = prefix("for you");
    assert_eq!(
        prefix.align_with_pending(
            &words("ask what you can do"),
            &words("ask what you do"),
            &words("for you"),
        ),
        Some(0),
    );
    assert_eq!(ownership(&prefix), 2);
    assert_eq!(prefix.observed_len(), 0);
    prefix.append(&words("ask what you"));
    assert_eq!(prefix.align(&words("for you ask what you can do")), Some(5));
    assert_eq!(ownership(&prefix), 5);
}

/// Unsupported hypotheses leave ownership unchanged.
#[test]
fn unsupported_hypotheses_leave_ownership_unchanged() {
    let mut prefix = prefix("A B");
    assert_eq!(prefix.align(&words("C D")), None);
    assert_eq!(ownership(&prefix), 2);
    assert_eq!(prefix.align(&words("A B E")), Some(2));
}

/// Discarded insertions do not accumulate history.
#[test]
fn discarded_insertions_do_not_accumulate_history() {
    let mut prefix = prefix("A D");
    for insertion in ["B", "C", "E", "F"] {
        assert_eq!(prefix.align(&words(&format!("A {insertion} D"))), Some(3));
        assert_eq!(prefix.align(&words("A D")), Some(2));
        assert_eq!(prefix.anchors.len(), 2);
        assert_eq!(ownership(&prefix), 2);
    }
    prefix.clear();
    assert!(prefix.anchors.is_empty());
}

/// Unsupported boundaries use the capped observed length without mutation.
#[test]
fn unsupported_boundaries_use_the_capped_observed_length_without_mutation() {
    let mut prefix = prefix("A B");
    for (text, expected) in [("X Y tail", 2), ("X", 1), ("", 0)] {
        assert_eq!(prefix.unsupported_boundary(&words(text), &[]), expected);
        assert_eq!(ownership(&prefix), 2);
        assert_eq!(prefix.observed_len(), 2);
    }
    assert_eq!(prefix.align(&words("A B next")), Some(2));
}

/// Restored owned spellings align only the prefix and preserve the tail.
#[test]
fn restored_owned_spellings_align_only_the_prefix_and_preserve_the_tail() {
    for spelling in ["Alice", "Bob"] {
        let mut prefix = prefix("Hello Alice");
        assert_eq!(prefix.align(&words("Hello Bob")), Some(2));
        assert_eq!(prefix.align(&words("Hello")), Some(1));
        let hypothesis = words(&format!("X Y {spelling} fresh new words"));
        assert_eq!(prefix.align(&hypothesis), None);
        assert_eq!(prefix.unsupported_boundary(&hypothesis, &[]), 3);
        assert_eq!(prefix.observed_len(), 3);
        assert_eq!(ownership(&prefix), 2);
    }
}

/// Recovering an omitted owner updates the observed prefix.
#[test]
fn recovering_an_omitted_owner_updates_the_observed_prefix() {
    let mut prefix = prefix("A B");
    assert_eq!(prefix.align(&words("A")), Some(1));
    let hypothesis = words("X Y B");
    assert_eq!(prefix.align(&hypothesis), None);
    assert_eq!(prefix.unsupported_boundary(&hypothesis, &[]), 3);
    assert_eq!(ownership(&prefix), 2);
    assert_eq!(prefix.observed_len(), 3);
    assert_eq!(prefix.align(&words("A B next")), Some(2));
}

/// Recovery retains supported paths that the unconstrained cells discard.
#[test]
fn recovery_retains_supported_paths_that_the_unconstrained_cells_discard() {
    let mut prefix = prefix("A B C");
    let hypothesis = words("X Y Z Q A fresh words");
    assert!(prefix.alignment(&hypothesis, &[], None).is_none());
    assert_eq!(prefix.unsupported_boundary(&hypothesis, &[]), 5);
    assert_eq!(prefix.observed_len(), 5);
    assert_eq!(ownership(&prefix), 3);
}

/// Recovery consumes all restored owners before exposing the tail.
#[test]
fn recovery_consumes_all_restored_owners_before_exposing_the_tail() {
    for (text, end) in [("X Y B filler C today", 5), ("X Y C B Z C today", 6)] {
        let mut prefix = prefix("A B C");
        assert_eq!(prefix.align(&words("A")), Some(1));
        let hypothesis = words(text);
        assert_eq!(prefix.align(&hypothesis), None);
        assert_eq!(prefix.unsupported_boundary(&hypothesis, &[]), end);
        assert_eq!(prefix.observed_len(), end);
        assert_eq!(ownership(&prefix), 3);
    }
}

/// Unsupported boundaries count observed insertions without treating them as owners.
#[test]
fn unsupported_boundaries_count_observed_insertions_without_treating_them_as_owners() {
    let mut prefix = prefix("A C");
    assert_eq!(prefix.align(&words("A B C")), Some(3));
    assert_eq!(ownership(&prefix), 2);
    assert_eq!(prefix.observed_len(), 3);
    // B was heard inside the owned span, but was never emitted itself.
    assert_eq!(prefix.unsupported_boundary(&words("X Y Z B tail"), &[]), 3);
    assert_eq!(ownership(&prefix), 2);
    assert_eq!(prefix.observed_len(), 3);
    assert_eq!(prefix.align(&words("A B C next")), Some(3));
}

/// Repeated pending speech can shorten but not extend an unsupported prefix.
#[test]
fn repeated_pending_speech_can_shorten_but_not_extend_an_unsupported_prefix() {
    for (text, pending, expected) in [
        ("X Y fresh", "fresh", 2),
        ("X Y fresh words", "fresh words", 2),
        ("X fresh Y fresh", "fresh", 3),
        ("X Y Z Q fresh", "fresh", 3),
        ("X Y Z", "missing", 3),
    ] {
        let mut prefix = prefix("A C");
        assert_eq!(prefix.align(&words("A B C")), Some(3));
        let hypothesis = words(text);
        assert_eq!(prefix.align(&hypothesis), None);
        assert_eq!(
            prefix.unsupported_boundary(&hypothesis, &words(pending)),
            expected,
            "{text}"
        );
        assert_eq!(prefix.observed_len(), expected);
        assert_eq!(ownership(&prefix), 2);
        assert_eq!(
            prefix
                .anchors
                .iter()
                .map(|anchor| (
                    anchor.text.as_str(),
                    anchor.canonical.as_deref(),
                    anchor.observed
                ))
                .collect::<Vec<_>>(),
            [
                ("A", Some("A"), true),
                ("B", None, true),
                ("C", Some("C"), true)
            ]
        );
        assert_eq!(prefix.align(&words("A B C next")), Some(3));
        assert_eq!(prefix.observed_len(), 3);
        assert_eq!(prefix.trim(2), 1);
        assert_eq!(prefix.observed_len(), 1);
        assert_eq!(ownership(&prefix), 1);
    }
}
