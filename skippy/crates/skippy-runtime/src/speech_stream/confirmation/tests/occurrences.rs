//! Confirmed word ownership must follow the same occurrence through revisions.

use super::*;

// Exact post-93 state from the base.en/MTL0/400 ms reproduction of issue #2779.
// Model revision: 5359861c739e955e79d9a303bcbc70fb988958b1.
// Model SHA256: a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002.
// JFK upstream revision: 231638a0717b7822784297e5980965ffcec05810.
// Four-copy PCM SHA256: 8cda406b2cf6e25a495646c906762a4033129b4acdf09f4da40d3502f71c2008.
// The first four and the earlier "your" anchor were omitted; "you" was inserted.
/// Rebuilds the confirmed prefix captured from the admitted reproduction.
fn captured_prefix() -> ConfirmedPrefix {
    let entries = [
        ("and", Some("and"), false),
        ("so", Some("so"), false),
        ("my", Some("my"), false),
        ("fellow", Some("fellow"), false),
        (",", Some("Americans!"), true),
        ("ask", Some("ask"), true),
        ("not!", Some("not!"), true),
        ("what", Some("what"), true),
        ("you", None, true),
        ("your", Some("your"), false),
        ("country", Some("country"), true),
        ("can", Some("can"), true),
        ("do", Some("do."), true),
        ("for", Some("for"), true),
        ("you.", Some("you."), true),
        ("ask", Some("ask"), true),
        ("what", Some("what"), true),
        ("you", Some("you"), true),
        ("can", Some("can"), true),
        ("do", Some("do."), true),
        ("for", Some("for"), true),
        ("your", Some("your"), true),
        ("country.", Some("country."), true),
        ("and", Some("and"), true),
        ("so", Some("so"), true),
        ("my", Some("my"), true),
        ("fellow", Some("fellow"), true),
        ("Americans...", Some("Americans..."), true),
        ("asked", Some("asked"), true),
    ];
    ConfirmedPrefix {
        observed_words: entries.iter().filter(|(_, _, observed)| *observed).count(),
        anchors: entries
            .into_iter()
            .map(|(text, canonical, observed)| Anchor {
                text: text.into(),
                canonical: canonical.map(str::to_string),
                observed,
            })
            .collect(),
    }
}

const CANONICAL: &str = "and so my fellow Americans! ask not! what your country can do. for you. ask what you can do. for your country. and so my fellow Americans... asked";
const PASS94: &str = ", ask what you can do for your country. ask not! what you country can do for you. ask what you country can do for your country. ask not!";
const PASS95: &str = ", ask what you can do for your country. ask not! what you country can do for you. ask what you country can do for your country. and so my fellow Americans... asked not!";

/// A restored opening keeps its confirmed occurrence after reordered clauses.
#[test]
fn a_restored_opening_keeps_its_confirmed_occurrence_after_reordered_clauses() {
    let mut prefix = captured_prefix();
    let canonical = words(CANONICAL);
    assert_eq!(ownership(&prefix), 28);
    assert_eq!(prefix.observed_len(), 24);
    assert_eq!(canonical.len(), 28);

    let pass94 = words(PASS94);
    let end94 = prefix
        .align_with_pending(&pass94, &words("not!"), &canonical)
        .unwrap();
    let pending94 = &pass94[end94..];
    let pass95 = words(PASS95);
    let end95 = prefix
        .align_with_pending(&pass95, pending94, &canonical)
        .unwrap();
    let agreed = pass95[end95..]
        .iter()
        .zip(pending94)
        .take_while(|(a, b)| same_word(a, b))
        .count();
    // The same last opening is present again at 95, after the earlier clause.
    // It already owns "asked". The earlier "for your country" is not new audio.
    assert_eq!(
        end95,
        pass95.len() - 1,
        "consume through the restored fourth asked, leaving not pending"
    );
    assert!(
        pass95[end95..end95 + agreed]
            .iter()
            .all(|word| same_word(word, "not")),
        "the earlier clause is not new confirmed speech"
    );
    assert_eq!(ownership(&prefix), 28);
    assert_eq!(
        prefix.align_with_pending(&pass95, &pass95[end95..], &canonical),
        Some(end95),
        "restoring the opening must remain stable on the next pass"
    );
    assert_eq!(ownership(&prefix), 28);
}

/// Canonical aliases do not steal a genuine later repetition after revision.
#[test]
fn canonical_aliases_do_not_steal_a_genuine_later_repetition_after_revision() {
    for tail in ["can run", "today can run"] {
        let mut prefix = prefix("I can run");
        assert_eq!(prefix.align(&words("I could walk")), Some(3));
        let hypothesis = words(&format!("I could walk {tail}"));
        let end = prefix.align(&hypothesis).unwrap();
        assert_eq!(&hypothesis[end..], words(tail));
    }
}

/// Repeating the omitted opening does not reassign its occurrence.
#[test]
fn repeating_the_omitted_opening_does_not_reassign_its_occurrence() {
    let mut prefix = captured_prefix();
    let mut canonical = words(CANONICAL);
    let mut pending = words("not!");
    for _ in 0..2 {
        let emitted = apply_agreement(&mut prefix, &mut canonical, &mut pending, PASS94);
        assert!(
            emitted.iter().all(|word| same_word(word, "not")),
            "the repeated earlier clause was already confirmed: {emitted:?}"
        );
        assert_eq!(ownership(&prefix), canonical.len());
    }
    let pass95 = words(PASS95);
    let restored = prefix
        .align_with_pending(&pass95, &pending, &canonical)
        .unwrap();
    let already_confirmed_not = usize::from(canonical.len() == 29);
    assert_eq!(
        restored,
        pass95.len() - 1 + already_confirmed_not,
        "repeated earlier text must not make its aliases own the fourth opening"
    );
    assert_eq!(ownership(&prefix), canonical.len());
    assert_eq!(
        prefix.align_with_pending(&pass95, &pass95[restored..], &canonical),
        Some(restored),
        "a second restored hypothesis must keep the same occurrence"
    );
    assert_eq!(ownership(&prefix), canonical.len());
}

// Model the owning stream's agreement and append transitions, including the
// state mutation between repeated hypotheses. No trimming occurs in this
// captured window: its only segment ends beyond the available audio.
/// Applies one hypothesis through the owning stream's agreement and append.
fn apply_agreement(
    prefix: &mut ConfirmedPrefix,
    canonical: &mut Vec<String>,
    pending: &mut Vec<String>,
    text: &str,
) -> Vec<String> {
    let hypothesis = words(text);
    let end = prefix
        .align_with_pending(&hypothesis, pending, canonical)
        .unwrap();
    let tail = &hypothesis[end..];
    let agreed = tail
        .iter()
        .zip(pending.iter())
        .take_while(|(a, b)| same_word(a, b))
        .count();
    let emitted = tail[..agreed].to_vec();
    prefix.append(&emitted);
    canonical.extend_from_slice(&emitted);
    *pending = tail[agreed..].to_vec();
    emitted
}

/// An omitted alias recovers without charging a new occurrence.
#[test]
fn an_omitted_alias_recovers_without_charging_a_new_occurrence() {
    for restored in ["asked", "ask"] {
        let mut prefix = prefix("Hello asked");
        assert_eq!(prefix.align(&words("Hello ask")), Some(2));
        assert_eq!(prefix.align(&words("Hello")), Some(1));
        let hypothesis = words(&format!("Hello {restored} today"));
        let end = prefix.align(&hypothesis).unwrap();
        assert_eq!(&hypothesis[end..], words("today"));
    }
}

/// Omission does not create debt that consumes a later repetition.
#[test]
fn omission_does_not_create_debt_that_consumes_a_later_repetition() {
    let mut prefix = prefix("Hello Alice");
    assert_eq!(prefix.align(&words("Hello Bob")), Some(2));
    assert_eq!(prefix.align(&words("Hello")), Some(1));
    let hypothesis = words("Hello today Alice");
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("today Alice"));
}

/// Stable revisions do not stall a genuine later occurrence.
#[test]
fn stable_revisions_do_not_stall_a_genuine_later_occurrence() {
    let mut prefix = prefix("Hello Alice");
    let hypothesis = words("Hello Bob hello Alice");
    for _ in 0..3 {
        let end = prefix.align(&hypothesis).unwrap();
        assert_eq!(&hypothesis[end..], words("hello Alice"));
    }
}

/// Successive revisions do not accumulate alignment debt.
#[test]
fn successive_revisions_do_not_accumulate_alignment_debt() {
    let mut prefix = prefix("Hello Alice");
    for revised in ["Bob", "Charlie", "Dana", "Bob"] {
        let hypothesis = words(&format!("Hello {revised}"));
        assert_eq!(prefix.align(&hypothesis), Some(2));
        assert_eq!(prefix.align(&hypothesis), Some(2));
        assert_eq!(ownership(&prefix), 2);
    }
    let hypothesis = words("Hello Bob today Alice");
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("today Alice"));
}

/// A stable reorder keeps new speech available without replaying owned words.
#[test]
fn a_stable_reorder_keeps_new_speech_available_without_replaying_owned_words() {
    let mut prefix = prefix("A B C D");
    let reordered = words("A C B D E");
    for _ in 0..2 {
        let end = prefix.align(&reordered).unwrap();
        assert_eq!(&reordered[end..], words("E"));
        assert_eq!(ownership(&prefix), 4);
    }
    prefix.append(&words("E"));
    let restored = words("A B C D E F");
    let end = prefix.align(&restored).unwrap();
    assert_eq!(&restored[end..], words("F"));
    assert_eq!(ownership(&prefix), 5);
}

/// An inserted duplicate does not pin an owner that was not omitted.
#[test]
fn an_inserted_duplicate_does_not_pin_an_owner_that_was_not_omitted() {
    let mut prefix = prefix("A C");
    let hypothesis = words("A A C D");
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("D"));
    assert_eq!(ownership(&prefix), 2);
}

/// An already omitted owner is not pinned by a later insertion.
#[test]
fn an_already_omitted_owner_is_not_pinned_by_a_later_insertion() {
    let mut prefix = prefix("A B C D");
    assert_eq!(prefix.align(&words("A C D")), Some(3));
    let hypothesis = words("A C B D E");
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("E"));
    assert_eq!(ownership(&prefix), 4);
}

/// Omission and duplicate context keep the new tail available.
#[test]
fn omission_and_duplicate_context_keep_the_new_tail_available() {
    let mut prefix = prefix("X A X B");
    let hypothesis = words("A X X B tail");
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("tail"));
    assert_eq!(ownership(&prefix), 4);
    assert_eq!(prefix.align(&words("X A X B tail")), Some(4));
}

/// Collision matching uses the same case and punctuation rules.
#[test]
fn collision_matching_uses_the_same_case_and_punctuation_rules() {
    let mut prefix = prefix("A Bee C D");
    let hypothesis = words("a c BEE! d E");
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("E"));
    assert!(prefix.anchors.iter().all(|anchor| anchor.observed));
    assert_eq!(ownership(&prefix), 4);
    assert_eq!(prefix.anchors.len(), 4);
}

/// Different punctuation-only words do not create a collision.
#[test]
fn different_punctuation_only_words_do_not_create_a_collision() {
    let mut prefix = prefix("A ! C D");
    let hypothesis = words("A C ? D E");
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("E"));
    assert!(
        prefix
            .anchors
            .iter()
            .any(|anchor| { anchor.canonical.as_deref() == Some("!") && !anchor.observed })
    );
    assert!(
        prefix
            .anchors
            .iter()
            .any(|anchor| { anchor.canonical.is_none() && anchor.text == "?" })
    );
}

/// Impossible pins have no reachable alignment.
#[test]
fn impossible_pins_have_no_reachable_alignment() {
    let prefix = prefix("A B");
    let hypothesis = words("A");
    let cells = prefix.alignment_cells(&hypothesis, &[true, true], None);
    assert!(
        prefix
            .best_alignment(&cells, &hypothesis, &[0, 0], false)
            .is_none()
    );
    assert_eq!(ownership(&prefix), 2);
    assert_eq!(prefix.observed_len(), 2);
}

/// Many independent reorders preserve the complete new tail.
#[test]
fn many_independent_reorders_preserve_the_complete_new_tail() {
    let mut canonical = Vec::new();
    let mut hypothesis = Vec::new();
    for index in 0..32 {
        canonical.extend(["A", "B", "C", "D"].map(|word| format!("{word}{index}")));
        hypothesis.extend(["A", "C", "B", "D"].map(|word| format!("{word}{index}")));
    }
    let mut prefix = ConfirmedPrefix::default();
    prefix.append(&canonical);
    hypothesis.extend(words("new words"));
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("new words"));
    assert_eq!(ownership(&prefix), canonical.len());
}

/// Without a collision, an inserted word does not absorb a new tail.
#[test]
fn without_a_collision_an_inserted_word_does_not_absorb_a_new_tail() {
    let mut prefix = prefix("A B C");
    let hypothesis = words("A X B Y");
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("Y"));
    assert_eq!(ownership(&prefix), 3);
}

/// Pending evidence keeps a new tail after insertion without a collision.
#[test]
fn pending_evidence_keeps_a_new_tail_after_insertion_without_a_collision() {
    let canonical = words("A B C");
    let mut prefix = prefix("A B C");
    let hypothesis = words("A X B Y");
    let end = prefix
        .align_with_pending(&hypothesis, &words("Y"), &canonical)
        .unwrap();
    assert_eq!(&hypothesis[end..], words("Y"));
    assert_eq!(ownership(&prefix), 3);
}

/// An earlier reorder does not absorb an unrelated new tail without pending.
#[test]
fn an_earlier_reorder_does_not_absorb_an_unrelated_new_tail_without_pending() {
    let mut prefix = prefix("A B C D P Q R");
    let hypothesis = words("A C B D P X Q Y");
    let end = prefix.align(&hypothesis).unwrap();
    assert_eq!(&hypothesis[end..], words("Y"));
    assert_eq!(ownership(&prefix), 7);
}

/// An earlier reorder does not absorb an unrelated pending tail.
#[test]
fn an_earlier_reorder_does_not_absorb_an_unrelated_pending_tail() {
    let canonical = words("A B C D P Q R");
    let mut prefix = prefix("A B C D P Q R");
    let hypothesis = words("A C B D P X Q Y");
    let end = prefix
        .align_with_pending(&hypothesis, &words("Y"), &canonical)
        .unwrap();
    assert_eq!(&hypothesis[end..], words("Y"));
    assert_eq!(ownership(&prefix), 7);
}

/// Unrelated pending keeps the existing new tail boundary.
#[test]
fn unrelated_pending_keeps_the_existing_new_tail_boundary() {
    for (canonical_text, hypothesis_text) in
        [("A B C", "A X B Y"), ("A B C D P Q R", "A C B D P X Q Y")]
    {
        let canonical = words(canonical_text);
        let mut prefix = prefix(canonical_text);
        let hypothesis = words(hypothesis_text);
        let end = prefix
            .align_with_pending(&hypothesis, &words("unrelated"), &canonical)
            .unwrap();
        assert_eq!(&hypothesis[end..], words("Y"));
    }
}

/// Pending tail evidence does not skip a genuine repeated phrase.
#[test]
fn pending_tail_evidence_does_not_skip_a_genuine_repeated_phrase() {
    let canonical = words("Go now");
    let mut prefix = prefix("Go now");
    let hypothesis = words("Go now Go now today");
    let pending = words("Go now today");
    let end = prefix
        .align_with_pending(&hypothesis, &pending, &canonical)
        .unwrap();
    assert_eq!(&hypothesis[end..], pending);
    assert_eq!(ownership(&prefix), 2);
}

/// An empty confirmed prefix leaves all first speech available.
#[test]
fn an_empty_confirmed_prefix_leaves_all_first_speech_available() {
    let mut prefix = ConfirmedPrefix::default();
    assert_eq!(
        prefix.align_with_pending(&words("first speech"), &words("first"), &[]),
        Some(0)
    );
    assert!(prefix.anchors.is_empty());
}
