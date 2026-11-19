use anyhow::bail;
use serde_json::json;

use super::*;
use crate::turn::TurnBinding;
use crate::wire::{RECEIPT_VERSION, StageCommitment};

const START: i32 = 10;
/// The end tag is two tokens.
const END: [i32; 2] = [20, 21];
/// A second end tag of one token.
const OTHER_END: i32 = 30;
/// The first bytes of a three-byte character, and its last byte alone.
const HALF: i32 = 40;
const REST: i32 = 41;
/// A token whose text starts with a space.
const SPACED: i32 = 50;

/// A toy tokenizer: a tag is its tokens, and a text of digits is one token
/// per word.
struct Toy;

impl Vocabulary for Toy {
    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            match word {
                "<think>" => tokens.push(START),
                "</think>" => tokens.extend(END),
                "<|end|>" => tokens.push(OTHER_END),
                "<nothing>" => {}
                "<broken>" => bail!("no such tag"),
                // a tokenizer that puts a space before its first token
                "<spaced>" => tokens.extend([SPACED, START]),
                word => tokens.push(word.parse()?),
            }
        }
        Ok(tokens)
    }

    fn piece(&self, token: i32) -> Result<Vec<u8>> {
        Ok(match token {
            HALF => vec![0xe2, 0x82],
            REST => vec![0xac],
            SPACED => b" x".to_vec(),
            99 => bail!("no such token"),
            _ => b"a".to_vec(),
        })
    }
}

/// The tokens the sampler forces over `sampled`, by index: one entry per
/// sampled token, `Some` where the sampler forced it.
fn forced(budget: i32, prefill: &[i32], sampled: &[i32]) -> Vec<Option<i32>> {
    let mut machine = Machine::new(
        vec![vec![START]],
        vec![END.to_vec(), vec![OTHER_END]],
        END.to_vec(),
        budget,
    );
    let mut whole = |token: i32| Ok(ends_a_character(&Toy.piece(token)?));
    for &token in prefill {
        machine.accept(token, &mut whole).unwrap();
    }
    sampled
        .iter()
        .map(|&token| {
            let forced = machine.forcing();
            machine.accept(token, &mut whole).unwrap();
            forced
        })
        .collect()
}

#[test]
fn the_sampler_counts_the_reasoning_and_then_forces_its_end() {
    // the start tag, two tokens of reasoning, then the end tag by force
    let turn = [1, START, 2, 3, END[0], END[1], 4, 5];
    assert_eq!(
        forced(2, &[], &turn),
        [
            None,
            None,
            None,
            None,
            Some(END[0]),
            Some(END[1]),
            None,
            None
        ]
    );
    // nothing is forced before the start tag, however long the turn
    assert!(forced(2, &[], &[1, 2, 3, 4, 5]).iter().all(Option::is_none));
    // a budget the reasoning does not use up forces nothing
    let short = [START, 1, END[0], END[1], 2, 3, 4];
    assert!(forced(5, &[], &short).iter().all(Option::is_none));
    // the other end tag ends the reasoning as well
    let other = [START, 1, OTHER_END, 2, 3, 4];
    assert!(forced(2, &[], &other).iter().all(Option::is_none));
}

#[test]
fn the_forced_tokens_are_due_whatever_the_node_sampled() {
    // the replay says what the sampler forced, not what the node took: a
    // node that sampled other tokens there is still owed the tag
    let turn = [START, 1, 7, 8, 9];
    assert_eq!(
        forced(1, &[], &turn),
        [None, None, Some(END[0]), Some(END[1]), None]
    );
}

#[test]
fn a_budget_of_no_tokens_forces_the_end_at_once() {
    let turn = [START, END[0], END[1], 1, 2];
    assert_eq!(
        forced(0, &[], &turn),
        [None, Some(END[0]), Some(END[1]), None, None]
    );
}

#[test]
fn a_second_start_tag_counts_again() {
    let turn = [
        START, 1, END[0], END[1], // ended by the model, inside the budget
        2, START, 3, 4, 6, END[0], END[1], // ended by force
        5,
    ];
    let mut expected = vec![None; turn.len()];
    expected[9] = Some(END[0]);
    expected[10] = Some(END[1]);
    assert_eq!(forced(3, &[], &turn), expected);
    // the first token of a tag the model began counts against the budget:
    // when it uses the budget up, the sampler forces the whole tag
    assert_eq!(
        forced(2, &[], &[START, 1, END[0], END[0], END[1], 2]),
        [None, None, None, Some(END[0]), Some(END[1]), None]
    );
    // and after a forced end
    let turn = [START, 1, END[0], END[1], START, 2, END[0], END[1], 3];
    assert_eq!(
        forced(1, &[], &turn),
        [
            None,
            None,
            Some(END[0]),
            Some(END[1]),
            None,
            None,
            Some(END[0]),
            Some(END[1]),
            None
        ]
    );
}

#[test]
fn a_budget_that_runs_out_inside_a_character_waits_for_its_end() {
    // the budget's last token is half of a character: the sampler lets
    // tokens through until one ends a character, and forces after it. A
    // token that only continues the character does not end one for the
    // sampler, which reads each token's text alone.
    let turn = [START, 1, HALF, REST, 5, END[0], END[1], 2];
    assert_eq!(
        forced(2, &[], &turn),
        [
            None,
            None,
            None,
            None,
            None,
            Some(END[0]),
            Some(END[1]),
            None
        ]
    );
    // an end tag while it waits ends the reasoning with no force
    let ended = [START, 1, HALF, OTHER_END, 2, 3];
    assert!(forced(2, &[], &ended).iter().all(Option::is_none));
    assert!(ends_a_character(b""));
    assert!(ends_a_character("a\u{20ac}".as_bytes()));
    assert!(!ends_a_character(&[b'a', 0xe2, 0x82]));
    assert!(!ends_a_character(&[0xf0, 0x9f, 0x98]));
    assert!(!ends_a_character(&[0x80, 0x80, 0x80, 0x80, 0x80]));
}

#[test]
fn a_prefill_that_opens_the_reasoning_starts_the_count() {
    // the template already wrote the start tag: the turn's first tokens
    // are reasoning
    assert_eq!(
        forced(2, &[1, START], &[2, 3, END[0], END[1], 4]),
        [None, None, Some(END[0]), Some(END[1]), None]
    );
    // a prefill that opens and closes it leaves nothing to force, also
    // with no budget: the sampler reads the end tag, and forces none of it
    let closed = [START, END[0], END[1]];
    assert!(forced(0, &closed, &[1, 2, 3]).iter().all(Option::is_none));
    assert!(forced(2, &closed, &[1, 2, 3]).iter().all(Option::is_none));
    // a prefill that opens it with no budget: the tag is forced first
    assert_eq!(
        forced(0, &[START], &[END[0], END[1], 1]),
        [Some(END[0]), Some(END[1]), None]
    );
}

fn statement(budget: i64, reasoning: serde_json::Value) -> String {
    turn::sampled(&json!({
        "temperature": 0.7,
        "reasoning_budget": budget,
        "reasoning": reasoning,
    }))
}

fn tags() -> serde_json::Value {
    json!({ "start": "<think>", "ends": ["</think>", "<|end|>"], "prefill": "1 <think>" })
}

/// A turn whose prompt ends at position 4 and that sampled `sampled`.
fn receipt(sampler: &str, sampled: &[i32]) -> Receipt {
    let (last, tokens) = sampled.split_last().expect("a sampled token");
    Receipt {
        version: RECEIPT_VERSION,
        receipt_id: "r".into(),
        model: "m".into(),
        position_start: 2,
        position_end: 5 + tokens.len() as u32,
        tokens: [&[0, 0, 0], tokens].concat(),
        generation_start: 5,
        final_token: Some(*last),
        stages: vec![StageCommitment {
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            root: String::new(),
            leaf_count: 0,
            backend: "CPU".into(),
            cache_types: Vec::new(),
            prefix: None,
            ubatches: Vec::new(),
            recurrent_state: None,
        }],
        binding: Some(TurnBinding {
            client_key: String::new(),
            parent: None,
            runtime: String::new(),
            sampler: sampler.to_owned(),
            node_key: String::new(),
            turn_root: String::new(),
            signature: String::new(),
        }),
    }
}

#[test]
fn a_replay_names_the_positions_the_sampler_forced() {
    // the prefill opened the reasoning; the budget is two tokens
    let turn = receipt(&statement(2, tags()), &[2, 3, END[0], END[1], 4]);
    let forcing = replay(&turn, Some(&Toy));
    let Forcing::Replayed(replayed) = &forcing else {
        panic!("{forcing:?}");
    };
    assert_eq!(replayed.statement.budget, 2);
    assert_eq!(replayed.statement.ends, ["</think>", "<|end|>"]);
    // the token sampled from position 6 sits at position 7: the third
    // sampled token, the first the sampler forced
    let forced: Vec<Option<i32>> = (4..9).map(|p| forcing.forced_at(p)).collect();
    assert_eq!(forced, [None, None, Some(END[0]), Some(END[1]), None]);
    assert!(forcing.unjudged().is_none());
    // tokens that are not the turn's own are not the sampler's
    assert_eq!(forcing.forced_at(3), None);
}

#[test]
fn a_statement_without_a_sampler_forces_nothing() {
    // greedy, no budget, or a budget and no tags: the runtime built none
    let free = [
        turn::GREEDY.to_owned(),
        turn::sampled(&json!({ "temperature": 0.7 })),
        turn::sampled(&json!({ "reasoning_budget": 64, "reasoning": null })),
        turn::sampled(&json!({ "reasoning_budget": 64 })),
    ];
    for sampler in free {
        let turn = receipt(&sampler, &[START, 1, 2, 3]);
        assert_eq!(replay(&turn, Some(&Toy)), Forcing::Free, "{sampler}");
        assert_eq!(replay(&turn, None), Forcing::Free, "{sampler}");
    }
    let mut unbound = receipt(turn::GREEDY, &[1]);
    unbound.binding = None;
    assert_eq!(replay(&unbound, Some(&Toy)), Forcing::Free);
    // a tag that is no token builds no sampler in the runtime either
    let no_tokens = json!({ "start": "<nothing>", "ends": ["</think>"] });
    let turn = receipt(&statement(0, no_tokens), &[START, 1, 2]);
    let forcing = replay(&turn, Some(&Toy));
    assert!(matches!(forcing, Forcing::Replayed(_)), "{forcing:?}");
    assert!((0..12).all(|p| forcing.forced_at(p).is_none()));
}

#[test]
fn the_prefill_drops_a_space_its_tokenizer_put_first() {
    // the tokenizer puts a spaced token before the tag, and the text
    // starts with none: the runtime leaves the token out
    let spaced = json!({ "start": "<think>", "ends": ["</think>"], "prefill": "<spaced>" });
    let turn = receipt(&statement(1, spaced), &[1, END[0], END[1], 2]);
    let forcing = replay(&turn, Some(&Toy));
    let forced: Vec<Option<i32>> = (4..8).map(|p| forcing.forced_at(p)).collect();
    assert_eq!(forced, [None, Some(END[0]), Some(END[1]), None]);
    assert_eq!(prefill_tokens("<spaced>", &Toy).unwrap(), [START]);
    // a text that starts with a space keeps it
    assert_eq!(prefill_tokens(" <spaced>", &Toy).unwrap(), [SPACED, START]);
    assert!(prefill_tokens("", &Toy).unwrap().is_empty());
}

#[test]
fn a_sampler_the_check_cannot_replay_leaves_the_tokens_unjudged() {
    let turn = receipt(&statement(2, tags()), &[2, 3, 4]);
    // no tokenizer
    let reason = replay(&turn, None);
    assert!(
        reason
            .unjudged()
            .is_some_and(|r| r.contains("no tokenizer")),
        "{reason:?}"
    );
    // a tokenizer that fails is the client's own
    let broken = json!({ "start": "<broken>", "ends": ["</think>"] });
    let forcing = replay(&receipt(&statement(2, broken), &[2, 3]), Some(&Toy));
    assert!(
        forcing
            .unjudged()
            .is_some_and(|r| r.contains("no such tag")),
        "{forcing:?}"
    );
    let unknown = receipt(&statement(1, tags()), &[99, 1, 2]);
    // token 99 uses the budget up, and its text cannot be read
    let forcing = replay(&unknown, Some(&Toy));
    assert!(
        forcing
            .unjudged()
            .is_some_and(|r| r.contains("no such token")),
        "{forcing:?}"
    );
    assert_eq!(forcing.forced_at(5), None);
}

#[test]
fn a_statement_outside_the_limits_is_refused() {
    let long = "x".repeat(MAX_TAG_BYTES + 1);
    let many: Vec<&str> = vec!["</think>"; MAX_END_TAGS + 1];
    let refused = [
        (
            json!({ "start": "", "ends": ["</think>"] }),
            "without a start tag",
        ),
        (
            json!({ "ends": ["</think>"] }),
            "start tag of the reasoning is not text",
        ),
        (json!({ "start": 7, "ends": ["</think>"] }), "is not text"),
        (
            json!({ "start": long, "ends": ["</think>"] }),
            "more than 256",
        ),
        (json!({ "start": "<think>" }), "no end tags"),
        (json!({ "start": "<think>", "ends": [] }), "0 end tags"),
        (json!({ "start": "<think>", "ends": many }), "9 end tags"),
        (json!({ "start": "<think>", "ends": [""] }), "empty end tag"),
        (
            json!({ "start": "<think>", "ends": [long] }),
            "more than 256",
        ),
        (
            json!({ "start": "<think>", "ends": [["a"]] }),
            "is not text",
        ),
        (
            json!({ "start": "<think>", "ends": ["</think>"], "prefill": "y".repeat(MAX_PREFILL_BYTES + 1) }),
            "more than 1024",
        ),
        (json!("<think>"), "is not text"),
    ];
    for (reasoning, why) in refused {
        let sampler = statement(4, reasoning.clone());
        let error = BudgetStatement::from_sampler(&sampler).unwrap_err();
        assert!(error.contains(why), "{reasoning}: {error}");
        let forcing = replay(&receipt(&sampler, &[1, 2]), Some(&Toy));
        assert_eq!(forcing.unjudged(), Some(error.as_str()));
    }
    // tags without a budget name no sampler the runtime builds
    for budget in [json!(null), json!(-1)] {
        let sampler = turn::sampled(&json!({ "reasoning_budget": budget, "reasoning": tags() }));
        let error = BudgetStatement::from_sampler(&sampler).unwrap_err();
        assert!(error.contains("no reasoning budget"), "{error}");
    }
    // the limits themselves pass, and a budget past any turn is kept
    let edge = json!({
        "start": "x".repeat(MAX_TAG_BYTES),
        "ends": vec!["</think>"; MAX_END_TAGS],
        "prefill": "y".repeat(MAX_PREFILL_BYTES),
    });
    let stated = BudgetStatement::from_sampler(&statement(i64::MAX, edge))
        .unwrap()
        .unwrap();
    assert_eq!(stated.budget, i32::MAX);
    assert_eq!(stated.ends.len(), MAX_END_TAGS);
}
