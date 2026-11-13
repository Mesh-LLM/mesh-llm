//! The part of a chat's template metadata that native sampling is given.
//!
//! Template metadata can hold a large serialized PEG response parser, which
//! native sampling never reads. Native sampling reads the grammar fields,
//! and the fields a reasoning budget is enforced from: the tags that open
//! and close a reasoning block, and the text the template placed after the
//! prompt, from which the budget learns whether that block is open.
//!
//! So a chat without a grammar is given a small payload instead of the
//! whole metadata:
//!
//! - [`EMPTY_CHAT_GRAMMAR_METADATA`] when no budget can apply. The native
//!   side reuses a plain chat's sampler for exactly this text.
//! - The reasoning fields when the chat has a finite budget and its
//!   reasoning block is not already closed, so the native side builds the
//!   budget sampler as it does for a chat with a grammar.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// What a chat without a grammar or an enforceable budget is given. The
/// native side compares against this exact text to reuse a sampler.
pub(crate) const EMPTY_CHAT_GRAMMAR_METADATA: &str = r#"{"grammar":""}"#;

/// The metadata fields a reasoning budget is enforced from.
#[derive(Debug, Default, Deserialize, Serialize)]
struct ReasoningFields {
    #[serde(default)]
    generation_prompt: String,
    /// Always empty here: the fields are only sent for a chat without one.
    #[serde(default)]
    grammar: String,
    #[serde(default)]
    supports_thinking: bool,
    #[serde(default)]
    thinking_end_tags: Vec<String>,
    #[serde(default)]
    thinking_start_tag: String,
}

impl ReasoningFields {
    /// Whether the template names both ends of a reasoning block. Without
    /// them the native side builds no budget sampler.
    fn names_a_block(&self) -> bool {
        !self.thinking_start_tag.is_empty() && self.thinking_end_tags.iter().any(|t| !t.is_empty())
    }

    /// Whether the text after the prompt ends with a closed reasoning
    /// block: the template writes that for a request with reasoning off.
    /// The budget bounds the reasoning a request asked for, so such a chat
    /// needs no budget sampler and keeps the plain sampler's fast paths.
    ///
    /// A block closed before more text does not count. A continued message
    /// can end with its text after its reasoning, and a model can open
    /// another reasoning block after text, which the budget must bound.
    fn block_is_closed(&self) -> bool {
        let Some(opened) = self.generation_prompt.rfind(&self.thinking_start_tag) else {
            return false;
        };
        let after = &self.generation_prompt[opened + self.thinking_start_tag.len()..];
        self.thinking_end_tags.iter().any(|end| {
            !end.is_empty()
                && after
                    .rfind(end.as_str())
                    .is_some_and(|closed| after[closed + end.len()..].trim().is_empty())
        })
    }
}

/// The metadata to give native sampling for a chat whose template metadata
/// is `metadata_json` and whose reasoning budget is `budget_tokens`:
/// negative for no limit, `None` when the chat passes no sampling
/// configuration, in which case the native side enforces no budget.
///
/// Whoever states what a turn's sampler was built from, as a receipt
/// does, must read the metadata through this.
pub fn chat_sampling_metadata_for_native(
    metadata_json: &str,
    budget_tokens: Option<i32>,
) -> Cow<'_, str> {
    // The renderer emits compact JSON, so an empty top-level grammar is
    // this byte sequence. Escaped JSON nested inside `chat_parser` cannot
    // contain it unescaped.
    if !metadata_json.contains(r#""grammar":"""#) {
        return Cow::Borrowed(metadata_json);
    }
    let empty = Cow::Borrowed(EMPTY_CHAT_GRAMMAR_METADATA);
    if !budget_tokens.is_some_and(|tokens| tokens >= 0) {
        return empty;
    }
    // Reading these few fields skips the parser without building it.
    let Ok(mut fields) = serde_json::from_str::<ReasoningFields>(metadata_json) else {
        return empty;
    };
    if !fields.names_a_block() || fields.block_is_closed() {
        return empty;
    }
    fields.grammar.clear();
    serde_json::to_string(&fields).map_or(empty, Cow::Owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Template metadata of a thinking model, with a parser that stands
    /// for the large one.
    fn metadata(generation_prompt: &str) -> String {
        serde_json::json!({
            "additional_stops": [],
            "chat_format": 1,
            "chat_parser": "a large serialized parser with \"grammar\":\"\" inside",
            "generation_prompt": generation_prompt,
            "grammar": "",
            "grammar_lazy": false,
            "grammar_triggers": [],
            "supports_thinking": true,
            "thinking_end_tags": ["</think>"],
            "thinking_start_tag": "<think>",
        })
        .to_string()
    }

    const OPEN: &str = "<|im_start|>assistant\n<think>\n";
    const CLOSED: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

    #[test]
    fn empty_chat_grammar_uses_minimal_native_metadata() {
        let metadata = r#"{"chat_parser":"large","grammar":"","grammar_triggers":[]}"#;
        for budget in [None, Some(-1), Some(0), Some(512)] {
            assert_eq!(
                chat_sampling_metadata_for_native(metadata, budget),
                EMPTY_CHAT_GRAMMAR_METADATA,
                "{budget:?}"
            );
        }
    }

    #[test]
    fn a_chat_with_a_grammar_is_given_its_whole_metadata() {
        let grammar = r#"{"grammar":"root ::= \"ok\"","thinking_start_tag":"<think>"}"#;
        for budget in [None, Some(-1), Some(16)] {
            assert_eq!(chat_sampling_metadata_for_native(grammar, budget), grammar);
        }
    }

    #[test]
    fn a_finite_budget_is_given_the_reasoning_fields_without_the_parser() {
        for prompt in [OPEN, "<|im_start|>assistant\n"] {
            for budget in [0, 16] {
                let metadata = metadata(prompt);
                let given = chat_sampling_metadata_for_native(&metadata, Some(budget));
                let given: serde_json::Value = serde_json::from_str(&given).unwrap();
                assert_eq!(
                    given,
                    serde_json::json!({
                        "generation_prompt": prompt,
                        "grammar": "",
                        "supports_thinking": true,
                        "thinking_end_tags": ["</think>"],
                        "thinking_start_tag": "<think>",
                    }),
                    "{prompt:?} with a budget of {budget}"
                );
            }
        }
    }

    #[test]
    fn a_chat_without_an_enforceable_budget_keeps_the_reusable_payload() {
        let open = metadata(OPEN);
        // no limit, and no sampling configuration: the native side would
        // build no budget sampler from the fields
        for budget in [None, Some(-1)] {
            assert_eq!(
                chat_sampling_metadata_for_native(&open, budget),
                EMPTY_CHAT_GRAMMAR_METADATA,
                "{budget:?}"
            );
        }
        // reasoning is off: the template closed the block
        assert_eq!(
            chat_sampling_metadata_for_native(&metadata(CLOSED), Some(16)),
            EMPTY_CHAT_GRAMMAR_METADATA
        );
    }

    #[test]
    fn a_block_opened_again_after_a_closed_one_is_open() {
        let reopened = metadata(&format!("{CLOSED}<think>\n"));
        let given = chat_sampling_metadata_for_native(&reopened, Some(16));
        assert!(given.contains("thinking_start_tag"), "{given}");
    }

    #[test]
    fn a_block_closed_before_more_text_keeps_the_budget() {
        // a continued message whose text follows its reasoning, as an
        // Inkling template writes it: the model can reason again after it
        let metadata = serde_json::json!({
            "chat_parser": "large",
            "generation_prompt": "<|message_model|><|content_thinking|>plan\
                                  <|end_message|><|content_text|>Partial answer",
            "grammar": "",
            "supports_thinking": true,
            "thinking_end_tags": ["<|end_message|>"],
            "thinking_start_tag": "<|content_thinking|>",
        })
        .to_string();
        let given = chat_sampling_metadata_for_native(&metadata, Some(16));
        assert!(given.contains("thinking_start_tag"), "{given}");
        assert!(!given.contains("chat_parser"), "{given}");
    }

    #[test]
    fn a_template_without_reasoning_tags_keeps_the_reusable_payload() {
        // a finite budget with `supports_thinking` and no tags is an error
        // on the native side, so such a chat must not be given the fields
        for metadata in [
            r#"{"generation_prompt":"x","grammar":"","supports_thinking":true}"#,
            r#"{"grammar":"","supports_thinking":true,"thinking_end_tags":[],"thinking_start_tag":"<think>"}"#,
            r#"{"grammar":"","thinking_end_tags":["</think>"],"thinking_start_tag":""}"#,
            r#"{"grammar":"","thinking_end_tags":[""],"thinking_start_tag":"<think>"}"#,
        ] {
            assert_eq!(
                chat_sampling_metadata_for_native(metadata, Some(16)),
                EMPTY_CHAT_GRAMMAR_METADATA,
                "{metadata}"
            );
        }
    }

    #[test]
    fn metadata_that_does_not_parse_keeps_the_reusable_payload() {
        for metadata in [
            r#"{"grammar":"","thinking_end_tags":"</think>"}"#,
            r#"{"grammar":"" "#,
        ] {
            assert_eq!(
                chat_sampling_metadata_for_native(metadata, Some(16)),
                EMPTY_CHAT_GRAMMAR_METADATA,
                "{metadata}"
            );
        }
    }
}
