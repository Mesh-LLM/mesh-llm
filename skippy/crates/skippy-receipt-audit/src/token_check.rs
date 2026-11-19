//! The response and the prompt a client holds, against the tokens a
//! receipt names.
//!
//! The response must be exactly the text of the tokens the receipt says
//! were sampled: the reasoning and the content together, up to whitespace
//! and the reasoning markers the output parser removes. A response that
//! drops or adds text fails, and so does an empty one for tokens that make
//! text.
//!
//! A response the node says it cut at the token limit may end before the
//! tokens' text does: the node holds back a character whose last token it
//! did not sample, and a reasoning marker it had only begun. Such a
//! response must be the start of the tokens' text, and what it leaves out
//! must be such an end ([`held_back`]). The node itself says that it cut
//! the response, so the rule lets it leave out nothing else.
//!
//! The prompt must be part of the text of the tokens before them, because
//! the chat template adds its own text around it. Only the message a
//! client passes is compared, usually its last user message: the system
//! prompt, the history and any text the template adds are not. A turn that
//! continued a cached prefix receipts the end of the prompt alone. The rest
//! is in the prefix, whose tokens the node opens with its answer, bound to
//! the digest the receipt commits ([`skippy_receipts::binding`]).
//!
//! A short message could be found anywhere in a long prefix, in an earlier
//! turn. So the message must end in the turn's own tokens ([`PromptMatch`]).
//! A turn whose prompt the prefix held all but its last token of, as when a
//! client sends a request again, is the exception: there the message must
//! be the last one in the prefix, with no more than the chat template's
//! closing text after it ([`TEMPLATE_TAIL_CHARS`]).
//!
//! Neither comparison is part of the evidence: the node signs neither the
//! response nor the request.

use anyhow::Result;
use skippy_receipts::wire::Receipt;

use crate::chat_reply::ChatReply;
use crate::claimed_model::ClaimedModel;
use crate::output::TokenOutput;

/// The most characters the sampled tokens' text may hold past the end of a
/// response cut at the token limit.
const CUT_TAIL_CHARS: usize = 16;

/// The most characters, whitespace and reasoning markers left out, that a
/// chat template writes after the last message: the end of the message,
/// the start of the reply and an empty reasoning block.
const TEMPLATE_TAIL_CHARS: usize = 128;

/// Where the client's message was found among the tokens before the
/// sampled ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PromptMatch {
    /// It ends in the turn's own tokens.
    Turn,
    /// It is the end of the cached prefix, followed only by the template's
    /// closing text.
    PrefixEnd,
    /// Neither.
    Missing,
}

impl PromptMatch {
    /// What was compared, in words, for a report.
    pub(crate) fn describe(self) -> &'static str {
        match self {
            Self::Turn => "the message ends in the turn's own tokens",
            Self::PrefixEnd => {
                "the message ends the cached prefix; the turn's own tokens hold only the template's text"
            }
            Self::Missing => {
                "the message does not end in the turn's own tokens or the cached prefix"
            }
        }
    }
}

/// Where `message` ends in the text of `prefix` and then `turn`, the prompt
/// tokens of the cached prefix and of the turn.
fn find_prompt(prefix: &str, turn: &str, message: &str) -> PromptMatch {
    let (prefix, turn, message) = (comparable(prefix), comparable(turn), comparable(message));
    let all = format!("{prefix}{turn}");
    let Some(start) = all.rfind(&message) else {
        return PromptMatch::Missing;
    };
    let end = start + message.len();
    if end > prefix.len() {
        PromptMatch::Turn
    } else if all.len() - end <= TEMPLATE_TAIL_CHARS {
        PromptMatch::PrefixEnd
    } else {
        PromptMatch::Missing
    }
}

/// Compares `reply` and `prompt` with the receipt's tokens and
/// `prefix_tokens`, the tokens of the cached prefix the turn continued, as
/// the node opened them. An error is the client's own model failing to turn
/// tokens into text.
pub(crate) fn check_tokens(
    model: &dyn ClaimedModel,
    receipt: &Receipt,
    prefix_tokens: &[i32],
    prompt: &str,
    reply: &ChatReply,
) -> Result<TokenOutput> {
    let split = receipt
        .generation_start
        .saturating_sub(receipt.position_start) as usize;
    let mut sampled = receipt.sampled_tokens();
    while let Some(&last) = sampled.last() {
        if !model.is_end_of_generation(last)? {
            break;
        }
        sampled.pop();
    }
    let generated = model.detokenize(&sampled)?;
    let answered = format!("{}{}", reply.reasoning, reply.content);
    let turn_tokens = receipt.tokens.get(..split).unwrap_or(&receipt.tokens);
    // every position before the turn must be there to be compared
    let found = if prefix_tokens.len() == receipt.position_start as usize {
        let prefix_text = model.detokenize(prefix_tokens)?;
        find_prompt(&prefix_text, &model.detokenize(turn_tokens)?, prompt)
    } else {
        PromptMatch::Missing
    };
    Ok(TokenOutput {
        response_ok: response_matches(&generated, &answered, reply.cut_short),
        prompt_ok: found != PromptMatch::Missing,
        prompt_compared: found.describe().to_owned(),
        generated,
    })
}

/// Text compared without whitespace and reasoning markers.
fn comparable(text: &str) -> String {
    text.replace("<think>", "")
        .replace("</think>", "")
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

fn response_matches(generated: &str, answered: &str, cut_short: bool) -> bool {
    let (generated, answered) = (comparable(generated), comparable(answered));
    if !cut_short {
        return generated == answered;
    }
    generated
        .strip_prefix(answered.as_str())
        .is_some_and(held_back)
}

/// Whether `tail`, the end of the sampled tokens' text, is what a node
/// holds back from a response it cut: nothing, a character whose bytes it
/// did not finish (which reads as replacement characters), or the start of
/// a marker.
fn held_back(tail: &str) -> bool {
    let unfinished = tail.chars().all(|c| c == char::REPLACEMENT_CHARACTER);
    let marker = tail.starts_with('<');
    (unfinished || marker) && tail.chars().count() <= CUT_TAIL_CHARS
}

#[cfg(test)]
mod tests {
    use skippy_receipts::glue::GlueEvaluator;
    use skippy_receipts::toy_turn::Turn;

    use super::*;
    use crate::claimed_model::CacheTypes;

    #[test]
    fn the_response_must_be_all_of_the_generated_text() {
        let whole = |generated, answered| response_matches(generated, answered, false);
        assert!(whole("<think>\nhm\n</think>\n\nParis.", "hmParis."));
        assert!(whole("Paris is big.", " Paris is  big. "));
        // dropped, added or empty text fails
        assert!(!whole("Paris is big.", "Paris"));
        assert!(!whole("Paris.", "Paris. Buy now."));
        assert!(!whole("Paris.", ""));
        assert!(whole("", ""));
        // a character the node did not finish is not text it left out
        assert!(!whole("Paris est gai\u{fffd}", "Paris est gai"));
    }

    #[test]
    fn a_response_cut_at_the_token_limit_is_the_start_of_the_generated_text() {
        let cut = |generated, answered| response_matches(generated, answered, true);
        // the last token was half of a character, which the node held back
        assert!(cut("Paris est gai\u{fffd}", "Paris est gai"));
        // or the start of the marker that ends the reasoning
        assert!(cut("<think>\nhm </thi", "hm"));
        assert!(cut("Paris.", "Paris."));
        // it is still the node's text: none added, none changed, and
        // nothing else left out
        assert!(!cut("Paris is big.", "Paris is"));
        assert!(!cut("Paris.", "Paris. Buy now."));
        assert!(!cut("Paris is big.", "Paris is bad"));
        assert!(!cut("Paris is the capital of France, and big.", "Paris is"));
        assert!(!cut("Paris is big.", ""));
    }

    /// A model whose token `n` is the text `<n>`.
    struct Numbered;

    impl ClaimedModel for Numbered {
        fn with_layer(
            &self,
            _: u32,
            _: CacheTypes,
            _: &mut dyn FnMut(&dyn GlueEvaluator) -> Result<()>,
        ) -> Result<()> {
            anyhow::bail!("no layers")
        }

        fn detokenize(&self, tokens: &[i32]) -> Result<String> {
            Ok(tokens.iter().map(|t| format!("<{t}>")).collect())
        }

        fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
            // a text of tokens, each written `<n>`
            text.split_terminator('>')
                .map(|token| Ok(token.trim_start_matches('<').parse()?))
                .collect()
        }

        fn token_piece(&self, token: i32) -> Result<Vec<u8>> {
            Ok(format!("<{token}>").into_bytes())
        }

        fn is_end_of_generation(&self, _: i32) -> Result<bool> {
            Ok(false)
        }

        fn runtime(&self) -> String {
            String::new()
        }
    }

    fn reply(receipt: &Receipt) -> ChatReply {
        ChatReply {
            model: String::new(),
            content: Numbered.detokenize(&receipt.sampled_tokens()).unwrap(),
            reasoning: String::new(),
            cut_short: false,
        }
    }

    #[test]
    fn the_prompt_is_checked_for_a_turn_from_position_zero() {
        let receipt = Turn::new().receipt;
        let reply = reply(&receipt);
        let checked = check_tokens(&Numbered, &receipt, &[], "<1><2><3>", &reply).unwrap();
        assert!(checked.response_ok);
        assert!(checked.prompt_ok);
        let other = check_tokens(&Numbered, &receipt, &[], "<7><7>", &reply).unwrap();
        assert!(!other.prompt_ok);
    }

    #[test]
    fn the_prompt_of_a_turn_that_continued_a_cached_prefix_spans_the_prefix() {
        // The receipt holds the end of the prompt alone: tokens 1, 2 and 3.
        // The node opens the 40 tokens before them with its answer.
        let receipt = Turn::starting_at(40).receipt;
        let reply = reply(&receipt);
        let prefix: Vec<i32> = (0..40).map(|p| 1 + p % 3).collect();
        let check = |prefix: &[i32], prompt: &str| {
            let checked = check_tokens(&Numbered, &receipt, prefix, prompt, &reply).unwrap();
            assert!(checked.response_ok);
            checked.prompt_ok
        };
        // the end of the prefix and the start of the turn, as one text
        assert!(check(&prefix, "<2><3><1><1><2><3>"));
        assert!(check(&prefix, "<1><2><3>"));
        assert!(!check(&prefix, "a prompt the node never saw"));
        // a prompt the node cut the start of is not in the tokens
        assert!(!check(&prefix, "<9><1><2><3>"));
        // without the tokens of the prefix the prompt cannot be found
        assert!(!check(&[], "<1><2><3>"));
        assert!(!check(&prefix[..39], "<1><2><3>"));
    }

    #[test]
    fn a_short_message_must_end_in_the_turn_not_anywhere_in_its_prefix() {
        let receipt = Turn::starting_at(40).receipt;
        let reply = reply(&receipt);
        let found = |prefix: &[i32], prompt: &str| {
            let checked = check_tokens(&Numbered, &receipt, prefix, prompt, &reply).unwrap();
            (checked.prompt_ok, checked.prompt_compared)
        };
        // an earlier turn said "<9>": a message found there is not the one
        // this turn read
        let mut prefix: Vec<i32> = (0..40).map(|p| 1000 + p).collect();
        prefix[2] = 9;
        let (ok, compared) = found(&prefix, "<9>");
        assert!(!ok, "{compared}");
        assert_eq!(compared, PromptMatch::Missing.describe());
        // the message that ends in the turn's own tokens is
        let (ok, compared) = found(&prefix, "<1039><1><2>");
        assert!(ok);
        assert_eq!(compared, PromptMatch::Turn.describe());
        // a request sent again: the prefix holds the message, and the turn
        // only the template's closing text
        prefix[39] = 9;
        let (ok, compared) = found(&prefix, "<9>");
        assert!(ok);
        assert_eq!(compared, PromptMatch::PrefixEnd.describe());
    }
}
