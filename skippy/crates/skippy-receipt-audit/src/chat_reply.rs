//! The reply a client received for the turn it audits.

/// What the node answered: its content and reasoning text, and the receipt
/// when the response carried it.
pub struct ChatReply {
    /// The model the request named.
    pub model: String,
    pub content: String,
    pub reasoning: String,
    /// The node says it stopped at the token limit (`finish_reason` is
    /// `length`). Its text may then end before the sampled tokens' text
    /// does: inside a character, or before a marker it held back.
    pub cut_short: bool,
}
