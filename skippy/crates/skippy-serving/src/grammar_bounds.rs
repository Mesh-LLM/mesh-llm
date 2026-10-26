//! Bounds on GBNF grammars before llama.cpp initializes them.
//!
//! Native grammar initialization has no limits of its own: deep group
//! nesting and long leftmost rule chains overflow its stack, and repetitions
//! can expand a small grammar until memory runs out. Grammars reach it from
//! request fields, from the chat template, which renders JSON schemas and
//! tool definitions into a grammar, and from the chat metadata a peer stage
//! sends. Each of those is checked here first.

mod expansion;
mod nesting;

/// Rejects a grammar that native initialization could not handle safely.
pub(crate) fn check_grammar(grammar: &str) -> Result<(), String> {
    // The expansion check parses groups recursively, so bound their depth
    // first.
    nesting::check_grammar_group_depth(grammar)?;
    expansion::check_grammar_expansion(grammar)
}

/// Checks the grammar native sampling builds from chat sampling metadata.
///
/// The JSON schema converter emits whatever bounds a schema gives, so
/// `minItems` above `maxItems` renders as an inverted repetition, and a peer
/// stage can send any metadata it likes.
pub(crate) fn check_chat_metadata_grammar(metadata_json: &str) -> Result<(), String> {
    let metadata: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|error| format!("chat sampling metadata is not valid JSON: {error}"))?;
    match metadata.get("grammar") {
        None | Some(serde_json::Value::Null) => Ok(()),
        Some(serde_json::Value::String(grammar)) if grammar.is_empty() => Ok(()),
        Some(serde_json::Value::String(grammar)) => check_grammar(grammar),
        Some(_) => Err("chat sampling metadata grammar must be a string".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_the_grammar_in_chat_metadata() {
        check_chat_metadata_grammar(r#"{"grammar":"","chat_format":1}"#).unwrap();
        check_chat_metadata_grammar(r#"{"chat_format":1}"#).unwrap();
        check_chat_metadata_grammar(r#"{"grammar":"root ::= \"a\""}"#).unwrap();
        // What the native converter renders for minItems 5, maxItems 3.
        let inverted = serde_json::json!({
            "grammar": "root ::= \"[\" \"1\" (\",\" \"1\"){4,2} \"]\"\n"
        });
        assert!(check_chat_metadata_grammar(&inverted.to_string()).is_err());
        assert!(check_chat_metadata_grammar(r#"{"grammar":7}"#).is_err());
        assert!(check_chat_metadata_grammar("not json").is_err());
    }
}
