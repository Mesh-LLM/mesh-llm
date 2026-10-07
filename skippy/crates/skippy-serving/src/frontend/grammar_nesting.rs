//! Request-side bound on GBNF group nesting.
//!
//! The native grammar parser recurses once per nested `(` group with no depth
//! limit, so a grammar of about 100,000 nested groups overflowed the stack.
//! Request grammars that nest groups more than [`MAX_GRAMMAR_GROUP_DEPTH`]
//! levels deep are rejected here, before they reach native code.

/// Deepest group nesting a request grammar may use.
pub(super) const MAX_GRAMMAR_GROUP_DEPTH: usize = 256;

/// Returns an error when `grammar` nests `(` groups deeper than
/// [`MAX_GRAMMAR_GROUP_DEPTH`].
///
/// Parentheses inside string literals, character classes, `<token>`
/// references and `#` comments are not groups and are skipped.
pub(super) fn check_grammar_group_depth(grammar: &str) -> Result<(), String> {
    let bytes = grammar.as_bytes();
    let mut depth = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => index = skip_escaped_until(bytes, index + 1, b'"'),
            b'[' => index = skip_escaped_until(bytes, index + 1, b']'),
            b'<' => index = skip_until(bytes, index + 1, |byte| byte == b'>'),
            b'#' => index = skip_until(bytes, index + 1, |byte| byte == b'\n' || byte == b'\r'),
            b'(' => {
                depth += 1;
                if depth > MAX_GRAMMAR_GROUP_DEPTH {
                    return Err(format!(
                        "grammar nests groups more than {MAX_GRAMMAR_GROUP_DEPTH} levels deep"
                    ));
                }
            }
            b')' => depth = depth.saturating_sub(1),
            _ => {}
        }
        index += 1;
    }
    Ok(())
}

/// Returns the index of the unescaped `terminator` at or after `index`, or the
/// input length when it is missing.
fn skip_escaped_until(bytes: &[u8], mut index: usize, terminator: u8) -> usize {
    while index < bytes.len() && bytes[index] != terminator {
        index += if bytes[index] == b'\\' { 2 } else { 1 };
    }
    index.min(bytes.len())
}

fn skip_until(bytes: &[u8], index: usize, is_end: impl Fn(u8) -> bool) -> usize {
    bytes[index.min(bytes.len())..]
        .iter()
        .position(|&byte| is_end(byte))
        .map_or(bytes.len(), |offset| index + offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nested(depth: usize) -> String {
        format!("root ::= {}\"a\"{}", "(".repeat(depth), ")".repeat(depth))
    }

    #[test]
    fn accepts_groups_up_to_the_limit() {
        check_grammar_group_depth(&nested(MAX_GRAMMAR_GROUP_DEPTH)).unwrap();
    }

    #[test]
    fn rejects_groups_past_the_limit() {
        assert!(check_grammar_group_depth(&nested(MAX_GRAMMAR_GROUP_DEPTH + 1)).is_err());
        assert!(check_grammar_group_depth(&nested(100_000)).is_err());
    }

    #[test]
    fn ignores_parentheses_that_are_not_groups() {
        let open = "(".repeat(MAX_GRAMMAR_GROUP_DEPTH + 1);
        for grammar in [
            format!("root ::= \"{open}\""),
            format!("root ::= \"\\\"{open}\""),
            format!("root ::= [{open}]"),
            format!("root ::= [\\]{open}]"),
            format!("root ::= <{open}>"),
            format!("# {open}\nroot ::= \"a\""),
        ] {
            check_grammar_group_depth(&grammar).unwrap();
        }
    }

    #[test]
    fn counts_groups_after_skipped_spans() {
        let depth = MAX_GRAMMAR_GROUP_DEPTH + 1;
        let grammar = format!(
            "root ::= \"(\" [(] <(> # (\n {}\"a\"{}",
            "(".repeat(depth),
            ")".repeat(depth)
        );
        assert!(check_grammar_group_depth(&grammar).is_err());
    }
}
