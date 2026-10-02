//! Bind admission guards to exact direct enclosing capability blocks.
use super::lexical::Token;
use super::{DynResult, balanced, mask, tokens};

fn opaque_control(lexed: &[Token<'_>], start: usize) -> DynResult<Option<usize>> {
    let word = lexed[start].text;
    if !["if", "for", "while", "switch", "catch", "else", "do", "try"].contains(&word) {
        return Ok(None);
    }
    let mut open = start + 1;
    if word == "else" && lexed.get(open).is_some_and(|token| token.text == "if") {
        return opaque_control(lexed, open);
    }
    if !["else", "do", "try"].contains(&word)
        && lexed.get(open).is_some_and(|token| token.text == "(")
    {
        open = balanced(lexed, open, "(", ")")?;
    }
    if lexed.get(open).is_none_or(|token| token.text != "{") {
        return Err("unbraced enclosing control can conditionally hide admission guard".into());
    }
    Ok(Some(balanced(lexed, open, "{", "}")?))
}

pub(super) fn direct_body<'a>(source: &'a str, header: &str) -> DynResult<&'a str> {
    let masked = mask(source);
    let lexed = tokens(&masked);
    let pattern = tokens(header);
    let mut at = 0;
    while at < lexed.len() {
        if ["return", "throw", "goto", "co_return"].contains(&lexed[at].text) {
            return Err("unconditional terminal precedes required admission guard".into());
        }
        if lexed[at..].len() >= pattern.len()
            && lexed[at..at + pattern.len()]
                .iter()
                .zip(&pattern)
                .all(|(actual, expected)| actual.text == expected.text)
        {
            let open = at + pattern.len();
            if lexed.get(open).is_none_or(|token| token.text != "{") {
                return Err("required capability/guard must have a braced direct body".into());
            }
            let end = balanced(&lexed, open, "{", "}")?;
            return Ok(&source[lexed[open].end..lexed[end - 1].start]);
        }
        // Every other body is opaque: expected guards cannot hide in a lambda,
        // an added conditional, a loop or another capability's branch.
        if let Some(end) = opaque_control(&lexed, at)? {
            at = end;
        } else if lexed[at].text == "{" {
            at = balanced(&lexed, at, "{", "}")?;
        } else {
            at += 1;
        }
    }
    Err("required direct capability/guard body not found".into())
}

pub(super) fn function_body(source: &str) -> DynResult<&str> {
    let masked = mask(source);
    let lexed = tokens(&masked);
    let signature = "enum skippy_status skippy_finish_model_open(";
    let start = super::find(&lexed, signature).ok_or("missing runtime admission function")?;
    let parameters = start + tokens(signature).len() - 1;
    let open = balanced(&lexed, parameters, "(", ")")?;
    if lexed.get(open).is_none_or(|token| token.text != "{") {
        return Err("runtime admission function must have direct body".into());
    }
    let end = balanced(&lexed, open, "{", "}")?;
    let terminator = super::find(&lexed[end..], "enum skippy_status skippy_model_open_impl(")
        .ok_or("missing runtime admission terminator")?
        + end;
    if lexed[end - 1].end > lexed[terminator].start {
        return Err("runtime admission function overlaps next function".into());
    }
    Ok(&source[lexed[open].end..lexed[end - 1].start])
}
