//! Bounded tokens for static C++ admission checks after comments/literals masking.
use crate::command::DynResult;

#[derive(Debug)]
pub(super) struct Token<'a> {
    pub(super) text: &'a str,
    pub(super) start: usize,
    pub(super) end: usize,
}

pub(super) fn tokens(source: &str) -> Vec<Token<'_>> {
    let mut result = Vec::new();
    let mut at = 0;
    while at < source.len() {
        let first = source[at..].chars().next().expect("valid string boundary");
        if first.is_whitespace() {
            at += first.len_utf8();
            continue;
        }
        let start = at;
        at += first.len_utf8();
        if first == '"' {
            while at < source.len() {
                let char = source[at..].chars().next().expect("valid literal boundary");
                at += char.len_utf8();
                if char == '\\' && at < source.len() {
                    at += source[at..].chars().next().unwrap().len_utf8();
                } else if char == '"' {
                    break;
                }
            }
        } else if first.is_alphanumeric() || first == '_' {
            while let Some(char) = source[at..].chars().next() {
                if !char.is_alphanumeric() && char != '_' {
                    break;
                }
                at += char.len_utf8();
            }
        }
        result.push(Token {
            text: &source[start..at],
            start,
            end: at,
        });
    }
    result
}
pub(super) fn find(tokens: &[Token<'_>], pattern: &str) -> Option<usize> {
    let wanted = self::tokens(pattern);
    tokens.windows(wanted.len()).position(|window| {
        window
            .iter()
            .zip(&wanted)
            .all(|(left, right)| left.text == right.text)
    })
}
pub(super) fn balanced(
    tokens: &[Token<'_>],
    start: usize,
    open: &str,
    close: &str,
) -> DynResult<usize> {
    let mut depth = 0_usize;
    for (at, token) in tokens.iter().enumerate().skip(start) {
        if token.text == open {
            depth += 1;
        } else if token.text == close {
            depth = depth.checked_sub(1).ok_or("unexpected close delimiter")?;
            if depth == 0 {
                return Ok(at + 1);
            }
        }
    }
    Err("unterminated runtime admission delimiter".into())
}
pub(super) fn unbraced(tokens: &[Token<'_>]) -> DynResult<bool> {
    let mut at = 0;
    while at < tokens.len() {
        let word = tokens[at].text;
        if !["if", "for", "while", "switch", "catch", "do", "try", "else"].contains(&word) {
            at += 1;
            continue;
        }
        at += 1;
        if word == "else" && tokens.get(at).is_some_and(|token| token.text == "if") {
            continue;
        }
        if !["do", "try", "else"].contains(&word)
            && tokens.get(at).is_some_and(|token| token.text == "(")
        {
            at = balanced(tokens, at, "(", ")")?;
        }
        if !tokens.get(at).is_some_and(|token| token.text == "{") {
            return Ok(true);
        }
        at = balanced(tokens, at, "{", "}")?;
    }
    Ok(false)
}
pub(super) fn top_level<'a>(tokens: &[Token<'a>]) -> Vec<Token<'a>> {
    let mut result = Vec::new();
    let mut depth = 0_usize;
    for token in tokens {
        if token.text == "{" {
            depth += 1;
        } else if token.text == "}" {
            depth = depth.saturating_sub(1);
        } else if depth == 0 {
            result.push(Token {
                text: token.text,
                start: token.start,
                end: token.end,
            });
        }
    }
    result
}
