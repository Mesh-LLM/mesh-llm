//! Python `fnmatch.fnmatchcase` for ownership patterns: `*` matches any run
//! of characters including `/`, `?` matches one character, and `[...]` is a
//! character class (`!` negates, a leading `]` is literal, `a-z` is a
//! range). An unterminated `[` is a literal bracket.

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Literal(char),
    AnyOne,
    AnyRun,
    Class {
        negated: bool,
        items: Vec<(char, char)>,
    },
}

fn parse_class(chars: &[char], open: usize) -> Option<(Token, usize)> {
    let mut index = open + 1;
    let negated = chars.get(index) == Some(&'!');
    if negated {
        index += 1;
    }
    let first = index;
    let mut items = Vec::new();
    while index < chars.len() {
        let ch = chars[index];
        if ch == ']' && index > first {
            return Some((Token::Class { negated, items }, index + 1));
        }
        if chars.get(index + 1) == Some(&'-') && chars.get(index + 2).is_some_and(|end| *end != ']')
        {
            let end = chars[index + 2];
            if ch <= end {
                items.push((ch, end));
            }
            index += 3;
        } else {
            items.push((ch, ch));
            index += 1;
        }
    }
    None
}

fn tokenize(pattern: &str) -> Vec<Token> {
    let chars = pattern.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            '*' => {
                if tokens.last() != Some(&Token::AnyRun) {
                    tokens.push(Token::AnyRun);
                }
                index += 1;
            }
            '?' => {
                tokens.push(Token::AnyOne);
                index += 1;
            }
            '[' => match parse_class(&chars, index) {
                Some((class, next)) => {
                    tokens.push(class);
                    index = next;
                }
                None => {
                    tokens.push(Token::Literal('['));
                    index += 1;
                }
            },
            ch => {
                tokens.push(Token::Literal(ch));
                index += 1;
            }
        }
    }
    tokens
}

fn single(token: &Token, ch: char) -> bool {
    match token {
        Token::Literal(expected) => *expected == ch,
        Token::AnyOne => true,
        Token::AnyRun => false,
        Token::Class { negated, items } => {
            items.iter().any(|(low, high)| (*low..=*high).contains(&ch)) != *negated
        }
    }
}

/// Iterative wildcard match with single-star backtracking.
pub(super) fn matches(text: &str, pattern: &str) -> bool {
    let tokens = tokenize(pattern);
    let chars = text.chars().collect::<Vec<_>>();
    let (mut token, mut position) = (0, 0);
    let mut resume: Option<(usize, usize)> = None;
    while position < chars.len() {
        match tokens.get(token) {
            Some(Token::AnyRun) => {
                resume = Some((token, position));
                token += 1;
            }
            Some(current) if single(current, chars[position]) => {
                token += 1;
                position += 1;
            }
            _ => match resume {
                Some((star, start)) => {
                    token = star + 1;
                    position = start + 1;
                    resume = Some((star, start + 1));
                }
                None => return false,
            },
        }
    }
    tokens[token..].iter().all(|rest| *rest == Token::AnyRun)
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn migration_ci_plan_glob_star_crosses_directories() {
        assert!(matches("crates/a/src/lib.rs", "crates/**"));
        assert!(matches("crates/a/src/lib.rs", "crates/*.rs"));
        assert!(matches("README.md", "*.md"));
        assert!(!matches("docs/a.md", "*.txt"));
        assert!(matches("", "*"));
        assert!(!matches("", "?"));
    }

    #[test]
    fn migration_ci_plan_glob_classes_follow_fnmatch() {
        assert!(matches("a1", "a[0-9]"));
        assert!(!matches("ab", "a[0-9]"));
        assert!(matches("ab", "a[!0-9]"));
        assert!(matches("a]", "a[]]"));
        assert!(matches("a[", "a["));
        assert!(matches("skippy-ffi", "skippy-?fi"));
    }
}
