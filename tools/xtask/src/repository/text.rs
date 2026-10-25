pub(crate) fn is_space(character: char) -> bool {
    character.is_whitespace()
}

pub(crate) fn strip(text: &str) -> &str {
    text.trim()
}

pub(crate) fn split_whitespace(text: &str) -> impl Iterator<Item = &str> {
    text.split_whitespace()
}

pub(crate) fn splitlines(text: &str) -> Vec<&str> {
    text.lines().collect()
}

pub(crate) fn is_upper(text: &str) -> bool {
    text.chars().any(char::is_uppercase) && !text.chars().any(char::is_lowercase)
}

pub(crate) fn is_decimal(character: char) -> bool {
    character.is_ascii_digit()
}

pub(crate) fn repr(text: &str) -> String {
    format!("{text:?}")
}
