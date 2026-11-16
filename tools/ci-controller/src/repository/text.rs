pub fn is_space(character: char) -> bool {
    character.is_whitespace()
}

pub fn strip(text: &str) -> &str {
    text.trim()
}

pub fn split_whitespace(text: &str) -> impl Iterator<Item = &str> {
    text.split_whitespace()
}

pub fn splitlines(text: &str) -> Vec<&str> {
    text.lines().collect()
}

pub fn repr(text: &str) -> String {
    format!("{text:?}")
}
