#[path = "command_summary_grammar/descriptors.rs"]
mod descriptors;
#[path = "command_summary_grammar/raw_options.rs"]
mod raw_options;
#[path = "command_summary_grammar/validation.rs"]
mod validation;
#[path = "command_summary_grammar/vocabulary.rs"]
mod vocabulary;

use descriptors::DESCRIPTOR_GROUPS;
use validation::validate_descriptor;

pub(super) fn is_safe_summary(value: &str) -> bool {
    let tokens = value.split(' ').collect::<Vec<_>>();
    !tokens.is_empty()
        && tokens
            .iter()
            .all(|token| !token.is_empty() && !token.chars().any(char::is_whitespace))
        && DESCRIPTOR_GROUPS
            .iter()
            .flat_map(|group| group.iter())
            .copied()
            .any(|descriptor| validate_descriptor(&tokens, descriptor))
}

#[cfg(test)]
#[path = "command_summary_grammar/tests.rs"]
mod tests;
