use crate::terminal::{self, ConfirmDefault};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupPromptKind {
    InstallService,
    GitHubStar,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupPromptDefault {
    Yes,
}

impl SetupPromptDefault {
    pub const fn resolve(self, reply: Option<bool>) -> bool {
        match (self, reply) {
            (_, Some(value)) => value,
            (Self::Yes, None) => true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SetupConfirmPrompt {
    pub kind: SetupPromptKind,
    pub message: &'static str,
    pub default: SetupPromptDefault,
}

pub trait SetupPrompter {
    fn confirm(&mut self, prompt: SetupConfirmPrompt) -> Option<bool>;
}

pub(crate) fn confirm_yes_no(message: &str) -> Option<bool> {
    match terminal::confirm_yes_no(message, ConfirmDefault::Yes) {
        Ok(reply) => reply,
        Err(_) => Some(false),
    }
}

#[cfg(test)]
mod tests {
    use super::SetupPromptDefault;

    #[test]
    fn default_yes_still_applies_to_hidden_prompts() {
        assert!(SetupPromptDefault::Yes.resolve(None));
    }

    #[test]
    fn explicit_false_overrides_default_yes() {
        assert!(!SetupPromptDefault::Yes.resolve(Some(false)));
    }
}
