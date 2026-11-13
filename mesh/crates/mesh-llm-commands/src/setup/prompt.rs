use crate::terminal::{self, ConfirmDefault};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupPromptKind {
    InstallService,
    InstallPlugins,
    GitHubStar,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupPromptDefault {
    Yes,
    No,
}

impl SetupPromptDefault {
    pub const fn resolve(self, reply: Option<bool>) -> bool {
        match (self, reply) {
            (_, Some(value)) => value,
            (Self::Yes, None) => true,
            (Self::No, None) => false,
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

pub(crate) fn confirm_yes_no(message: &str, default: SetupPromptDefault) -> Option<bool> {
    let default = match default {
        SetupPromptDefault::Yes => ConfirmDefault::Yes,
        SetupPromptDefault::No => ConfirmDefault::No,
    };
    match terminal::confirm_yes_no(message, default) {
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
