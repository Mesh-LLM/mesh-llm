use dialoguer::Confirm;
use std::io::{self, IsTerminal};

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
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return None;
    }

    Confirm::new()
        .with_prompt(message)
        .default(true)
        .wait_for_newline(true)
        .interact_opt()
        .ok()
        .flatten()
}
