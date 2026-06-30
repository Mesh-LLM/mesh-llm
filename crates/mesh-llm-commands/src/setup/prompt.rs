use std::io::{self, IsTerminal, Write};

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

    loop {
        eprint!("{} {} [Y/n] ", prompt_marker(), message);
        let _ = io::stderr().flush();

        let mut reply = String::new();
        if io::stdin().read_line(&mut reply).is_err() {
            return Some(false);
        }

        match reply.trim().to_ascii_lowercase().as_str() {
            "" | "y" | "yes" => return Some(true),
            "n" | "no" => return Some(false),
            _ => eprintln!("Please answer y or n."),
        }
    }
}

fn prompt_marker() -> String {
    if io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none() {
        "\x1b[36m?\x1b[0m".to_string()
    } else {
        "?".to_string()
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
