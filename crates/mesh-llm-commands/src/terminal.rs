use anyhow::{Context, Result};
use std::io::{self, IsTerminal, Write};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConfirmDefault {
    Yes,
    No,
}

impl ConfirmDefault {
    const fn prompt_suffix(self) -> &'static str {
        match self {
            Self::Yes => "[Y/n]",
            Self::No => "[y/N]",
        }
    }

    const fn empty_reply(self) -> bool {
        match self {
            Self::Yes => true,
            Self::No => false,
        }
    }
}

pub(crate) fn confirm_yes_no(message: &str, default: ConfirmDefault) -> Result<Option<bool>> {
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return Ok(None);
    }

    loop {
        eprint!(
            "{} {} {} ",
            prompt_marker(),
            message,
            default.prompt_suffix()
        );
        io::stderr()
            .flush()
            .context("failed to flush confirmation prompt")?;

        let mut reply = String::new();
        let bytes_read = io::stdin()
            .read_line(&mut reply)
            .context("failed to read confirmation")?;
        if bytes_read == 0 {
            return Ok(Some(false));
        }

        match reply.trim().to_ascii_lowercase().as_str() {
            "" => return Ok(Some(default.empty_reply())),
            "y" | "yes" => return Ok(Some(true)),
            "n" | "no" => return Ok(Some(false)),
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

pub(crate) fn style_ok(text: &str) -> String {
    style(text, "32")
}

pub(crate) fn style_warn(text: &str) -> String {
    style(text, "33")
}

pub(crate) fn style_muted(text: &str) -> String {
    style(text, "2")
}

fn style(text: &str, ansi_code: &str) -> String {
    if io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none() {
        format!("\x1b[{ansi_code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

#[cfg(test)]
pub(crate) fn strip_ansi_styles(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut plain = String::with_capacity(text.len());
    let mut cursor = 0;

    while cursor < bytes.len() {
        if bytes[cursor] == b'\x1b' && bytes.get(cursor + 1) == Some(&b'[') {
            let mut end = cursor + 2;
            while bytes
                .get(end)
                .is_some_and(|byte| byte.is_ascii_digit() || *byte == b';')
            {
                end += 1;
            }
            if bytes.get(end) == Some(&b'm') {
                cursor = end + 1;
                continue;
            }
        }

        let character = text[cursor..].chars().next().expect("valid UTF-8 boundary");
        plain.push(character);
        cursor += character.len_utf8();
    }

    plain
}

#[cfg(test)]
mod tests {
    use super::strip_ansi_styles;

    #[test]
    fn strip_ansi_styles_preserves_plain_and_unicode_text() {
        assert_eq!(strip_ansi_styles("plain ✓"), "plain ✓");
        assert_eq!(strip_ansi_styles("\x1b[1;32m✓\x1b[0m ready"), "✓ ready");
    }
}
