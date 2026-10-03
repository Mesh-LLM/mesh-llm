//! Credential-safe diagnostics for the trusted canary publisher.
use crate::command::DynResult;
use crate::repository::check_report::CheckReport;
use std::io::{Read, Write};

const USAGE: &str = "cargo xtool automation canary-receipts redact-publication-log";

pub(super) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        return CheckReport::success(format!(
            "usage: {USAGE}\nRedact the inherited CANARY_REPAIR_TOKEN from UTF-8 stdin.\n"
        ))
        .emit();
    }
    if !args.is_empty() {
        return CheckReport::failure(
            String::new(),
            format!("usage: {USAGE}\nNo arguments are accepted.\n"),
        )
        .emit();
    }
    let token = std::env::var("CANARY_REPAIR_TOKEN")
        .map_err(|_| "canary publication redaction requires a UTF-8 CANARY_REPAIR_TOKEN")?;
    redact(std::io::stdin().lock(), std::io::stdout().lock(), &token)
}

fn redact(mut input: impl Read, mut output: impl Write, token: &str) -> DynResult<()> {
    if token.is_empty() {
        return Err("canary publication redaction requires a nonempty CANARY_REPAIR_TOKEN".into());
    }
    // Validate the complete input before emitting any diagnostics. Neither a
    // malformed stream nor an input read failure may print unredacted bytes.
    let mut text = String::new();
    input
        .read_to_string(&mut text)
        .map_err(|_| "canary publication diagnostics could not be read as UTF-8")?;
    output.write_all(text.replace(token, "***redacted***").as_bytes())?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_nonsecret_unicode_newlines_and_literal_token_matches() {
        let mut output = Vec::new();
        redact(
            "λ token.[x]\r\n\ntoken.[x] end".as_bytes(),
            &mut output,
            "token.[x]",
        )
        .unwrap();
        assert_eq!(
            output,
            "λ ***redacted***\r\n\n***redacted*** end".as_bytes()
        );
    }

    #[test]
    fn replacement_consumes_nonoverlapping_matches_and_preserves_unterminated_input() {
        let mut output = Vec::new();
        redact("ababa aba".as_bytes(), &mut output, "aba").unwrap();
        assert_eq!(output, b"***redacted***ba ***redacted***");
    }

    #[test]
    fn empty_token_and_invalid_utf8_never_emit_input() {
        for (input, token) in [
            (b"visible".as_slice(), ""),
            (b"token\xff".as_slice(), "token"),
        ] {
            let mut output = Vec::new();
            assert!(redact(input, &mut output, token).is_err());
            assert!(output.is_empty());
        }
    }

    #[test]
    fn incomplete_input_read_never_emits_credential_bytes() {
        struct Interrupted(bool);
        impl Read for Interrupted {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                if self.0 {
                    return Err(std::io::Error::other("fixture failure"));
                }
                self.0 = true;
                let count = output.len().min(5);
                output[..count].copy_from_slice(&b"token"[..count]);
                Ok(count)
            }
        }
        let mut output = Vec::new();
        assert!(redact(Interrupted(false), &mut output, "token").is_err());
        assert!(output.is_empty());
    }
}
