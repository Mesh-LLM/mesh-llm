use super::ManifestError;
use crate::repository::text::is_space;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Field {
    Url,
    Checksum,
}

impl Field {
    const fn name(self) -> &'static str {
        match self {
            Self::Url => "remoteFFIXCFrameworkURL",
            Self::Checksum => "remoteFFIXCFrameworkChecksum",
        }
    }

    fn first(self, text: &str) -> Result<(Range<usize>, Range<usize>), ManifestError> {
        for (start, _) in text.match_indices("let") {
            let after_let = &text[start + 3..];
            let spaced = after_let.trim_start_matches(is_space);
            if spaced.len() == after_let.len() {
                continue;
            }
            let Some(after_name) = spaced.strip_prefix(self.name()) else {
                continue;
            };
            let Some(after_equals) = after_name.trim_start_matches(is_space).strip_prefix('=')
            else {
                continue;
            };
            let Some(value) = after_equals.trim_start_matches(is_space).strip_prefix('"') else {
                continue;
            };
            let Some(length) = value.find('"') else {
                continue;
            };
            let value_start = text.len() - value.len();
            let value_end = value_start + length;
            return Ok((start..value_end + 1, value_start..value_end));
        }
        Err(ManifestError::Missing(self))
    }

    pub(super) fn value(self, text: &str) -> Result<&str, ManifestError> {
        let (_, value) = self.first(text)?;
        Ok(&text[value])
    }

    pub(super) fn replace_first(self, text: &str, value: &str) -> Result<String, ManifestError> {
        let (matched, _) = self.first(text)?;
        let mut updated = text.to_owned();
        updated.replace_range(matched, &format!("let {} = \"{value}\"", self.name()));
        Ok(updated)
    }
}

impl std::fmt::Display for Field {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}
