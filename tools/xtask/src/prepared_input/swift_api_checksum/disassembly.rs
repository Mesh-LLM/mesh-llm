//! Bounded UniFFI checksum facts projected from complete otool output.
use super::{SYMBOL, instruction};
use crate::{
    command::DynResult,
    process::{ObservedLine, Stream},
};
use std::collections::BTreeMap;

const MAX_CONSTANTS: usize = 4096;

enum Pending {
    None,
    Load(String),
    Return(String, u16),
}

pub(super) struct Constants {
    pending: Pending,
    values: BTreeMap<String, u16>,
    failed: Option<&'static str>,
}

impl Constants {
    pub(super) fn new() -> Self {
        Self {
            pending: Pending::None,
            values: BTreeMap::new(),
            failed: None,
        }
    }

    pub(super) fn observe(&mut self, line: ObservedLine<'_>) {
        if line.stream != Stream::Stdout || self.failed.is_some() {
            return;
        }
        if let Err(reason) = self.line(line.bytes) {
            self.failed = Some(reason);
        }
    }

    fn line(&mut self, bytes: &[u8]) -> Result<(), &'static str> {
        let line = std::str::from_utf8(bytes).map_err(|_| "otool output must be UTF-8")?;
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::Load(name) => {
                let value = super::load_constant(line)
                    .map_err(|_| "invalid checksum constant instruction")?;
                self.pending = Pending::Return(name, value);
            }
            Pending::Return(name, value) => {
                let (ret, operands) =
                    instruction(line).map_err(|_| "invalid checksum return instruction")?;
                if !matches!(ret, "ret" | "retq") || !operands.is_empty() {
                    return Err("checksum constant must be followed by return");
                }
                if self.values.len() >= MAX_CONSTANTS {
                    return Err("too many UniFFI checksum constants");
                }
                if self.values.insert(name, value).is_some() {
                    return Err("ambiguous duplicate checksum symbol");
                }
            }
            Pending::None => {
                if let Some(name) = line
                    .trim()
                    .strip_prefix('_')
                    .and_then(|name| name.strip_suffix(':'))
                    && name.starts_with(SYMBOL)
                {
                    if name.len() > 512
                        || !name
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                    {
                        return Err("invalid UniFFI checksum symbol");
                    }
                    self.pending = Pending::Load(name.to_owned());
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish(self) -> DynResult<BTreeMap<String, u16>> {
        if let Some(reason) = self.failed {
            return Err(reason.into());
        }
        if !matches!(self.pending, Pending::None) {
            return Err("incomplete checksum constant/return".into());
        }
        if self.values.is_empty() {
            return Err("otool output contains no UniFFI API checksum constants".into());
        }
        Ok(self.values)
    }
}
