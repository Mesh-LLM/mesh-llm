//! The metadata arrays of a header that a check reads before the native
//! loader sees the file. The header reader skips every other array.
//!
//! The runtime's loader asserts on some of them, which ends the process:
//! two equal strings in `tokenizer.ggml.tokens`, a per-layer expert count
//! above the expert count, rope sections that do not fit a head. So these
//! are read here, and a caller checks them first (the auditor's
//! `native_model::loader_asserts`).

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

use anyhow::Result;

use super::super::limits::{MAX_STRING_BYTES, refuse};
use super::{Cursor, GGUF_STRING, scalar_bytes};

/// The vocabulary's tokens, which the loader keeps by text.
pub(crate) const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
/// The most tokens a vocabulary may hold here. The largest vocabularies
/// of real models have about 262,000.
const MAX_TOKENS: u64 = 1 << 22;
/// The most elements of an integer array that is kept.
const MAX_KEPT_INTS: u64 = 4096;
/// The integer arrays kept, by the end of their key: per-layer counts and
/// rope sections.
const KEPT_INT_SUFFIXES: [&str; 4] = [
    ".expert_used_count",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".rope.dimension_sections",
];

/// A metadata array a check of the native loader's preconditions reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetaArray {
    /// Integers, signed ones sign-extended.
    Ints(Vec<i64>),
    /// The vocabulary, as the loader reads it ([`TokenList`]).
    Tokens(TokenList),
}

/// What the loader would make of `tokenizer.ggml.tokens`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenList {
    pub count: u64,
    /// The first token whose text an earlier token has, after the loader
    /// names an empty token `[EMPTY_i]`: the loader asserts there is none.
    pub duplicate: Option<u64>,
    /// Per byte, a bit set when a token stands for it as a SentencePiece
    /// vocabulary looks it up: `<0xXX>`, or the byte alone.
    bytes: [u64; 4],
}

impl TokenList {
    /// Whether every byte has a token, which a SentencePiece vocabulary
    /// falls back to for text it has no token of.
    pub fn has_every_byte(&self) -> bool {
        self.bytes.iter().all(|&bits| bits == u64::MAX)
    }

    /// Whether a token stands for `byte`.
    pub fn has_byte(&self, byte: u8) -> bool {
        self.bytes[usize::from(byte >> 6)] >> (byte & 63) & 1 == 1
    }
}

/// Whether the integer array under `key` is kept.
pub(super) fn keeps_ints(key: &str) -> bool {
    KEPT_INT_SUFFIXES.iter().any(|suffix| key.ends_with(suffix))
}

/// The byte a token of `text` stands for: `<0xXX>` with upper-case digits,
/// as the loader writes the name, or a text of one byte.
fn byte_of(text: &[u8]) -> Option<u8> {
    match text {
        [byte] => Some(*byte),
        [b'<', b'0', b'x', high, low, b'>'] => {
            let digit = |d: u8| match d {
                b'0'..=b'9' => Some(d - b'0'),
                b'A'..=b'F' => Some(d - b'A' + 10),
                _ => None,
            };
            Some(digit(*high)? << 4 | digit(*low)?)
        }
        _ => None,
    }
}

fn hash(text: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

impl Cursor<'_> {
    /// Reads an array of element type `inner` and `count` elements, after
    /// its type and count, as integers; `None`, having skipped it, when its
    /// elements are not integers.
    pub(super) fn int_array(&mut self, inner: u32, count: u64) -> Result<Option<Vec<i64>>> {
        let signed = matches!(inner, 1 | 3 | 5 | 11);
        let Some(width) = scalar_bytes(inner).filter(|_| matches!(inner, 0..=5 | 10 | 11)) else {
            self.skip_elements(inner, count, 1)?;
            return Ok(None);
        };
        if count > MAX_KEPT_INTS {
            return refuse(format!(
                "its header has an integer array of {count} elements, more than {MAX_KEPT_INTS}"
            ));
        }
        let mut out = Vec::with_capacity(usize::try_from(count)?);
        for _ in 0..count {
            let bytes = self.bytes(width as u64)?;
            let mut le = [0u8; 8];
            le[..width].copy_from_slice(bytes);
            let mut value = i64::from_le_bytes(le);
            if signed && width < 8 && bytes[width - 1] & 0x80 != 0 {
                value |= -1i64 << (width * 8);
            }
            out.push(value);
        }
        Ok(Some(out))
    }

    /// Reads `tokenizer.ggml.tokens`, an array of `count` strings after its
    /// type and count, as the loader keeps it. Equal texts are found by a
    /// hash of each: two equal texts always meet, and two texts that only
    /// share a hash refuse a file no loader would.
    pub(super) fn token_list(&mut self, count: u64) -> Result<TokenList> {
        if count == 0 || count > MAX_TOKENS {
            return refuse(format!(
                "its vocabulary holds {count} tokens, not 1 to {MAX_TOKENS}"
            ));
        }
        let mut seen = HashSet::with_capacity(usize::try_from(count)?);
        let mut list = TokenList {
            count,
            duplicate: None,
            bytes: [0; 4],
        };
        for index in 0..count {
            let n = self.u64()?;
            if n > MAX_STRING_BYTES {
                return refuse(format!(
                    "a token of {n} bytes is longer than {MAX_STRING_BYTES} in its header"
                ));
            }
            let text = self.bytes(n)?;
            if let Some(byte) = byte_of(text) {
                list.bytes[usize::from(byte >> 6)] |= 1 << (byte & 63);
            }
            let key = match text.is_empty() {
                true => hash(format!("[EMPTY_{index}]").as_bytes()),
                false => hash(text),
            };
            if !seen.insert(key) && list.duplicate.is_none() {
                list.duplicate = Some(index);
            }
        }
        Ok(list)
    }

    /// Reads the array under `key`, after its element type and count: kept
    /// when a check reads it, else skipped.
    pub(super) fn kept_array(&mut self, key: &str) -> Result<Option<MetaArray>> {
        let inner = self.u32()?;
        let count = self.u64()?;
        if key == TOKENS_KEY && inner == GGUF_STRING {
            return Ok(Some(MetaArray::Tokens(self.token_list(count)?)));
        }
        if keeps_ints(key) {
            return Ok(self.int_array(inner, count)?.map(MetaArray::Ints));
        }
        self.skip_elements(inner, count, 1)?;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_byte_token_is_named_as_the_loader_names_it() {
        assert_eq!(byte_of(b"<0x0A>"), Some(10));
        assert_eq!(byte_of(b"<0xFF>"), Some(255));
        assert_eq!(byte_of(b"a"), Some(b'a'));
        assert_eq!(byte_of(b"<0xff>"), None);
        assert_eq!(byte_of(b"<0x0A"), None);
        assert_eq!(byte_of(b"ab"), None);
        assert_eq!(byte_of(b""), None);
    }

    #[test]
    fn only_the_arrays_a_check_reads_are_kept() {
        for key in [
            "qwen3moe.expert_used_count",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "qwen35.rope.dimension_sections",
        ] {
            assert!(keeps_ints(key), "{key}");
        }
        assert!(!keeps_ints("tokenizer.ggml.scores"));
        assert!(!keeps_ints("tokenizer.ggml.token_type"));
        assert!(!keeps_ints("gemma4.attention.sliding_window_pattern"));
    }
}
