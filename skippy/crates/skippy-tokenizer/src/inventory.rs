//! Typed vocabulary data independent of any plugin ABI.

pub const TOKENIZER_INVENTORY_SCHEMA: u32 = 1;
/// Maximum vocabulary size exposed by a bound tokenizer capability.
pub const MAX_TOKENIZER_INVENTORY_ENTRIES: usize = 1_000_000;

/// Immutable model vocabulary exposed by a bound Skippy tokenizer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerInventory {
    pub schema_version: u32,
    pub model_id: String,
    pub source_model_sha256: String,
    pub tokenizer_id: String,
    pub tokens: Vec<TokenizerInventoryToken>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerInventoryToken {
    pub id: u32,
    pub piece: TokenizerInventoryPiece,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerInventoryPiece {
    Bytes {
        bytes: Vec<u8>,
    },
    /// Opaque bytes for a native special-token descriptor. Consumers must
    /// preserve its native meaning instead of treating it as ordinary text.
    Control {
        descriptor: Vec<u8>,
    },
}
