use std::path::Path;

use model_artifact::gguf::scan_gguf_tokenizer_inventory;
use skippy_protocol::StageConfig;
use skippy_tokenizer::TokenizerIdentity;
use skippy_tokenizer::inventory;

use super::{TOKENIZER_VERSION, TokenizerSource};

pub(super) fn source_gguf_path(config: &StageConfig) -> Option<&Path> {
    [
        config.source_model_path.as_deref(),
        config.model_path.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(Path::new)
    .find(|path| path.is_file())
}

pub(super) fn inventory_from_stage(
    config: &StageConfig,
    identity: &TokenizerIdentity,
    source: &dyn TokenizerSource,
) -> Option<inventory::TokenizerInventory> {
    let source_path = source_gguf_path(config)?;
    let vocabulary = scan_gguf_tokenizer_inventory(source_path)?;
    if vocabulary.tokens.len() > inventory::MAX_TOKENIZER_INVENTORY_ENTRIES {
        return None;
    }
    let token_ids = (0..vocabulary.tokens.len())
        .map(|id| i32::try_from(id).ok())
        .collect::<Option<Vec<_>>>()?;
    let token_pieces = source.token_pieces(&token_ids).ok()?;
    if token_pieces.len() != vocabulary.tokens.len() {
        return None;
    }
    let mut tokens = Vec::with_capacity(vocabulary.tokens.len());
    for (id, (token, bytes)) in vocabulary.tokens.into_iter().zip(token_pieces).enumerate() {
        let id = u32::try_from(id).ok()?;
        let piece = if token.is_control {
            inventory::TokenizerInventoryPiece::Control {
                descriptor: token.raw,
            }
        } else {
            inventory::TokenizerInventoryPiece::Bytes { bytes }
        };
        tokens.push(inventory::TokenizerInventoryToken { id, piece });
    }
    Some(inventory::TokenizerInventory {
        schema_version: inventory::TOKENIZER_INVENTORY_SCHEMA,
        model_id: identity.model_id.clone(),
        source_model_sha256: identity.source_model_sha256.clone(),
        tokenizer_id: identity.tokenizer_id.clone(),
        tokens,
    })
}

pub(super) fn tokenizer_binding_digest(
    identity: &TokenizerIdentity,
    inventory: &inventory::TokenizerInventory,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"mesh-native-tokenizer-binding-v1\0");
    update_string(&mut hasher, &identity.model_id);
    update_string(&mut hasher, &identity.source_model_sha256);
    update_string(&mut hasher, &identity.tokenizer_id);
    update_string(&mut hasher, TOKENIZER_VERSION);
    hasher.update(&inventory.schema_version.to_le_bytes());
    hasher.update(&(inventory.tokens.len() as u64).to_le_bytes());
    for token in &inventory.tokens {
        hasher.update(&token.id.to_le_bytes());
        match &token.piece {
            inventory::TokenizerInventoryPiece::Bytes { bytes } => {
                hasher.update(&[0]);
                hasher.update(&(bytes.len() as u64).to_le_bytes());
                hasher.update(bytes);
            }
            inventory::TokenizerInventoryPiece::Control { descriptor } => {
                hasher.update(&[1]);
                hasher.update(&(descriptor.len() as u64).to_le_bytes());
                hasher.update(descriptor);
            }
        }
    }
    *hasher.finalize().as_bytes()
}

fn update_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}
