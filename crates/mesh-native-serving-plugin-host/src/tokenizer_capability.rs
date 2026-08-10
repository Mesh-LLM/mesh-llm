use std::{ffi::c_void, mem::size_of, sync::Arc};

use anyhow::{Context, Result, anyhow, bail};
use mesh_native_serving_plugin_api as abi;
use skippy_server::tokenizer::TokenizerCapability;
use skippy_tokenizer::{EncodeRequest, InputPiece};

pub(super) struct ActivationInventory {
    entries: Vec<abi::TokenizerInventoryEntry>,
}

impl ActivationInventory {
    fn from_inventory(inventory: &abi::TokenizerInventory) -> Result<Self> {
        if inventory.schema_version != abi::TOKENIZER_INVENTORY_SCHEMA
            || inventory.tokens.is_empty()
            || inventory.tokens.len() > abi::MAX_TOKENIZER_INVENTORY_ENTRIES
        {
            bail!("bound model exposes an unsupported or empty tokenizer inventory");
        }
        let mut previous_id = None;
        let entries = inventory
            .tokens
            .iter()
            .map(|entry| {
                if entry.id > abi::MAX_TOKENIZER_INVENTORY_ENTRIES as u32 {
                    return Err(anyhow!(
                        "native tokenizer ID exceeds the bounded inventory limit"
                    ));
                }
                if previous_id.is_some_and(|previous| entry.id <= previous) {
                    return Err(anyhow!(
                        "native tokenizer inventory IDs must be strictly increasing"
                    ));
                }
                previous_id = Some(entry.id);
                let (piece_kind, bytes) = match &entry.piece {
                    abi::TokenizerInventoryPiece::Bytes { bytes } => {
                        (abi::TokenizerPieceKind::BYTES, bytes.as_slice())
                    }
                    abi::TokenizerInventoryPiece::Control { descriptor } => {
                        if descriptor.is_empty() {
                            return Err(anyhow!(
                                "native tokenizer control descriptor must not be empty"
                            ));
                        }
                        (abi::TokenizerPieceKind::CONTROL, descriptor.as_slice())
                    }
                };
                Ok(abi::TokenizerInventoryEntry {
                    id: entry.id,
                    piece_kind,
                    bytes: abi::ByteSlice::from_bytes(bytes),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { entries })
    }

    fn view(&self) -> abi::TokenizerInventoryView {
        abi::TokenizerInventoryView {
            struct_size: size_of::<abi::TokenizerInventoryView>(),
            schema_version: abi::TOKENIZER_INVENTORY_SCHEMA,
            entries: self.entries.as_ptr(),
            entry_count: self.entries.len(),
        }
    }
}

pub(super) struct HostTokenizerCapability {
    pub(super) tokenizer: TokenizerCapability,
    inventory: ActivationInventory,
    inventory_view: abi::TokenizerInventoryView,
    pub(super) abi: abi::TokenizerCapability,
}

// SAFETY: the raw pointers are immutable views into fields of this boxed
// value, which is kept alive by ActivePlugin for the whole plugin lifetime.
unsafe impl Send for HostTokenizerCapability {}
// SAFETY: the callback delegates to the model-bound, synchronized tokenizer.
unsafe impl Sync for HostTokenizerCapability {}

impl HostTokenizerCapability {
    pub(super) fn new(tokenizer: TokenizerCapability) -> Result<Arc<Self>> {
        let inventory = ActivationInventory::from_inventory(
            tokenizer
                .inventory()
                .context("bound model does not expose a tokenizer inventory")?,
        )?;
        let binding_digest = tokenizer
            .binding_digest()
            .context("bound model does not expose a tokenizer binding digest")?;
        let limits = tokenizer.limits();
        let mut capability = Arc::new(Self {
            tokenizer,
            inventory,
            inventory_view: abi::TokenizerInventoryView {
                struct_size: 0,
                schema_version: 0,
                entries: std::ptr::null(),
                entry_count: 0,
            },
            abi: abi::TokenizerCapability {
                struct_size: size_of::<abi::TokenizerCapability>(),
                abi_version: abi::TOKENIZER_CAPABILITY_ABI,
                model_id: abi::ByteSlice::default(),
                source_model_sha256: abi::ByteSlice::default(),
                tokenizer_id: abi::ByteSlice::default(),
                limits: abi::TokenizerLimits {
                    max_input_bytes: limits.max_input_bytes,
                    max_output_tokens: limits.max_output_tokens,
                },
                binding_digest,
                inventory: std::ptr::null(),
                context: std::ptr::null_mut(),
                encode: encode_tokenizer,
            },
        });
        let inner = Arc::get_mut(&mut capability).expect("new tokenizer capability is unique");
        let identity = inner.tokenizer.identity().clone();
        inner.abi.model_id = abi::ByteSlice::from_bytes(identity.model_id.as_bytes());
        inner.abi.source_model_sha256 =
            abi::ByteSlice::from_bytes(identity.source_model_sha256.as_bytes());
        inner.abi.tokenizer_id = abi::ByteSlice::from_bytes(identity.tokenizer_id.as_bytes());
        inner.inventory_view = inner.inventory.view();
        inner.abi.inventory = &raw const inner.inventory_view;
        inner.abi.context = (&raw const inner.tokenizer).cast_mut().cast();
        Ok(capability)
    }
}

unsafe extern "C" fn encode_tokenizer(
    context: *mut c_void,
    input_pieces: *const abi::TokenizerInputPiece,
    input_piece_count: usize,
    output_tokens: *mut i32,
    output_capacity: usize,
    output_length: *mut usize,
) -> abi::TokenizerEncodeStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if context.is_null() || output_length.is_null() {
            return abi::TokenizerEncodeStatus::INVALID_ARGUMENT;
        }
        if input_pieces.is_null() && input_piece_count != 0 {
            return abi::TokenizerEncodeStatus::INVALID_ARGUMENT;
        }
        if output_tokens.is_null() && output_capacity != 0 {
            return abi::TokenizerEncodeStatus::INVALID_ARGUMENT;
        }
        if input_piece_count > abi::MAX_TOKENIZER_INPUT_PIECES {
            unsafe { *output_length = 0 };
            return abi::TokenizerEncodeStatus::LIMIT_EXCEEDED;
        }
        let pieces = if input_piece_count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(input_pieces, input_piece_count) }
        };
        let mut total_input_bytes = 0usize;
        for piece in pieces {
            if piece.bytes.pointer.is_null() && piece.bytes.length != 0 {
                return abi::TokenizerEncodeStatus::INVALID_ARGUMENT;
            }
            total_input_bytes = match total_input_bytes.checked_add(piece.bytes.length) {
                Some(total) => total,
                None => return abi::TokenizerEncodeStatus::LIMIT_EXCEEDED,
            };
        }
        let tokenizer = unsafe { &*context.cast::<TokenizerCapability>() };
        let limits = tokenizer.limits();
        if total_input_bytes > limits.max_input_bytes {
            unsafe { *output_length = 0 };
            return abi::TokenizerEncodeStatus::LIMIT_EXCEEDED;
        }
        unsafe { *output_length = 0 };
        let mut owned_pieces = Vec::with_capacity(input_piece_count);
        for piece in pieces {
            let bytes = if piece.bytes.length == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(piece.bytes.pointer, piece.bytes.length) }
            };
            owned_pieces.push(match piece.kind {
                kind if kind == abi::TokenizerInputPieceKind::BYTES => {
                    InputPiece::Bytes(bytes.to_vec())
                }
                kind if kind == abi::TokenizerInputPieceKind::CONTROL => InputPiece::Control {
                    descriptor: bytes.to_vec(),
                },
                _ => return abi::TokenizerEncodeStatus::UNSUPPORTED_INPUT,
            });
        }
        let request = EncodeRequest::new(tokenizer.identity().clone(), owned_pieces);
        let encoded = match tokenizer.encode(request) {
            Ok(encoded) => encoded,
            Err(error) => return encode_status(error),
        };
        write_encode_output(
            &encoded.token_ids,
            output_tokens,
            output_capacity,
            output_length,
        )
    }));
    result.unwrap_or(abi::TokenizerEncodeStatus::INTERNAL_ERROR)
}

fn write_encode_output(
    token_ids: &[i32],
    output_tokens: *mut i32,
    output_capacity: usize,
    output_length: *mut usize,
) -> abi::TokenizerEncodeStatus {
    if output_length.is_null() || (output_tokens.is_null() && output_capacity != 0) {
        return abi::TokenizerEncodeStatus::INVALID_ARGUMENT;
    }
    unsafe { *output_length = token_ids.len() };
    if token_ids.len() > output_capacity {
        return abi::TokenizerEncodeStatus::OUTPUT_TOO_SMALL;
    }
    if !token_ids.is_empty() {
        unsafe {
            std::ptr::copy_nonoverlapping(token_ids.as_ptr(), output_tokens, token_ids.len());
        }
    }
    abi::TokenizerEncodeStatus::OK
}

fn encode_status(error: skippy_tokenizer::TokenizerError) -> abi::TokenizerEncodeStatus {
    match error {
        skippy_tokenizer::TokenizerError::UnsupportedInput { .. } => {
            abi::TokenizerEncodeStatus::UNSUPPORTED_INPUT
        }
        skippy_tokenizer::TokenizerError::RuntimeUnavailable => {
            abi::TokenizerEncodeStatus::UNAVAILABLE
        }
        skippy_tokenizer::TokenizerError::InputTooLarge { .. }
        | skippy_tokenizer::TokenizerError::TooManyPieces { .. }
        | skippy_tokenizer::TokenizerError::TooManyTokens { .. } => {
            abi::TokenizerEncodeStatus::LIMIT_EXCEEDED
        }
        skippy_tokenizer::TokenizerError::IdentityMismatch { .. }
        | skippy_tokenizer::TokenizerError::IdentityUnavailable
        | skippy_tokenizer::TokenizerError::StageZeroRequired
        | skippy_tokenizer::TokenizerError::UnsupportedStage { .. } => {
            abi::TokenizerEncodeStatus::INVALID_ARGUMENT
        }
        skippy_tokenizer::TokenizerError::BackendFailure { .. }
        | skippy_tokenizer::TokenizerError::BatchTooLarge { .. }
        | skippy_tokenizer::TokenizerError::BatchInputTooLarge { .. } => {
            abi::TokenizerEncodeStatus::INTERNAL_ERROR
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_inventory_view_borrows_host_owned_bytes_for_activation() {
        let inventory = abi::TokenizerInventory {
            schema_version: abi::TOKENIZER_INVENTORY_SCHEMA,
            model_id: "glm".to_string(),
            source_model_sha256: "a".repeat(64),
            tokenizer_id: "gguf-source-sha256:test".to_string(),
            tokens: vec![abi::TokenizerInventoryToken {
                id: 0,
                piece: abi::TokenizerInventoryPiece::Bytes {
                    bytes: b"hello".to_vec(),
                },
            }],
        };
        let activation = ActivationInventory::from_inventory(&inventory).unwrap();
        let view = activation.view();
        assert_eq!(view.schema_version, abi::TOKENIZER_INVENTORY_SCHEMA);
        assert_eq!(view.entry_count, 1);
        let entry = unsafe { &*view.entries };
        assert_eq!(entry.piece_kind, abi::TokenizerPieceKind::BYTES);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(entry.bytes.pointer, entry.bytes.length) },
            b"hello"
        );
    }
}
