use std::io;

use crate::StageActivationCodec;

use super::{
    activation_codec::{
        ActivationShape, decode_activation, encode_activation, encoded_len,
        select_lossless_activation_codec_from_f32_payload,
    },
    invalid_data, state_flags,
    types::{MAX_STAGE_DECODED_ACTIVATION_BYTES, MAX_STAGE_SIDEBAND_VALUES},
};

pub fn activation_wire_bytes(token_count: i32, n_embd: i32) -> io::Result<usize> {
    activation_wire_bytes_with_state_flags(token_count, n_embd, 0)
}

pub fn activation_wire_bytes_with_state_flags(
    token_count: i32,
    n_embd: i32,
    state_flag_bits: i32,
) -> io::Result<usize> {
    if token_count < 0 || n_embd < 0 {
        return Err(invalid_data("negative activation dimensions"));
    }
    let token_count = (token_count as usize)
        .checked_mul(activation_payload_multiplier_from_state_flags(
            state_flag_bits,
        ))
        .ok_or_else(|| invalid_data("activation token count overflow"))?;
    let n_embd = n_embd as usize;
    let elements = token_count
        .checked_mul(n_embd)
        .ok_or_else(|| invalid_data("activation element count overflow"))?;
    elements
        .checked_mul(4)
        .ok_or_else(|| invalid_data("activation byte count overflow"))
}

pub fn activation_wire_bytes_for_codec_with_state_flags(
    codec: StageActivationCodec,
    token_count: i32,
    n_embd: i32,
    state_flag_bits: i32,
) -> io::Result<usize> {
    encoded_len(
        codec,
        activation_shape(token_count, n_embd, state_flag_bits)?,
    )
}

pub(crate) fn validate_activation_wire_payload_len(
    codec: StageActivationCodec,
    token_count: i32,
    n_embd: i32,
    state_flag_bits: i32,
    actual_wire_bytes: usize,
) -> io::Result<usize> {
    let shape = activation_shape(token_count, n_embd, state_flag_bits)?;
    let expected_wire_bytes = encoded_len(codec, shape)?;
    if has_glm_dsa_top_k_sideband(state_flag_bits) {
        if actual_wire_bytes < expected_wire_bytes {
            return Err(invalid_data("activation payload size mismatch"));
        }
        let sideband_bytes = actual_wire_bytes - expected_wire_bytes;
        validate_glm_dsa_top_k_sideband_bytes(token_count, sideband_bytes)?;
        let decoded_bytes = shape_decoded_bytes(shape)?
            .checked_add(sideband_bytes)
            .ok_or_else(|| invalid_data("decoded activation byte count overflow"))?;
        if decoded_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
            return Err(invalid_data(
                "decoded activation payload byte count exceeds maximum",
            ));
        }
        Ok(decoded_bytes)
    } else {
        if actual_wire_bytes != expected_wire_bytes {
            return Err(invalid_data("activation payload size mismatch"));
        }
        let decoded_bytes = shape_decoded_bytes(shape)?;
        if decoded_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
            return Err(invalid_data(
                "decoded activation payload byte count exceeds maximum",
            ));
        }
        Ok(decoded_bytes)
    }
}

pub fn encode_f32_activation_payload(
    token_count: i32,
    n_embd: i32,
    f32_payload: &[u8],
) -> io::Result<Vec<u8>> {
    encode_f32_activation_payload_with_state_flags(token_count, n_embd, f32_payload, 0)
}

pub fn encode_f32_activation_payload_with_state_flags(
    token_count: i32,
    n_embd: i32,
    f32_payload: &[u8],
    state_flag_bits: i32,
) -> io::Result<Vec<u8>> {
    encode_activation_payload_with_state_flags(
        StageActivationCodec::RawF32V1,
        token_count,
        n_embd,
        f32_payload,
        state_flag_bits,
    )
}

pub fn encode_activation_payload_with_state_flags(
    codec: StageActivationCodec,
    token_count: i32,
    n_embd: i32,
    f32_payload: &[u8],
    state_flag_bits: i32,
) -> io::Result<Vec<u8>> {
    let shape = activation_shape(token_count, n_embd, state_flag_bits)?;
    let primary_bytes = shape_decoded_bytes(shape)?;
    validate_decoded_payload_limit(primary_bytes)?;
    if has_glm_dsa_top_k_sideband(state_flag_bits) {
        if f32_payload.len() < primary_bytes {
            return Err(invalid_data("F32 activation payload size mismatch"));
        }
        let (primary, sideband) = f32_payload.split_at(primary_bytes);
        validate_glm_dsa_top_k_sideband_bytes(token_count, sideband.len())?;
        validate_decoded_payload_limit(f32_payload.len())?;
        let mut encoded = encode_f32_values(codec, shape, primary)?;
        encoded.extend_from_slice(sideband);
        Ok(encoded)
    } else {
        if f32_payload.len() != primary_bytes {
            return Err(invalid_data("F32 activation payload size mismatch"));
        }
        validate_decoded_payload_limit(f32_payload.len())?;
        encode_f32_values(codec, shape, f32_payload)
    }
}

pub fn select_lossless_activation_codec_with_state_flags(
    token_count: i32,
    n_embd: i32,
    f32_payload: &[u8],
    state_flag_bits: i32,
    permitted_codecs: &[StageActivationCodec],
) -> io::Result<StageActivationCodec> {
    let shape = activation_shape(token_count, n_embd, state_flag_bits)?;
    let primary_bytes = shape_decoded_bytes(shape)?;
    validate_decoded_payload_limit(primary_bytes)?;
    let primary = if has_glm_dsa_top_k_sideband(state_flag_bits) {
        if f32_payload.len() < primary_bytes {
            return Err(invalid_data("F32 activation payload size mismatch"));
        }
        let (primary, sideband) = f32_payload.split_at(primary_bytes);
        validate_glm_dsa_top_k_sideband_bytes(token_count, sideband.len())?;
        primary
    } else {
        if f32_payload.len() != primary_bytes {
            return Err(invalid_data("F32 activation payload size mismatch"));
        }
        f32_payload
    };
    validate_decoded_payload_limit(f32_payload.len())?;
    select_lossless_activation_codec_from_f32_payload(shape, primary, permitted_codecs)
}

pub(crate) fn decode_activation_payload_with_state_flags(
    codec: StageActivationCodec,
    token_count: i32,
    n_embd: i32,
    payload: &[u8],
    state_flag_bits: i32,
) -> io::Result<Vec<u8>> {
    let shape = activation_shape(token_count, n_embd, state_flag_bits)?;
    let primary_wire_bytes = encoded_len(codec, shape)?;
    let (primary_payload, sideband) = if has_glm_dsa_top_k_sideband(state_flag_bits) {
        if payload.len() < primary_wire_bytes {
            return Err(invalid_data("activation payload size mismatch"));
        }
        let (primary, sideband) = payload.split_at(primary_wire_bytes);
        validate_glm_dsa_top_k_sideband_bytes(token_count, sideband.len())?;
        (primary, sideband)
    } else {
        (payload, &[][..])
    };
    let values = decode_activation(codec, shape, primary_payload)?;
    let decoded_capacity = values
        .len()
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(sideband.len()))
        .ok_or_else(|| invalid_data("decoded activation byte count overflow"))?;
    validate_decoded_payload_limit(decoded_capacity)?;
    let mut decoded = Vec::with_capacity(decoded_capacity);
    for value in values {
        decoded.extend_from_slice(&value.to_le_bytes());
    }
    decoded.extend_from_slice(sideband);
    Ok(decoded)
}

fn encode_f32_values(
    codec: StageActivationCodec,
    shape: ActivationShape,
    payload: &[u8],
) -> io::Result<Vec<u8>> {
    if !payload.len().is_multiple_of(4) {
        return Err(invalid_data("F32 activation payload size mismatch"));
    }
    let values = payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    encode_activation(codec, shape, &values)
}

fn activation_shape(
    token_count: i32,
    n_embd: i32,
    state_flag_bits: i32,
) -> io::Result<ActivationShape> {
    if token_count < 0 || n_embd <= 0 {
        return Err(invalid_data("negative activation dimensions"));
    }
    let primary_rows = token_count as usize;
    let multiplier = activation_payload_multiplier_from_state_flags(state_flag_bits);
    let sideband_rows = primary_rows
        .checked_mul(multiplier.saturating_sub(1))
        .ok_or_else(|| invalid_data("activation token count overflow"))?;
    Ok(ActivationShape::new(
        primary_rows,
        sideband_rows,
        n_embd as usize,
    ))
}

fn shape_decoded_bytes(shape: ActivationShape) -> io::Result<usize> {
    shape
        .primary_rows
        .checked_add(shape.sideband_rows)
        .and_then(|rows| rows.checked_mul(shape.columns))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| invalid_data("decoded activation byte count overflow"))
}

fn validate_decoded_payload_limit(payload_bytes: usize) -> io::Result<()> {
    if payload_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
        return Err(invalid_data(
            "decoded activation payload byte count exceeds maximum",
        ));
    }
    Ok(())
}

fn has_glm_dsa_top_k_sideband(state_flag_bits: i32) -> bool {
    state_flag_bits & state_flags::GLM_DSA_TOP_K_SIDEBAND != 0
}

fn validate_glm_dsa_top_k_sideband_bytes(
    token_count: i32,
    sideband_bytes: usize,
) -> io::Result<()> {
    if token_count <= 0 {
        return Err(invalid_data(
            "GLM-DSA top-k sideband requires positive token count",
        ));
    }
    let bytes_per_column = (token_count as usize)
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| invalid_data("GLM-DSA top-k sideband byte count overflow"))?;
    if sideband_bytes == 0 || !sideband_bytes.is_multiple_of(bytes_per_column) {
        return Err(invalid_data(
            "GLM-DSA top-k sideband is not token-major i32",
        ));
    }
    let values = sideband_bytes / std::mem::size_of::<i32>();
    if values > MAX_STAGE_SIDEBAND_VALUES {
        return Err(invalid_data(
            "GLM-DSA top-k sideband value count exceeds maximum",
        ));
    }
    Ok(())
}

pub fn activation_payload_multiplier_from_state_flags(state_flag_bits: i32) -> usize {
    // Generation 7 graph boundaries report the complete Gemma3n AltUp tensor
    // width. Its flag describes the tensor semantics; it must not multiply an
    // already multidimensional payload a second time.
    if (state_flag_bits
        & (state_flags::INKLING_MTP_EMBD_SIDEBAND | state_flags::RWKV7_V_FIRST_SIDEBAND))
        != 0
    {
        2
    } else {
        1
    }
}
