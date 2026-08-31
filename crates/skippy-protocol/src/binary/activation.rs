use std::io;

use super::{invalid_data, state_flags, types::MAX_STAGE_DECODED_ACTIVATION_BYTES};

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

pub(crate) fn activation_decoded_f32_bytes_with_state_flags(
    token_count: i32,
    n_embd: i32,
    state_flag_bits: i32,
) -> io::Result<usize> {
    activation_wire_bytes_with_state_flags(token_count, n_embd, state_flag_bits)
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
    let expected_f32_bytes =
        activation_decoded_f32_bytes_with_state_flags(token_count, n_embd, state_flag_bits)?;
    if expected_f32_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
        return Err(invalid_data(
            "decoded activation payload byte count exceeds maximum",
        ));
    }
    if f32_payload.len() != expected_f32_bytes {
        return Err(invalid_data("F32 activation payload size mismatch"));
    }
    Ok(f32_payload.to_vec())
}

pub fn activation_payload_multiplier_from_state_flags(state_flag_bits: i32) -> usize {
    if (state_flag_bits & state_flags::GEMMA3N_ALTUP_SIDEBAND) != 0 {
        4
    } else if (state_flag_bits
        & (state_flags::INKLING_MTP_EMBD_SIDEBAND | state_flags::RWKV7_V_FIRST_SIDEBAND))
        != 0
    {
        2
    } else {
        1
    }
}
