use std::io::{self, Cursor, Read};

use crate::StageActivationCodec;

use super::{
    activation_codec::{ActivationShape, decode_activation, encode_activation, encoded_len},
    invalid_data,
    types::{
        MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_ACTIVATION_DIMS, MAX_STAGE_ACTIVATION_PARTS,
        MAX_STAGE_DECODED_ACTIVATION_BYTES, STAGE_ACTIVATION_FRAME_VERSION,
        STAGE_ACTIVATION_IDENTITY_BYTES, STAGE_ACTIVATION_PART_OPTIONAL, StageActivationDesc,
        StageActivationFrame, StageActivationPartDesc,
    },
};

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_I32: u32 = 26;
const GGML_TYPE_BF16: u32 = 30;
const FRAME_FIXED_HEADER_BYTES: usize = 68;
const PART_HEADER_BYTES: usize = 128;

pub fn activation_frame_wire_bytes(
    codec: StageActivationCodec,
    desc: &StageActivationDesc,
) -> io::Result<usize> {
    validate_descriptor(desc, None)?;
    let header_bytes = FRAME_FIXED_HEADER_BYTES
        .checked_add(
            desc.parts
                .len()
                .checked_mul(PART_HEADER_BYTES)
                .ok_or_else(|| invalid_data("activation frame header size overflow"))?,
        )
        .ok_or_else(|| invalid_data("activation frame header size overflow"))?;
    let mut total = header_bytes;
    for part in &desc.parts {
        total = total
            .checked_add(part_wire_bytes(codec, part)?)
            .ok_or_else(|| invalid_data("activation frame wire size overflow"))?;
    }
    if total > MAX_STAGE_ACTIVATION_BYTES {
        return Err(invalid_data(
            "activation frame wire byte count exceeds maximum",
        ));
    }
    Ok(total)
}

pub fn encode_activation_frame(
    codec: StageActivationCodec,
    desc: &StageActivationDesc,
    payload: &[u8],
) -> io::Result<Vec<u8>> {
    validate_descriptor(desc, Some(payload.len()))?;
    let mut encoded = Vec::with_capacity(activation_frame_wire_bytes(codec, desc)?);
    write_descriptor(&mut encoded, desc)?;
    for part in &desc.parts {
        let bytes = part_payload(part, payload)?;
        if part.ggml_type == GGML_TYPE_F32 {
            let values = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>();
            encoded.extend_from_slice(&encode_activation(codec, part_shape(part)?, &values)?);
        } else {
            encoded.extend_from_slice(bytes);
        }
    }
    Ok(encoded)
}

pub fn decode_activation_frame(
    codec: StageActivationCodec,
    encoded: &[u8],
) -> io::Result<StageActivationFrame> {
    let (desc, header_bytes) = read_descriptor(encoded)?;
    validate_descriptor(&desc, None)?;
    if activation_frame_wire_bytes(codec, &desc)? != encoded.len() {
        return Err(invalid_data("activation frame wire byte count mismatch"));
    }
    let mut wire_offset = header_bytes;
    let mut payload = Vec::with_capacity(
        usize::try_from(desc.payload_bytes)
            .map_err(|_| invalid_data("activation payload byte count exceeds usize"))?,
    );
    for part in &desc.parts {
        let wire_bytes = part_wire_bytes(codec, part)?;
        let end = wire_offset
            .checked_add(wire_bytes)
            .ok_or_else(|| invalid_data("activation part wire range overflow"))?;
        let bytes = encoded
            .get(wire_offset..end)
            .ok_or_else(|| invalid_data("activation part wire range exceeds frame"))?;
        if part.ggml_type == GGML_TYPE_F32 {
            for value in decode_activation(codec, part_shape(part)?, bytes)? {
                payload.extend_from_slice(&value.to_le_bytes());
            }
        } else {
            payload.extend_from_slice(bytes);
        }
        wire_offset = end;
    }
    validate_descriptor(&desc, Some(payload.len()))?;
    Ok(StageActivationFrame { desc, payload })
}

pub fn encode_raw_activation_frame(frame: &StageActivationFrame) -> io::Result<Vec<u8>> {
    encode_activation_frame(StageActivationCodec::RawF32V1, &frame.desc, &frame.payload)
}

pub fn decode_raw_activation_frame(encoded: &[u8]) -> io::Result<StageActivationFrame> {
    decode_activation_frame(StageActivationCodec::RawF32V1, encoded)
}

pub fn select_lossless_activation_codec(
    desc: &StageActivationDesc,
    payload: &[u8],
    permitted_codecs: &[StageActivationCodec],
) -> io::Result<StageActivationCodec> {
    validate_descriptor(desc, Some(payload.len()))?;
    if !permitted_codecs.contains(&StageActivationCodec::RawF32V1) {
        return Err(invalid_data(
            "lossless activation selection requires RawF32 fallback",
        ));
    }

    let preference = [
        StageActivationCodec::Bf16RneV1,
        StageActivationCodec::F16RneV1,
        StageActivationCodec::S8RowF32RneV1,
        StageActivationCodec::RawF32V1,
    ];
    let mut selected = StageActivationCodec::RawF32V1;
    let mut selected_bytes = activation_frame_wire_bytes(selected, desc)?;
    for codec in preference {
        if !permitted_codecs.contains(&codec) || !codec_is_lossless(codec, desc, payload)? {
            continue;
        }
        let wire_bytes = activation_frame_wire_bytes(codec, desc)?;
        if wire_bytes < selected_bytes {
            selected = codec;
            selected_bytes = wire_bytes;
        }
    }
    Ok(selected)
}

fn codec_is_lossless(
    codec: StageActivationCodec,
    desc: &StageActivationDesc,
    payload: &[u8],
) -> io::Result<bool> {
    for part in &desc.parts {
        if part.ggml_type != GGML_TYPE_F32 {
            continue;
        }
        let bytes = part_payload(part, payload)?;
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        let encoded = match encode_activation(codec, part_shape(part)?, &values) {
            Ok(encoded) => encoded,
            Err(_) => return Ok(false),
        };
        let decoded = match decode_activation(codec, part_shape(part)?, &encoded) {
            Ok(decoded) => decoded,
            Err(_) => return Ok(false),
        };
        if decoded
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .ne(bytes.iter().copied())
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn validate_descriptor(desc: &StageActivationDesc, payload_len: Option<usize>) -> io::Result<()> {
    if desc.version != STAGE_ACTIVATION_FRAME_VERSION {
        return Err(invalid_data("unsupported activation frame version"));
    }
    if desc.parts.is_empty() || desc.parts.len() > MAX_STAGE_ACTIVATION_PARTS {
        return Err(invalid_data("activation frame part count is invalid"));
    }
    let expected_payload_bytes = usize::try_from(desc.payload_bytes)
        .map_err(|_| invalid_data("activation payload byte count exceeds usize"))?;
    if expected_payload_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
        return Err(invalid_data(
            "decoded activation payload byte count exceeds maximum",
        ));
    }
    if payload_len.is_some_and(|actual| actual != expected_payload_bytes) {
        return Err(invalid_data("activation payload byte count mismatch"));
    }

    let mut payload_offset = 0usize;
    for (index, part) in desc.parts.iter().enumerate() {
        if part.flags & !STAGE_ACTIVATION_PART_OPTIONAL != 0 {
            return Err(invalid_data("activation part has unsupported flags"));
        }
        if desc.parts[..index]
            .iter()
            .any(|candidate| candidate.identity == part.identity)
        {
            return Err(invalid_data(
                "activation frame has duplicate part identities",
            ));
        }
        let rank = usize::try_from(part.rank)
            .map_err(|_| invalid_data("activation part rank exceeds usize"))?;
        let token_axis = usize::try_from(part.token_axis)
            .map_err(|_| invalid_data("activation part has a negative token axis"))?;
        if rank == 0 || rank > MAX_STAGE_ACTIVATION_DIMS || token_axis >= rank {
            return Err(invalid_data("activation part shape is invalid"));
        }
        if part.dimensions[token_axis] != i64::from(desc.token_count) {
            return Err(invalid_data(
                "activation part token dimension does not match frame",
            ));
        }
        let element_bytes = ggml_type_bytes(part.ggml_type)?;
        let mut elements = 1usize;
        let mut expected_stride = element_bytes;
        for axis in 0..rank {
            let dimension = usize::try_from(part.dimensions[axis])
                .map_err(|_| invalid_data("activation part dimension is not positive"))?;
            if dimension == 0 {
                return Err(invalid_data("activation part dimension is not positive"));
            }
            if usize::try_from(part.byte_strides[axis])
                .map_err(|_| invalid_data("activation part stride exceeds usize"))?
                != expected_stride
            {
                return Err(invalid_data("activation part is not densely packed"));
            }
            elements = elements
                .checked_mul(dimension)
                .ok_or_else(|| invalid_data("activation part element count overflow"))?;
            expected_stride = expected_stride
                .checked_mul(dimension)
                .ok_or_else(|| invalid_data("activation part stride overflow"))?;
        }
        let part_bytes = elements
            .checked_mul(element_bytes)
            .ok_or_else(|| invalid_data("activation part byte count overflow"))?;
        if usize::try_from(part.payload_offset)
            .map_err(|_| invalid_data("activation part offset exceeds usize"))?
            != payload_offset
            || usize::try_from(part.payload_bytes)
                .map_err(|_| invalid_data("activation part byte count exceeds usize"))?
                != part_bytes
        {
            return Err(invalid_data(
                "activation part payload layout is inconsistent",
            ));
        }
        payload_offset = payload_offset
            .checked_add(part_bytes)
            .ok_or_else(|| invalid_data("activation payload byte count overflow"))?;
    }
    if payload_offset != expected_payload_bytes {
        return Err(invalid_data("activation descriptor payload size mismatch"));
    }
    Ok(())
}

fn ggml_type_bytes(ggml_type: u32) -> io::Result<usize> {
    match ggml_type {
        GGML_TYPE_F32 | GGML_TYPE_I32 => Ok(4),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => Ok(2),
        _ => Err(invalid_data("activation part has unsupported ggml type")),
    }
}

fn part_shape(part: &StageActivationPartDesc) -> io::Result<ActivationShape> {
    let elements = usize::try_from(part.payload_bytes)
        .map_err(|_| invalid_data("activation part byte count exceeds usize"))?
        .checked_div(4)
        .ok_or_else(|| invalid_data("activation part shape is invalid"))?;
    let columns = usize::try_from(part.dimensions[0])
        .map_err(|_| invalid_data("activation part first dimension exceeds usize"))?;
    if columns == 0 || !elements.is_multiple_of(columns) {
        return Err(invalid_data("activation part shape is invalid"));
    }
    Ok(ActivationShape::new(elements / columns, 0, columns))
}

fn part_wire_bytes(
    codec: StageActivationCodec,
    part: &StageActivationPartDesc,
) -> io::Result<usize> {
    if part.ggml_type == GGML_TYPE_F32 {
        encoded_len(codec, part_shape(part)?)
    } else {
        usize::try_from(part.payload_bytes)
            .map_err(|_| invalid_data("activation part byte count exceeds usize"))
    }
}

fn part_payload<'a>(part: &StageActivationPartDesc, payload: &'a [u8]) -> io::Result<&'a [u8]> {
    let start = usize::try_from(part.payload_offset)
        .map_err(|_| invalid_data("activation part offset exceeds usize"))?;
    let bytes = usize::try_from(part.payload_bytes)
        .map_err(|_| invalid_data("activation part byte count exceeds usize"))?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| invalid_data("activation part payload range overflow"))?;
    payload
        .get(start..end)
        .ok_or_else(|| invalid_data("activation part payload range exceeds frame"))
}

fn write_descriptor(output: &mut Vec<u8>, desc: &StageActivationDesc) -> io::Result<()> {
    output.extend_from_slice(&desc.version.to_le_bytes());
    output.extend_from_slice(&desc.producer_stage_index.to_le_bytes());
    output.extend_from_slice(&desc.layer_start.to_le_bytes());
    output.extend_from_slice(&desc.layer_end.to_le_bytes());
    output.extend_from_slice(&desc.token_count.to_le_bytes());
    output.extend_from_slice(&desc.sequence_count.to_le_bytes());
    output.extend_from_slice(
        &u32::try_from(desc.parts.len())
            .map_err(|_| invalid_data("activation frame part count exceeds u32"))?
            .to_le_bytes(),
    );
    output.extend_from_slice(&desc.payload_bytes.to_le_bytes());
    output.extend_from_slice(&desc.frontier_identity);
    for part in &desc.parts {
        output.extend_from_slice(&part.identity);
        output.extend_from_slice(&part.ggml_type.to_le_bytes());
        output.extend_from_slice(&part.rank.to_le_bytes());
        output.extend_from_slice(&part.token_axis.to_le_bytes());
        output.extend_from_slice(&part.flags.to_le_bytes());
        for dimension in part.dimensions {
            output.extend_from_slice(&dimension.to_le_bytes());
        }
        for stride in part.byte_strides {
            output.extend_from_slice(&stride.to_le_bytes());
        }
        output.extend_from_slice(&part.payload_offset.to_le_bytes());
        output.extend_from_slice(&part.payload_bytes.to_le_bytes());
    }
    Ok(())
}

fn read_descriptor(encoded: &[u8]) -> io::Result<(StageActivationDesc, usize)> {
    let mut reader = Cursor::new(encoded);
    let version = read_u32(&mut reader)?;
    let producer_stage_index = read_i32(&mut reader)?;
    let layer_start = read_i32(&mut reader)?;
    let layer_end = read_i32(&mut reader)?;
    let token_count = read_u32(&mut reader)?;
    let sequence_count = read_u32(&mut reader)?;
    let part_count = usize::try_from(read_u32(&mut reader)?)
        .map_err(|_| invalid_data("activation frame part count exceeds usize"))?;
    if part_count == 0 || part_count > MAX_STAGE_ACTIVATION_PARTS {
        return Err(invalid_data("activation frame part count is invalid"));
    }
    let payload_bytes = read_u64(&mut reader)?;
    let frontier_identity = read_array::<STAGE_ACTIVATION_IDENTITY_BYTES>(&mut reader)?;
    let mut parts = Vec::with_capacity(part_count);
    for _ in 0..part_count {
        let identity = read_array::<STAGE_ACTIVATION_IDENTITY_BYTES>(&mut reader)?;
        let ggml_type = read_u32(&mut reader)?;
        let rank = read_u32(&mut reader)?;
        let token_axis = read_i32(&mut reader)?;
        let flags = read_u32(&mut reader)?;
        let mut dimensions = [0_i64; MAX_STAGE_ACTIVATION_DIMS];
        for dimension in &mut dimensions {
            *dimension = read_i64(&mut reader)?;
        }
        let mut byte_strides = [0_u64; MAX_STAGE_ACTIVATION_DIMS];
        for stride in &mut byte_strides {
            *stride = read_u64(&mut reader)?;
        }
        parts.push(StageActivationPartDesc {
            identity,
            ggml_type,
            rank,
            token_axis,
            flags,
            dimensions,
            byte_strides,
            payload_offset: read_u64(&mut reader)?,
            payload_bytes: read_u64(&mut reader)?,
        });
    }
    let header_bytes = usize::try_from(reader.position())
        .map_err(|_| invalid_data("activation frame header position exceeds usize"))?;
    Ok((
        StageActivationDesc {
            version,
            producer_stage_index,
            layer_start,
            layer_end,
            token_count,
            sequence_count,
            payload_bytes,
            frontier_identity,
            parts,
        },
        header_bytes,
    ))
}

fn read_array<const N: usize>(reader: &mut impl Read) -> io::Result<[u8; N]> {
    let mut bytes = [0_u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_array(reader)?))
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    Ok(i32::from_le_bytes(read_array(reader)?))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_array(reader)?))
}

fn read_i64(reader: &mut impl Read) -> io::Result<i64> {
    Ok(i64::from_le_bytes(read_array(reader)?))
}
