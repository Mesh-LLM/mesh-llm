//! Immutable connection-scoped activation layouts. Native graph reuse is independent.
use std::io::{self, Cursor, Read};

use serde::{Deserialize, Serialize};

use super::{
    MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_ACTIVATION_DIMS, MAX_STAGE_ACTIVATION_PARTS,
    STAGE_ACTIVATION_FRAME_VERSION, STAGE_ACTIVATION_PART_OPTIONAL, StageActivationDesc,
    StageActivationFrame, StageActivationPartDesc,
    activation::{
        decode_profile_payload, encode_profile_payload, ggml_type_bytes, validate_descriptor,
    },
    invalid_data,
};
use crate::StageActivationCodec;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationDimension {
    Fixed(u64),
    Tokens,
    Dynamic { min: u64, max: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationPartProfile {
    pub identity: [u8; 32],
    pub ggml_type: u32,
    pub rank: u32,
    pub token_axis: u32,
    pub optional: bool,
    pub dimensions: [ActivationDimension; MAX_STAGE_ACTIVATION_DIMS],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationProfile {
    pub id: u32,
    pub producer_stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub frontier_identity: [u8; 32],
    pub max_tokens: u32,
    pub max_sequences: u32,
    pub parts: Vec<ActivationPartProfile>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationAgreement {
    /// Fresh for each connection, never reused with a different table.
    pub generation: [u8; 16],
    pub profiles: Vec<ActivationProfile>,
}

impl ActivationAgreement {
    pub fn validate(&self) -> io::Result<()> {
        if self.generation == [0; 16] || self.profiles.len() > 16 {
            return Err(invalid_data("invalid activation agreement"));
        }
        for (index, profile) in self.profiles.iter().enumerate() {
            if profile.id == 0 || self.profiles[..index].iter().any(|p| p.id == profile.id) {
                return Err(invalid_data("duplicate or zero activation profile ID"));
            }
            profile.validate()?;
        }
        Ok(())
    }

    pub fn wire_bytes(
        &self,
        codec: StageActivationCodec,
        desc: &StageActivationDesc,
    ) -> io::Result<usize> {
        self.validate()?;
        validate_descriptor(desc, None)?;
        for profile in &self.profiles {
            if let Ok(dynamic) = profile.encode_dimensions(desc) {
                let payload = super::activation::activation_frame_wire_bytes(codec, desc)?
                    - super::activation::FRAME_FIXED_HEADER_BYTES
                    - super::activation::PART_HEADER_BYTES * desc.parts.len();
                let total = 20 + dynamic.len() + payload;
                if total > MAX_STAGE_ACTIVATION_BYTES {
                    return Err(invalid_data("activation frame exceeds maximum"));
                }
                return Ok(total);
            }
        }
        Err(invalid_data(
            "activation does not match an admitted profile",
        ))
    }

    /// Reframe existing codec bytes without another lossy encode/decode cycle.
    pub(super) fn encode_existing(
        &self,
        codec: StageActivationCodec,
        encoded: &[u8],
    ) -> io::Result<Vec<u8>> {
        self.validate()?;
        let (desc, offset) = super::activation::read_descriptor(encoded)?;
        if super::activation::activation_frame_wire_bytes(codec, &desc)? != encoded.len() {
            return Err(invalid_data("activation frame wire byte count mismatch"));
        }
        for profile in &self.profiles {
            if let Ok(dynamic) = profile.encode_dimensions(&desc) {
                let mut bytes = Vec::with_capacity(self.wire_bytes(codec, &desc)?);
                bytes.extend_from_slice(&self.generation);
                bytes.extend_from_slice(&profile.id.to_le_bytes());
                bytes.extend_from_slice(&dynamic);
                bytes.extend_from_slice(&encoded[offset..]);
                return Ok(bytes);
            }
        }
        Err(invalid_data(
            "activation does not match an admitted profile",
        ))
    }

    pub fn encode(
        &self,
        codec: StageActivationCodec,
        frame: &StageActivationFrame,
    ) -> io::Result<Vec<u8>> {
        self.validate()?;
        validate_descriptor(&frame.desc, Some(frame.payload.len()))?;
        for profile in &self.profiles {
            if let Ok(dynamic) = profile.encode_dimensions(&frame.desc) {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&self.generation);
                bytes.extend_from_slice(&profile.id.to_le_bytes());
                bytes.extend_from_slice(&dynamic);
                bytes.extend_from_slice(&encode_profile_payload(
                    codec,
                    &frame.desc,
                    &frame.payload,
                )?);
                if bytes.len() > MAX_STAGE_ACTIVATION_BYTES {
                    return Err(invalid_data("activation frame exceeds maximum"));
                }
                return Ok(bytes);
            }
        }
        Err(invalid_data(
            "activation does not match an admitted profile",
        ))
    }

    pub fn decode(
        &self,
        codec: StageActivationCodec,
        bytes: &[u8],
    ) -> io::Result<StageActivationFrame> {
        if bytes.len() > MAX_STAGE_ACTIVATION_BYTES {
            return Err(invalid_data("activation frame exceeds maximum"));
        }
        self.validate()?;
        let mut input = Cursor::new(bytes);
        if read::<16>(&mut input)? != self.generation {
            return Err(invalid_data("stale activation agreement"));
        }
        let id = u32::from_le_bytes(read(&mut input)?);
        let profile = self
            .profiles
            .iter()
            .find(|p| p.id == id)
            .ok_or_else(|| invalid_data("unknown activation profile"))?;
        let desc = profile.decode_dimensions(&mut input)?;
        decode_profile_payload(codec, desc, &bytes[input.position() as usize..])
    }
}

impl ActivationProfile {
    fn validate(&self) -> io::Result<()> {
        if self.producer_stage_index < 0
            || self.layer_start < 0
            || self.layer_end < self.layer_start
            || self.max_tokens == 0
            || self.max_sequences == 0
            || self.parts.is_empty()
            || self.parts.len() > MAX_STAGE_ACTIVATION_PARTS
        {
            return Err(invalid_data("invalid activation profile"));
        }
        for (index, part) in self.parts.iter().enumerate() {
            ggml_type_bytes(part.ggml_type)?;
            if part.rank == 0
                || part.rank as usize > MAX_STAGE_ACTIVATION_DIMS
                || part.token_axis >= part.rank
                || self.parts[..index]
                    .iter()
                    .any(|p| p.identity == part.identity)
            {
                return Err(invalid_data("invalid activation part profile"));
            }
            for (axis, dimension) in part.dimensions.iter().enumerate() {
                let valid = match dimension {
                    ActivationDimension::Tokens => axis == part.token_axis as usize,
                    ActivationDimension::Fixed(value) => {
                        *value > 0
                            && *value <= i64::MAX as u64
                            && axis != part.token_axis as usize
                            && (axis < part.rank as usize || *value == 1)
                    }
                    ActivationDimension::Dynamic { min, max } => {
                        *min > 0
                            && min <= max
                            && *max <= i64::MAX as u64
                            && axis < part.rank as usize
                            && axis != part.token_axis as usize
                    }
                };
                if !valid {
                    return Err(invalid_data("invalid admitted activation dimension"));
                }
            }
        }
        Ok(())
    }

    fn encode_dimensions(&self, desc: &StageActivationDesc) -> io::Result<Vec<u8>> {
        if desc.producer_stage_index != self.producer_stage_index
            || desc.layer_start != self.layer_start
            || desc.layer_end != self.layer_end
            || desc.frontier_identity != self.frontier_identity
        {
            return Err(invalid_data("activation boundary differs from agreement"));
        }
        let mut mask = 0u16;
        let mut values = Vec::new();
        let mut actual = desc.parts.iter().peekable();
        for (index, part) in self.parts.iter().enumerate() {
            let Some(value) = actual.peek().filter(|p| p.identity == part.identity) else {
                if part.optional {
                    continue;
                }
                return Err(invalid_data("required activation part absent"));
            };
            if value.ggml_type != part.ggml_type
                || value.rank != part.rank
                || value.token_axis != part.token_axis as i32
                || value.flags
                    != if part.optional {
                        STAGE_ACTIVATION_PART_OPTIONAL
                    } else {
                        0
                    }
            {
                return Err(invalid_data("activation part differs from agreement"));
            }
            mask |= 1 << index;
            for (axis, dimension) in part.dimensions.iter().enumerate().take(part.rank as usize) {
                let got = u64::try_from(value.dimensions[axis])
                    .map_err(|_| invalid_data("negative activation dimension"))?;
                match dimension {
                    ActivationDimension::Fixed(expected) if got != *expected => {
                        return Err(invalid_data("fixed activation dimension differs"));
                    }
                    ActivationDimension::Tokens if got != u64::from(desc.token_count) => {
                        return Err(invalid_data("token dimension differs"));
                    }
                    ActivationDimension::Dynamic { min, max } => {
                        if got < *min || got > *max {
                            return Err(invalid_data(
                                "dynamic activation dimension outside bounds",
                            ));
                        }
                        values.extend_from_slice(&got.to_le_bytes());
                    }
                    _ => {}
                }
            }
            actual.next();
        }
        if actual.next().is_some() {
            return Err(invalid_data("unadmitted activation part"));
        }
        let mut output = Vec::new();
        output.extend_from_slice(&desc.token_count.to_le_bytes());
        output.extend_from_slice(&desc.sequence_count.to_le_bytes());
        output.extend_from_slice(&mask.to_le_bytes());
        output.extend_from_slice(&values);
        // Use the same checked layout construction and bounds on both endpoints.
        self.decode_dimensions(&mut Cursor::new(&output))?;
        Ok(output)
    }

    fn decode_dimensions(&self, input: &mut impl Read) -> io::Result<StageActivationDesc> {
        let token_count = u32::from_le_bytes(read(input)?);
        let sequence_count = u32::from_le_bytes(read(input)?);
        let mask = u16::from_le_bytes(read(input)?);
        if token_count == 0
            || token_count > self.max_tokens
            || sequence_count == 0
            || sequence_count > self.max_sequences
            || mask == 0
            || u32::from(mask) >> self.parts.len() != 0
        {
            return Err(invalid_data(
                "activation counts or presence outside agreement",
            ));
        }
        let mut desc = StageActivationDesc {
            version: STAGE_ACTIVATION_FRAME_VERSION,
            producer_stage_index: self.producer_stage_index,
            layer_start: self.layer_start,
            layer_end: self.layer_end,
            token_count,
            sequence_count,
            payload_bytes: 0,
            frontier_identity: self.frontier_identity,
            parts: Vec::new(),
        };
        for (index, profile) in self.parts.iter().enumerate() {
            if mask & (1 << index) == 0 {
                if !profile.optional {
                    return Err(invalid_data("required activation part absent"));
                }
                continue;
            }
            let mut part = StageActivationPartDesc {
                identity: profile.identity,
                ggml_type: profile.ggml_type,
                rank: profile.rank,
                token_axis: profile.token_axis as i32,
                flags: if profile.optional {
                    STAGE_ACTIVATION_PART_OPTIONAL
                } else {
                    0
                },
                payload_offset: desc.payload_bytes,
                ..Default::default()
            };
            let mut stride = ggml_type_bytes(part.ggml_type)? as u64;
            for (axis, dimension) in profile.dimensions.iter().enumerate() {
                let value = match dimension {
                    ActivationDimension::Fixed(value) => *value,
                    ActivationDimension::Tokens => u64::from(token_count),
                    ActivationDimension::Dynamic { min, max } => {
                        let value = u64::from_le_bytes(read(input)?);
                        if value < *min || value > *max {
                            return Err(invalid_data(
                                "dynamic activation dimension outside bounds",
                            ));
                        }
                        value
                    }
                };
                part.dimensions[axis] = i64::try_from(value)
                    .map_err(|_| invalid_data("activation dimension overflow"))?;
                part.byte_strides[axis] = stride;
                stride = stride
                    .checked_mul(value)
                    .ok_or_else(|| invalid_data("activation layout overflow"))?;
            }
            part.payload_bytes = stride;
            desc.payload_bytes = desc
                .payload_bytes
                .checked_add(stride)
                .ok_or_else(|| invalid_data("activation layout overflow"))?;
            desc.parts.push(part);
        }
        validate_descriptor(&desc, None)?;
        Ok(desc)
    }
}

fn read<const N: usize>(input: &mut impl Read) -> io::Result<[u8; N]> {
    let mut value = [0; N];
    input.read_exact(&mut value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn agreement() -> ActivationAgreement {
        let part = ActivationPartProfile {
            identity: [3; 32],
            ggml_type: 0,
            rank: 3,
            token_axis: 1,
            optional: false,
            dimensions: [
                ActivationDimension::Fixed(2),
                ActivationDimension::Tokens,
                ActivationDimension::Dynamic { min: 1, max: 4 },
                ActivationDimension::Fixed(1),
            ],
        };
        let mut optional = part.clone();
        optional.identity = [4; 32];
        optional.optional = true;
        optional.ggml_type = 26;
        ActivationAgreement {
            generation: [9; 16],
            profiles: vec![ActivationProfile {
                id: 1,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 4,
                frontier_identity: [5; 32],
                max_tokens: 17,
                max_sequences: 8,
                parts: vec![part, optional],
            }],
        }
    }
    fn frame(
        a: &ActivationAgreement,
        tokens: u32,
        extra: u64,
        optional: bool,
    ) -> StageActivationFrame {
        let mut dimensions = Vec::new();
        dimensions.extend_from_slice(&tokens.to_le_bytes());
        dimensions.extend_from_slice(&1u32.to_le_bytes());
        dimensions.extend_from_slice(&(if optional { 3u16 } else { 1u16 }).to_le_bytes());
        dimensions.extend_from_slice(&extra.to_le_bytes());
        if optional {
            dimensions.extend_from_slice(&extra.to_le_bytes());
        }
        let desc = a.profiles[0]
            .decode_dimensions(&mut Cursor::new(dimensions))
            .unwrap();
        let payload = vec![0; desc.payload_bytes as usize];
        StageActivationFrame { desc, payload }
    }
    #[test]
    fn one_agreement_preserves_variable_batches_non_token_dimensions_and_optional_parts() {
        let a = agreement();
        a.validate().unwrap();
        for tokens in [1, 3, 17] {
            for extra in [1, 4] {
                for optional in [false, true] {
                    let frame = frame(&a, tokens, extra, optional);
                    for codec in [
                        StageActivationCodec::RawF32V1,
                        StageActivationCodec::F16RneV1,
                        StageActivationCodec::Bf16RneV1,
                        StageActivationCodec::S8RowF32RneV1,
                    ] {
                        let bytes = a.encode(codec, &frame).unwrap();
                        assert_eq!(bytes.len(), a.wire_bytes(codec, &frame.desc).unwrap());
                        assert_eq!(a.decode(codec, &bytes).unwrap(), frame);
                        let internal = super::super::activation::encode_activation_frame(
                            codec,
                            &frame.desc,
                            &frame.payload,
                        )
                        .unwrap();
                        assert_eq!(a.encode_existing(codec, &internal).unwrap(), bytes);
                        assert!(
                            a.decode(codec, &internal).is_err(),
                            "full descriptors are not a wire fallback"
                        );
                        assert!(bytes.len() < internal.len());
                    }
                }
            }
        }
    }
    #[test]
    fn malformed_or_stale_references_fail_before_payload_decode() {
        let a = agreement();
        let bytes = a
            .encode(StageActivationCodec::RawF32V1, &frame(&a, 3, 2, false))
            .unwrap();
        for (offset, replacement) in [(0, 0u8), (16, 2), (20, 18), (28, 0), (30, 5)] {
            let mut bad = bytes.clone();
            bad[offset] = replacement;
            assert!(
                a.decode(StageActivationCodec::RawF32V1, &bad).is_err(),
                "offset {offset}"
            );
        }
        for length in 0..bytes.len() {
            assert!(
                a.decode(StageActivationCodec::RawF32V1, &bytes[..length])
                    .is_err()
            );
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(a.decode(StageActivationCodec::RawF32V1, &trailing).is_err());
    }
    #[test]
    fn invalid_profiles_are_rejected_at_admission() {
        let mut a = agreement();
        a.profiles[0].parts[0].dimensions[1] = ActivationDimension::Fixed(3);
        assert!(a.validate().is_err());
        let mut a = agreement();
        a.profiles.push(a.profiles[0].clone());
        assert!(a.validate().is_err());
        let mut a = agreement();
        a.profiles[0].parts[0].dimensions[2] = ActivationDimension::Dynamic { min: 4, max: 1 };
        assert!(a.validate().is_err());
    }
}
