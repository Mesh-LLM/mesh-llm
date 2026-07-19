use std::ffi::c_void;
use std::ptr;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::{
    ActivationDType, ActivationDesc as RawActivationDesc, ActivationLayout,
    NativeMtpDraft as RawNativeMtpDraft, SamplingConfig as RawSamplingConfig,
};

use crate::error::{ensure_ok, free_error};
use crate::session::StageSession;
use crate::types::empty_raw_activation_desc;
use crate::{ActivationFrame, DecodeFrameBatchOutput, NativeMtpDraft, SamplingConfig, Status};

type RawInputFrame = (Option<RawActivationDesc>, *const c_void);

fn raw_input_frame(input: Option<&ActivationFrame>) -> Result<RawInputFrame> {
    let Some(frame) = input else {
        return Ok((None, ptr::null()));
    };
    frame.validate_payload_len()?;
    Ok((Some(frame.desc.as_raw()), frame.payload.as_ptr().cast()))
}

fn raw_input_desc_ptr(input: &RawInputFrame) -> *const RawActivationDesc {
    input
        .0
        .as_ref()
        .map_or(ptr::null(), |desc| desc as *const RawActivationDesc)
}

pub struct DecodeFrameBatchRequest<'a> {
    pub session: &'a mut StageSession,
    pub token_id: i32,
    pub sampling: Option<&'a SamplingConfig>,
    pub input: Option<&'a ActivationFrame>,
}

impl StageSession {
    pub fn prefill_chunk_frame(
        &mut self,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<ActivationFrame> {
        let (output_desc, output_payload) =
            self.prefill_chunk_frame_raw(token_ids, &[], input, output_capacity)?;
        Ok(ActivationFrame {
            desc: output_desc.into(),
            payload: output_payload,
        })
    }

    pub fn prefill_chunk_frame_with_positions(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<ActivationFrame> {
        let (output_desc, output_payload) =
            self.prefill_chunk_frame_raw(token_ids, positions, input, output_capacity)?;
        Ok(ActivationFrame {
            desc: output_desc.into(),
            payload: output_payload,
        })
    }

    fn prefill_chunk_frame_raw(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(RawActivationDesc, Vec<u8>)> {
        let raw_input = raw_input_frame(input)?;
        let input_desc_ptr = raw_input_desc_ptr(&raw_input);
        let input_payload_ptr = raw_input.1;
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            if positions.is_empty() {
                skippy_ffi::skippy_prefill_chunk_frame(
                    self.raw,
                    token_ids.as_ptr(),
                    token_ids.len(),
                    input_desc_ptr,
                    input_payload_ptr,
                    &mut output_desc,
                    output_payload.as_mut_ptr().cast(),
                    output_payload.len(),
                    &mut output_bytes,
                    &mut error,
                )
            } else {
                skippy_ffi::skippy_prefill_chunk_frame_with_positions(
                    self.raw,
                    token_ids.as_ptr(),
                    token_ids.len(),
                    positions.as_ptr(),
                    positions.len(),
                    input_desc_ptr,
                    input_payload_ptr,
                    &mut output_desc,
                    output_payload.as_mut_ptr().cast(),
                    output_payload.len(),
                    &mut output_bytes,
                    &mut error,
                )
            }
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.prefill_chunk_frame_raw(token_ids, positions, input, output_bytes);
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        Ok((output_desc, output_payload))
    }

    pub fn prefill_chunk_frame_sampled(
        &mut self,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        let (predicted_token, output_desc, output_payload) =
            self.prefill_chunk_frame_sampled_raw(token_ids, &[], sampling, input, output_capacity)?;
        Ok((
            predicted_token,
            ActivationFrame {
                desc: output_desc.into(),
                payload: output_payload,
            },
        ))
    }

    pub fn prefill_chunk_frame_sampled_with_positions(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        let (predicted_token, output_desc, output_payload) = self.prefill_chunk_frame_sampled_raw(
            token_ids,
            positions,
            sampling,
            input,
            output_capacity,
        )?;
        Ok((
            predicted_token,
            ActivationFrame {
                desc: output_desc.into(),
                payload: output_payload,
            },
        ))
    }

    fn prefill_chunk_frame_sampled_raw(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, RawActivationDesc, Vec<u8>)> {
        let raw_input = raw_input_frame(input)?;
        let input_desc_ptr = raw_input_desc_ptr(&raw_input);
        let input_payload_ptr = raw_input.1;
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut predicted_token = 0_i32;
        let mut error = ptr::null_mut();
        let status = unsafe {
            if positions.is_empty() {
                skippy_ffi::skippy_prefill_chunk_frame_sampled(
                    self.raw,
                    token_ids.as_ptr(),
                    token_ids.len(),
                    sampling_ptr,
                    input_desc_ptr,
                    input_payload_ptr,
                    &mut output_desc,
                    output_payload.as_mut_ptr().cast(),
                    output_payload.len(),
                    &mut output_bytes,
                    &mut predicted_token,
                    &mut error,
                )
            } else {
                skippy_ffi::skippy_prefill_chunk_frame_sampled_with_positions(
                    self.raw,
                    token_ids.as_ptr(),
                    token_ids.len(),
                    positions.as_ptr(),
                    positions.len(),
                    sampling_ptr,
                    input_desc_ptr,
                    input_payload_ptr,
                    &mut output_desc,
                    output_payload.as_mut_ptr().cast(),
                    output_payload.len(),
                    &mut output_bytes,
                    &mut predicted_token,
                    &mut error,
                )
            }
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.prefill_chunk_frame_sampled_raw(
                token_ids,
                positions,
                sampling,
                input,
                output_bytes,
            );
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        Ok((predicted_token, output_desc, output_payload))
    }

    pub fn decode_step_frame(
        &mut self,
        token_id: i32,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        self.decode_step_frame_sampled(token_id, None, input, output_capacity)
    }

    pub fn decode_step_frame_sampled(
        &mut self,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        let (predicted_token, output_desc, output_payload) =
            self.decode_step_frame_raw(token_id, sampling, input, output_capacity)?;
        Ok((
            predicted_token,
            ActivationFrame {
                desc: output_desc.into(),
                payload: output_payload,
            },
        ))
    }

    pub fn decode_step_frame_sampled_mtp(
        &mut self,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
        max_draft_tokens: usize,
    ) -> Result<(i32, Option<NativeMtpDraft>, ActivationFrame)> {
        let (predicted_token, mtp_draft, output_desc, output_payload) = self
            .decode_step_frame_mtp_raw(
                token_id,
                sampling,
                input,
                output_capacity,
                max_draft_tokens,
            )?;
        Ok((
            predicted_token,
            mtp_draft,
            ActivationFrame {
                desc: output_desc.into(),
                payload: output_payload,
            },
        ))
    }

    fn decode_step_frame_raw(
        &mut self,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, RawActivationDesc, Vec<u8>)> {
        let raw_input = raw_input_frame(input)?;
        let input_desc_ptr = raw_input_desc_ptr(&raw_input);
        let input_payload_ptr = raw_input.1;
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut predicted_token = 0_i32;
        let mut error = ptr::null_mut();
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let status = unsafe {
            skippy_ffi::skippy_decode_step_frame_sampled(
                self.raw,
                token_id,
                sampling_ptr,
                input_desc_ptr,
                input_payload_ptr,
                &mut output_desc,
                output_payload.as_mut_ptr().cast(),
                output_payload.len(),
                &mut output_bytes,
                &mut predicted_token,
                &mut error,
            )
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.decode_step_frame_raw(token_id, sampling, input, output_bytes);
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(1)
            .context("session token count overflow")?;
        Ok((predicted_token, output_desc, output_payload))
    }

    fn decode_step_frame_mtp_raw(
        &mut self,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
        max_draft_tokens: usize,
    ) -> Result<(i32, Option<NativeMtpDraft>, RawActivationDesc, Vec<u8>)> {
        let raw_input = raw_input_frame(input)?;
        let input_desc_ptr = raw_input_desc_ptr(&raw_input);
        let input_payload_ptr = raw_input.1;
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut predicted_token = 0_i32;
        let mut mtp_draft = RawNativeMtpDraft::default();
        let mut error = ptr::null_mut();
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let status = unsafe {
            skippy_ffi::skippy_decode_step_frame_sampled_mtp(
                self.raw,
                token_id,
                sampling_ptr,
                input_desc_ptr,
                input_payload_ptr,
                &mut output_desc,
                output_payload.as_mut_ptr().cast(),
                output_payload.len(),
                &mut output_bytes,
                &mut predicted_token,
                max_draft_tokens.min(skippy_ffi::NATIVE_MTP_MAX_DRAFT_TOKENS),
                &mut mtp_draft,
                &mut error,
            )
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.decode_step_frame_mtp_raw(
                token_id,
                sampling,
                input,
                output_bytes,
                max_draft_tokens,
            );
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(1)
            .context("session token count overflow")?;
        Ok((
            predicted_token,
            NativeMtpDraft::from_raw(mtp_draft),
            output_desc,
            output_payload,
        ))
    }

    pub fn decode_step_frame_batch_sampled(
        requests: &mut [DecodeFrameBatchRequest<'_>],
    ) -> Result<Vec<DecodeFrameBatchOutput>> {
        Self::decode_step_frame_batch_sampled_raw(requests, &vec![0; requests.len()])
    }

    fn decode_step_frame_batch_sampled_raw(
        requests: &mut [DecodeFrameBatchRequest<'_>],
        output_capacities: &[usize],
    ) -> Result<Vec<DecodeFrameBatchOutput>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let sessions = requests
            .iter_mut()
            .map(|request| request.session.raw)
            .collect::<Vec<_>>();
        let token_ids = requests
            .iter()
            .map(|request| request.token_id)
            .collect::<Vec<_>>();
        let raw_sampling = requests
            .iter()
            .map(|request| request.sampling.map(SamplingConfig::as_raw))
            .collect::<Vec<_>>();
        let sampling = raw_sampling
            .iter()
            .map(|sampling| {
                sampling
                    .as_ref()
                    .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig)
            })
            .collect::<Vec<_>>();
        let input_frames = requests
            .iter()
            .map(|request| raw_input_frame(request.input))
            .collect::<Result<Vec<_>>>()?;
        let input_desc_ptrs = input_frames
            .iter()
            .map(raw_input_desc_ptr)
            .collect::<Vec<_>>();
        let input_payloads = input_frames.iter().map(|input| input.1).collect::<Vec<_>>();
        let mut output_descs = vec![empty_raw_activation_desc(); requests.len()];
        let mut output_payloads = output_capacities
            .iter()
            .map(|capacity| vec![0_u8; *capacity])
            .collect::<Vec<_>>();
        let output_payload_ptrs = output_payloads
            .iter_mut()
            .map(|payload| payload.as_mut_ptr().cast())
            .collect::<Vec<_>>();
        let mut output_bytes = vec![0_usize; requests.len()];
        let mut predicted_tokens = vec![0_i32; requests.len()];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_decode_step_frame_batch_sampled(
                sessions.as_ptr(),
                token_ids.as_ptr(),
                sampling.as_ptr(),
                input_desc_ptrs.as_ptr(),
                input_payloads.as_ptr(),
                output_descs.as_mut_ptr(),
                output_payload_ptrs.as_ptr(),
                output_capacities.as_ptr(),
                output_bytes.as_mut_ptr(),
                predicted_tokens.as_mut_ptr(),
                predicted_tokens.len(),
                requests.len(),
                &mut error,
            )
        };
        if status == Status::BufferTooSmall {
            free_error(error);
            error = ptr::null_mut();
            if output_bytes
                .iter()
                .zip(output_capacities.iter())
                .any(|(required, capacity)| required > capacity)
            {
                return Self::decode_step_frame_batch_sampled_raw(requests, &output_bytes);
            }
        }
        if status == Status::Unsupported {
            free_error(error);
            return Self::decode_step_frame_batch_sampled_serial(requests);
        }
        ensure_ok(status, error)?;
        for request in requests.iter_mut() {
            request.session.token_count = request
                .session
                .token_count
                .checked_add(1)
                .context("session token count overflow")?;
        }
        Ok(output_payloads
            .into_iter()
            .zip(output_descs)
            .zip(output_bytes)
            .zip(predicted_tokens)
            .map(|(((mut payload, desc), bytes), predicted_token)| {
                payload.truncate(bytes);
                DecodeFrameBatchOutput {
                    predicted_token,
                    output: ActivationFrame {
                        desc: desc.into(),
                        payload,
                    },
                }
            })
            .collect())
    }

    fn decode_step_frame_batch_sampled_serial(
        requests: &mut [DecodeFrameBatchRequest<'_>],
    ) -> Result<Vec<DecodeFrameBatchOutput>> {
        requests
            .iter_mut()
            .map(|request| {
                let (predicted_token, output) = request.session.decode_step_frame_sampled(
                    request.token_id,
                    request.sampling,
                    request.input,
                    0,
                )?;
                Ok(DecodeFrameBatchOutput {
                    predicted_token,
                    output,
                })
            })
            .collect()
    }

    pub fn verify_tokens_frame(
        &mut self,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(Vec<i32>, ActivationFrame)> {
        self.verify_tokens_frame_sampled(token_ids, None, input, output_capacity)
    }

    pub fn verify_tokens_frame_sampled(
        &mut self,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(Vec<i32>, ActivationFrame)> {
        if token_ids.is_empty() {
            return Err(anyhow!("verify_tokens_frame requires at least one token"));
        }
        let (predicted_tokens, output_desc, output_payload) =
            self.verify_tokens_frame_raw(token_ids, sampling, input, output_capacity)?;
        Ok((
            predicted_tokens,
            ActivationFrame {
                desc: output_desc.into(),
                payload: output_payload,
            },
        ))
    }

    fn verify_tokens_frame_raw(
        &mut self,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(Vec<i32>, RawActivationDesc, Vec<u8>)> {
        let raw_input = raw_input_frame(input)?;
        let input_desc_ptr = raw_input_desc_ptr(&raw_input);
        let input_payload_ptr = raw_input.1;
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut predicted = vec![0_i32; token_ids.len().saturating_add(3)];
        let mut output_token_count = 0usize;
        let mut error = ptr::null_mut();
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let status = unsafe {
            skippy_ffi::skippy_verify_tokens_frame_sampled(
                self.raw,
                token_ids.as_ptr(),
                token_ids.len(),
                sampling_ptr,
                input_desc_ptr,
                input_payload_ptr,
                &mut output_desc,
                output_payload.as_mut_ptr().cast(),
                output_payload.len(),
                &mut output_bytes,
                predicted.as_mut_ptr(),
                predicted.len(),
                &mut output_token_count,
                &mut error,
            )
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.verify_tokens_frame_raw(token_ids, sampling, input, output_bytes);
        }
        ensure_ok(status, error)?;
        predicted.truncate(output_token_count);
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        Ok((predicted, output_desc, output_payload))
    }

    pub fn copy_output_activation_frame(
        &mut self,
        token_count: usize,
        output_capacity: usize,
    ) -> Result<ActivationFrame> {
        let (output_desc, output_payload) =
            self.copy_output_activation_frame_raw(token_count, output_capacity)?;
        Ok(ActivationFrame {
            desc: output_desc.into(),
            payload: output_payload,
        })
    }

    fn copy_output_activation_frame_raw(
        &mut self,
        token_count: usize,
        output_capacity: usize,
    ) -> Result<(RawActivationDesc, Vec<u8>)> {
        if token_count == 0 {
            return Err(anyhow!(
                "copy_output_activation_frame requires at least one token"
            ));
        }
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_copy_output_activation_frame(
                self.raw,
                token_count,
                &mut output_desc,
                output_payload.as_mut_ptr().cast(),
                output_payload.len(),
                &mut output_bytes,
                &mut error,
            )
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.copy_output_activation_frame_raw(token_count, output_bytes);
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        Ok((output_desc, output_payload))
    }

    pub fn sample_current(&mut self, sampling: Option<&SamplingConfig>) -> Result<i32> {
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let mut predicted = 0_i32;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_sample_current(
                self.raw,
                sampling_ptr,
                &mut predicted,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        Ok(predicted)
    }
}

#[cfg(test)]
mod tests {
    use super::raw_input_frame;
    use crate::{ActivationDesc, ActivationFrame, RuntimeActivationDType, RuntimeActivationLayout};

    fn activation_desc(payload_bytes: u64) -> ActivationDesc {
        ActivationDesc {
            version: 1,
            dtype: RuntimeActivationDType::F32,
            layout: RuntimeActivationLayout::TokenMajor,
            producer_stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            token_count: 1,
            sequence_count: 1,
            payload_bytes,
            flags: 0,
        }
    }

    #[test]
    fn raw_input_frame_rejects_payload_len_mismatch() {
        let frame = ActivationFrame {
            desc: activation_desc(2),
            payload: vec![1],
        };

        let error = raw_input_frame(Some(&frame)).unwrap_err().to_string();

        assert!(
            error.contains("activation payload length 1 does not match descriptor payload_bytes 2"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn raw_input_frame_accepts_matching_payload_len() -> anyhow::Result<()> {
        let frame = ActivationFrame {
            desc: activation_desc(1),
            payload: vec![1],
        };

        let (desc, payload) = raw_input_frame(Some(&frame))?;

        assert_eq!(desc.unwrap().payload_bytes, 1);
        assert_eq!(payload, frame.payload.as_ptr().cast());
        Ok(())
    }
}
