//! Scheduler batch execution and bounded selected-token readouts.

use super::*;
use crate::dflash::DecodedRows;

/// Collect valid sampled tokens while retaining their original batch request indexes.
pub(super) fn collect_iteration_samples(
    request_count: usize,
    sampled_output_count: usize,
    sampled_request_indexes: &[usize],
    predicted_tokens: &[i32],
) -> Result<Vec<IterationSample>> {
    if sampled_output_count > request_count
        || sampled_output_count > sampled_request_indexes.len()
        || sampled_output_count > predicted_tokens.len()
    {
        return Err(anyhow!(
            "native iteration returned {sampled_output_count} samples for {request_count} requests"
        ));
    }
    let mut seen_samples = vec![false; request_count];
    sampled_request_indexes
        .iter()
        .copied()
        .zip(predicted_tokens.iter().copied())
        .take(sampled_output_count)
        .map(|(request_index, predicted_token)| {
            let sampled = seen_samples.get_mut(request_index).ok_or_else(|| {
                anyhow!(
                    "native iteration sample references request {request_index}, but only {request_count} requests were submitted"
                )
            })?;
            if *sampled {
                return Err(anyhow!(
                    "native iteration returned duplicate sample for request {request_index}"
                ));
            }
            *sampled = true;
            Ok(IterationSample {
                request_index,
                predicted_token,
            })
        })
        .collect()
}

/// Appends each split-DFlash stage's captured target-layer rows to the frame
/// its request produced, so a decode run through the scheduler carries them
/// to the final stage like one run as a stage message. Without them the
/// final stage's draft drops its whole context.
fn forward_dflash_features(
    requests: &mut [IterationBatchRequest<'_>],
    outputs: &mut [ActivationFrame],
) {
    for (request, frame) in requests.iter_mut().zip(outputs) {
        if let Err(error) = request.session.dflash_forward_captured(frame) {
            crate::write_native_log_note(format!("dflash feature forwarding failed: {error:#}"));
        }
    }
}

impl StageSession {
    /// Execute one scheduler iteration for every request as a single native
    /// batch. Frames that carry optional parts, and exporting stages whose
    /// memory splits an all-output batch by sequence, are executed one at a
    /// time here, by decision, so the native layer never has to signal a
    /// fallback.
    pub fn iteration_batch_sampled(
        requests: &mut [IterationBatchRequest<'_>],
    ) -> Result<IterationBatchOutput> {
        let mut outputs = if requests
            .iter()
            .any(iteration_request_requires_one_at_a_time)
        {
            Self::iteration_batch_sampled_one_at_a_time(requests)?
        } else {
            Self::iteration_batch_sampled_raw(requests, &vec![0; requests.len()], &[])?.0
        };
        forward_dflash_features(requests, &mut outputs.request_outputs);
        Ok(outputs)
    }

    /// Evaluate independent sequences in one batch and read only the selected
    /// next-token log probabilities. The vocabulary never crosses the native
    /// boundary. A non-terminal stage returns empty score vectors.
    pub fn iteration_batch_scored(
        requests: &mut [IterationBatchRequest<'_>],
        selections: &[&[i32]],
    ) -> Result<(IterationBatchOutput, Vec<Vec<f32>>)> {
        anyhow::ensure!(
            requests.len() == selections.len(),
            "readout request count mismatch"
        );
        if selections.iter().all(|selected| selected.is_empty()) {
            return Self::iteration_batch_sampled(requests)
                .map(|output| (output, vec![Vec::new(); requests.len()]));
        }
        for (request, selected) in requests.iter().zip(selections) {
            anyhow::ensure!(selected.len() <= 256, "readout exceeds 256 selected tokens");
            anyhow::ensure!(
                selected.is_empty() || !request.sample_last,
                "a readout cannot sample"
            );
            anyhow::ensure!(
                selected.is_empty() || !frame_has_optional_parts(request.input),
                "causal readouts require a text-only activation frontier"
            );
        }
        if requests
            .iter()
            .any(iteration_request_requires_one_at_a_time)
        {
            let mut outputs = IterationBatchOutput {
                request_outputs: Vec::new(),
                samples: Vec::new(),
            };
            let mut scores = Vec::new();
            for (index, (request, selected)) in requests.iter_mut().zip(selections).enumerate() {
                if selected.is_empty() {
                    let (predicted_token, output) = Self::execute_one_iteration_request(request)
                        .map_err(|error| {
                            anyhow::Error::new(PartialBatchExecution::new(index, error))
                        })?;
                    if iteration_request_should_emit_sample(request) {
                        outputs.samples.push(IterationSample {
                            request_index: index,
                            predicted_token,
                        });
                    }
                    outputs.request_outputs.push(output);
                    scores.push(Vec::new());
                    continue;
                }
                let (mut result, mut readouts) = Self::iteration_batch_sampled_raw(
                    std::slice::from_mut(request),
                    &[0],
                    &[*selected],
                )
                .map_err(|error| anyhow::Error::new(PartialBatchExecution::new(index, error)))?;
                outputs.request_outputs.append(&mut result.request_outputs);
                outputs
                    .samples
                    .extend(result.samples.into_iter().map(|sample| IterationSample {
                        request_index: index,
                        predicted_token: sample.predicted_token,
                    }));
                scores.append(&mut readouts);
            }
            forward_dflash_features(requests, &mut outputs.request_outputs);
            return Ok((outputs, scores));
        }
        let (mut outputs, scores) =
            Self::iteration_batch_sampled_raw(requests, &vec![0; requests.len()], selections)?;
        forward_dflash_features(requests, &mut outputs.request_outputs);
        Ok((outputs, scores))
    }

    /// Execute a native batch and validate its sampled or selected-token outputs.
    fn iteration_batch_sampled_raw(
        requests: &mut [IterationBatchRequest<'_>],
        output_capacities: &[usize],
        selections: &[&[i32]],
    ) -> Result<(IterationBatchOutput, Vec<Vec<f32>>)> {
        if requests.is_empty() {
            return Ok((
                IterationBatchOutput {
                    request_outputs: Vec::new(),
                    samples: Vec::new(),
                },
                Vec::new(),
            ));
        }
        let raw_sampling = requests
            .iter()
            .map(|request| request.sampling.map(SamplingConfig::as_raw).transpose())
            .collect::<Result<Vec<_>>>()?;
        let input_frames = requests
            .iter()
            .map(|request| raw_input_frame(request.input))
            .collect::<Result<Vec<_>>>()?;
        let mut score_outputs = selections
            .iter()
            .map(|tokens| vec![f32::NAN; tokens.len()])
            .collect::<Vec<_>>();
        let readouts = selections
            .iter()
            .zip(&mut score_outputs)
            .map(|(tokens, output)| skippy_ffi::TokenScoreRequest {
                token_ids: tokens.as_ptr(),
                token_count: tokens.len(),
                out_log_probabilities: output.as_mut_ptr(),
            })
            .collect::<Vec<_>>();
        let raw_requests =
            raw_iteration_requests(requests, &raw_sampling, &input_frames, &readouts);
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
        let mut sampled_request_indexes = vec![0_usize; requests.len()];
        let mut predicted_tokens = vec![0_i32; requests.len()];
        let mut sampled_output_count = 0_usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_iteration_batch_sampled(
                raw_requests.as_ptr(),
                raw_requests.len(),
                output_descs.as_mut_ptr(),
                output_payload_ptrs.as_ptr(),
                output_capacities.as_ptr(),
                output_bytes.as_mut_ptr(),
                sampled_request_indexes.as_mut_ptr(),
                predicted_tokens.as_mut_ptr(),
                predicted_tokens.len(),
                &mut sampled_output_count,
                &mut error,
            )
        };
        if status == Status::BufferTooSmall
            && output_bytes
                .iter()
                .zip(output_capacities.iter())
                .any(|(required, capacity)| required > capacity)
        {
            free_error(error);
            return Self::iteration_batch_sampled_raw(requests, &output_bytes, selections);
        }
        ensure_ok(status, error)?;
        let samples = collect_iteration_samples(
            requests.len(),
            sampled_output_count,
            &sampled_request_indexes,
            &predicted_tokens,
        )?;
        advance_iteration_counts(requests)?;
        validate_score_outputs(requests, &mut score_outputs)?;
        let request_outputs = iteration_frames(output_payloads, output_descs, output_bytes)?;
        Ok((
            IterationBatchOutput {
                request_outputs,
                samples,
            },
            score_outputs,
        ))
    }

    /// Execute requests serially when the native stage cannot export a shared batch.
    fn iteration_batch_sampled_one_at_a_time(
        requests: &mut [IterationBatchRequest<'_>],
    ) -> Result<IterationBatchOutput> {
        let mut request_outputs = Vec::with_capacity(requests.len());
        let mut samples = Vec::new();
        for (request_index, request) in requests.iter_mut().enumerate() {
            let (predicted_token, output) =
                Self::execute_one_iteration_request(request).map_err(|error| {
                    anyhow::Error::new(PartialBatchExecution::new(request_index, error))
                })?;
            if iteration_request_should_emit_sample(request) {
                samples.push(IterationSample {
                    request_index,
                    predicted_token,
                });
            }
            request_outputs.push(output);
        }
        Ok(IterationBatchOutput {
            request_outputs,
            samples,
        })
    }

    /// Execute one scheduler iteration request on its own. The caller owns the
    /// partial-failure contract, so this returns plainly and every failure is
    /// reported where the batch knows how many requests already ran.
    fn execute_one_iteration_request(
        request: &mut IterationBatchRequest<'_>,
    ) -> Result<(i32, ActivationFrame)> {
        if request.phase == IterationBatchPhase::Decode {
            validate_serial_decode_request(request)?;
            return request.session.decode_step_frame_sampled(
                request.token_ids[0],
                request.sampling,
                request.input,
                0,
            );
        }
        if request.sample_last {
            return if request.positions.is_empty() {
                request.session.prefill_chunk_frame_sampled(
                    request.token_ids,
                    request.sampling,
                    request.input,
                    0,
                )
            } else {
                request.session.prefill_chunk_frame_sampled_with_positions(
                    request.token_ids,
                    request.positions,
                    request.sampling,
                    request.input,
                    0,
                )
            };
        }
        let output = if request.positions.is_empty() {
            request
                .session
                .prefill_chunk_frame(request.token_ids, request.input, 0)?
        } else {
            request.session.prefill_chunk_frame_with_positions(
                request.token_ids,
                request.positions,
                request.input,
                0,
            )?
        };
        Ok((-1, output))
    }
}

// Every pointer borrows storage owned by the caller for the entire native call.
/// Borrow request buffers into the C ABI layout for one native iteration.
fn raw_iteration_requests(
    requests: &[IterationBatchRequest<'_>],
    sampling: &[Option<RawSamplingConfig>],
    inputs: &[RawInputFrame],
    readouts: &[skippy_ffi::TokenScoreRequest],
) -> Vec<RawIterationRequest> {
    requests
        .iter()
        .zip(sampling)
        .zip(inputs)
        .enumerate()
        .map(
            |(index, ((request, sampling), input))| RawIterationRequest {
                session: request.session.raw,
                token_ids: request.token_ids.as_ptr(),
                token_count: request.token_ids.len(),
                positions: if request.positions.is_empty() {
                    ptr::null()
                } else {
                    request.positions.as_ptr()
                },
                position_count: request.positions.len(),
                sampling: sampling
                    .as_ref()
                    .map_or(ptr::null(), |config| config as *const _),
                input_desc: raw_input_desc_ptr(input),
                input_payload: input.1,
                sample_last: request.sample_last,
                readout: readouts
                    .get(index)
                    .filter(|readout| readout.token_count > 0)
                    .map_or(ptr::null(), |readout| readout as *const _),
            },
        )
        .collect()
}

/// Advance session token counts only for requests that actually completed.
fn advance_iteration_counts(requests: &mut [IterationBatchRequest<'_>]) -> Result<()> {
    // Native already advanced all sessions. Do not update only a subset of the
    // Rust counts if overflow is detected; the caller invalidates the batch.
    let counts = requests
        .iter()
        .map(|request| {
            request
                .session
                .token_count
                .checked_add(
                    u64::try_from(request.token_ids.len()).context("token count exceeds u64")?,
                )
                .context("session token count overflow")
        })
        .collect::<Result<Vec<_>>>()?;
    // Native batches concatenate each request's rows in request order.
    let mut first_row = 0usize;
    for (request, count) in requests.iter_mut().zip(counts) {
        request.session.token_count = count;
        let rows = request.token_ids.len();
        let decoded = DecodedRows::for_prefill(rows, request.positions).offset(first_row);
        request.session.dflash_after_decode(decoded);
        first_row += rows;
    }
    Ok(())
}

/// Reject score outputs with incorrect cardinality or invalid log probabilities.
fn validate_score_outputs(
    requests: &[IterationBatchRequest<'_>],
    outputs: &mut [Vec<f32>],
) -> Result<()> {
    for (request, scores) in requests.iter().zip(outputs) {
        if request.session.terminal_stage {
            anyhow::ensure!(
                scores
                    .iter()
                    .all(|score| score.is_finite() && *score <= 0.0),
                "native readout returned invalid log probabilities"
            );
        } else {
            scores.clear();
        }
    }
    Ok(())
}

/// Read each completed request's activation frame from the native iteration result.
fn iteration_frames(
    payloads: Vec<Vec<u8>>,
    descriptors: Vec<RawActivationDesc>,
    sizes: Vec<usize>,
) -> Result<Vec<ActivationFrame>> {
    payloads
        .into_iter()
        .zip(descriptors)
        .zip(sizes)
        .map(|((mut payload, desc), size)| {
            anyhow::ensure!(
                size <= payload.len(),
                "native iteration output exceeds its buffer"
            );
            payload.truncate(size);
            Ok(ActivationFrame {
                desc: desc.into(),
                payload,
            })
        })
        .collect()
}

/// Decide whether an output-bearing request should emit a sampled token.
pub(super) fn iteration_request_should_emit_sample(request: &IterationBatchRequest<'_>) -> bool {
    request.sample_last && request.session.terminal_stage
}

/// Reject unsupported serial-decode shapes before invoking native execution.
pub(super) fn validate_serial_decode_request(request: &IterationBatchRequest<'_>) -> Result<()> {
    let framed_origin = request.session.token_count() == 0
        && request.input.is_some()
        && (request.positions.is_empty() || request.positions == [0]);
    anyhow::ensure!(
        request.session.token_count() > 0 || framed_origin,
        "serial decode fallback requires an established session or a framed origin decode"
    );
    anyhow::ensure!(
        request.token_ids.len() == 1,
        "serial decode fallback requires exactly one token"
    );
    anyhow::ensure!(
        request.sample_last,
        "serial decode fallback requires sampling"
    );
    if let Some(position) = request.positions.first() {
        anyhow::ensure!(
            request.positions.len() == 1,
            "serial decode fallback accepts at most one explicit position"
        );
        let position = u64::try_from(*position).context("decode position is negative")?;
        anyhow::ensure!(
            position == request.session.token_count(),
            "serial decode position {position} does not match session position {}",
            request.session.token_count()
        );
    }
    Ok(())
}

/// Whether a frame carries parts beyond the planned frontier (optional ports),
/// which the native batched path does not concatenate.
pub(super) fn frame_has_optional_parts(frame: Option<&ActivationFrame>) -> bool {
    frame.is_some_and(|frame| {
        frame
            .desc
            .parts()
            .map(|parts| parts.iter().any(|part| part.is_optional()))
            .unwrap_or(true)
    })
}

/// Whether one iteration request has to run on its own, outside the native
/// batch.
///
/// The native batched path reads activation exports out of the graph result of
/// the *last* native microbatch and slices them by request offset. That is
/// sound only while the whole iteration is a single microbatch, so a frame
/// carrying optional ports (which the batched frontier does not concatenate)
/// or an exporting stage whose memory splits an all-output batch by sequence
/// cannot batch.
pub(super) fn iteration_request_requires_one_at_a_time(
    request: &IterationBatchRequest<'_>,
) -> bool {
    frame_has_optional_parts(request.input) || exporting_stage_requires_one_request(request.session)
}

/// Identify exporting stages whose capabilities require serial request execution.
pub(super) fn exporting_stage_requires_one_request(session: &StageSession) -> bool {
    !session.terminal_stage && !session.batched_activation_exports
}

#[cfg(test)]
mod tests;
