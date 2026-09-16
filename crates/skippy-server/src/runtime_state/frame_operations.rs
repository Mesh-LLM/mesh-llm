use super::*;

fn final_sampled_chunk_start(token_count: usize, batch_size: usize) -> usize {
    debug_assert!(token_count > 0);
    (token_count - 1) / batch_size.max(1) * batch_size.max(1)
}

impl RuntimeState {
    pub fn prefill(&mut self, session_id: &str, token_ids: &[i32]) -> Result<()> {
        let session = self.session(session_id)?;
        session.prefill_chunked(token_ids)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(())
    }

    pub fn prefill_chunked_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
    ) -> Result<i32> {
        if token_ids.is_empty() {
            bail!("sampled prefill requires at least one token");
        }
        let session = self.session(session_id)?;
        let batch_size = session.batch_size()?.max(1);
        let final_chunk_start = final_sampled_chunk_start(token_ids.len(), batch_size);
        session.prefill_chunked(&token_ids[..final_chunk_start])?;
        let (predicted, _) = session.prefill_chunk_frame_sampled(
            &token_ids[final_chunk_start..],
            sampling,
            None,
            0,
        )?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(predicted)
    }

    pub fn media_marker(&self) -> String {
        self.model.media_marker()
    }

    pub fn has_media_projector(&self) -> bool {
        self.model.has_media_projector()
    }

    pub fn prefill_media(
        &mut self,
        session_id: &str,
        prompt: &str,
        media: &[MediaInput],
        sampling: Option<&SamplingConfig>,
    ) -> Result<MediaPrefill> {
        let model = &self.model as *const StageModel;
        let session = self.session(session_id)?;
        // `session()` mutably borrows the session map, while the projector lives
        // on the same RuntimeState. RuntimeState serializes access behind one
        // outer mutex, so this split borrow only aliases immutable model state.
        let prefill = unsafe { (&*model).prefill_media(session, prompt, media, sampling) }?;
        self.session_token_counts
            .insert(session_id.to_string(), prefill.position);
        Ok(prefill)
    }

    pub fn prefill_media_frame(
        &mut self,
        session_id: &str,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<MediaPrefillFrame> {
        let model = &self.model as *const StageModel;
        let session = self.session(session_id)?;
        // `session()` mutably borrows the session map, while the projector lives
        // on the same RuntimeState. RuntimeState serializes access behind one
        // outer mutex, so this split borrow only aliases immutable model state.
        let prefill = unsafe { (&*model).prefill_media_frame(session, prompt, media) }?;
        self.session_token_counts
            .insert(session_id.to_string(), prefill.position);
        Ok(prefill)
    }

    pub fn decode(&mut self, session_id: &str, token_id: i32) -> Result<i32> {
        self.decode_sampled(session_id, token_id, None)
    }

    pub fn decode_sampled(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
    ) -> Result<i32> {
        let session = self.session(session_id)?;
        let token = session.decode_step_sampled(token_id, sampling)?;
        self.add_session_tokens(session_id, 1);
        Ok(token)
    }

    pub fn decode_batch_sampled(
        &mut self,
        requests: &[RuntimeDecodeBatchRequest<'_>],
    ) -> Result<Vec<i32>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        Self::ensure_unique_batch_sessions(requests)?;
        for request in requests {
            self.session(request.session_id)?;
        }

        let mut lane_sessions = Vec::with_capacity(requests.len());
        for request in requests {
            let lane_session = self.sessions.remove(request.session_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "session {} was not active after admission",
                    request.session_id
                )
            })?;
            lane_sessions.push((request.session_id.to_string(), lane_session));
        }

        let result = {
            let mut decode_requests = lane_sessions
                .iter_mut()
                .zip(requests.iter())
                .map(|((_, lane_session), request)| DecodeBatchRequest {
                    session: &mut lane_session.session,
                    token_id: request.token_id,
                    sampling: request.sampling,
                })
                .collect::<Vec<_>>();
            StageSession::decode_batch_sampled(&mut decode_requests)
        };

        for (session_id, lane_session) in lane_sessions {
            self.sessions.insert(session_id, lane_session);
        }
        if result.is_ok() {
            for request in requests {
                self.add_session_tokens(request.session_id, 1);
            }
        }
        result
    }

    /// Returns a session's batch size, admitting a new session when necessary.
    pub fn admit_session_batch_size(&mut self, session_id: &str) -> Result<usize> {
        self.session(session_id)?.batch_size()
    }

    /// Returns the batch size of an already admitted session.
    pub fn active_session_batch_size(&mut self, session_id: &str) -> Result<usize> {
        self.active_session(session_id)?.batch_size()
    }

    pub fn ensure_session_active(&mut self, session_id: &str) -> Result<()> {
        self.session(session_id).map(|_| ())
    }

    pub fn configure_chat_sampling(
        &mut self,
        session_id: &str,
        metadata_json: &str,
        prompt_token_count: u64,
        sampling: Option<&SamplingConfig>,
    ) -> Result<()> {
        self.session(session_id)?.configure_chat_sampling(
            metadata_json,
            prompt_token_count,
            sampling,
        )
    }

    pub fn last_token_signal(&mut self, session_id: &str) -> Result<TokenSignal> {
        self.session(session_id)?.last_token_signal()
    }

    pub fn signal_window(
        &mut self,
        session_id: &str,
        window_tokens: u32,
    ) -> Result<GenerationSignalWindow> {
        self.session(session_id)?.signal_window(window_tokens)
    }

    pub fn prefill_frame(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
    ) -> Result<ActivationFrame> {
        self.prefill_frame_with_positions(session_id, token_ids, &[], input)
    }

    pub fn prefill_frame_with_positions(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        positions: &[i32],
        input: Option<&ActivationFrame>,
    ) -> Result<ActivationFrame> {
        let session = self.session(session_id)?;
        let frame = session.prefill_chunk_frame_with_positions(token_ids, positions, input, 0)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(frame)
    }

    pub fn prefill_final_frame_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        let session = self.session(session_id)?;
        let (predicted, frame) = session
            .prefill_chunk_frame_sampled_with_positions(token_ids, positions, sampling, input, 0)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok((predicted, frame))
    }

    pub fn decode_frame(
        &mut self,
        session_id: &str,
        token_id: i32,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        self.decode_frame_sampled(session_id, token_id, None, input, 0)
    }

    pub fn decode_frame_sampled(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output =
            session.decode_step_frame_sampled(token_id, sampling, input, output_capacity)?;
        self.add_session_tokens(session_id, 1);
        Ok(output)
    }

    pub fn decode_frame_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
        max_draft_tokens: usize,
    ) -> Result<(i32, Option<NativeMtpDraft>, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output = session.decode_step_frame_sampled_mtp(
            token_id,
            sampling,
            input,
            output_capacity,
            max_draft_tokens,
        )?;
        self.add_session_tokens(session_id, 1);
        Ok(output)
    }

    pub fn decode_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        max_draft_tokens: usize,
    ) -> Result<(i32, Option<NativeMtpDraft>)> {
        let session = self.session(session_id)?;
        let output = session.decode_step_sampled_mtp(token_id, sampling, max_draft_tokens)?;
        self.add_session_tokens(session_id, 1);
        Ok(output)
    }

    pub fn decode_frame_batch_sampled(
        &mut self,
        requests: &[RuntimeDecodeFrameBatchRequest<'_>],
    ) -> Result<Vec<DecodeFrameBatchOutput>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        Self::ensure_unique_frame_batch_sessions(requests)?;
        for request in requests {
            self.session(request.session_id)?;
        }

        let mut lane_sessions = Vec::with_capacity(requests.len());
        for request in requests {
            let lane_session = self.sessions.remove(request.session_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "session {} was not active after admission",
                    request.session_id
                )
            })?;
            lane_sessions.push((request.session_id.to_string(), lane_session));
        }

        let result = {
            let mut decode_requests = lane_sessions
                .iter_mut()
                .zip(requests.iter())
                .map(|((_, lane_session), request)| DecodeFrameBatchRequest {
                    session: &mut lane_session.session,
                    token_id: request.token_id,
                    sampling: request.sampling,
                    input: request.input,
                })
                .collect::<Vec<_>>();
            StageSession::decode_step_frame_batch_sampled(&mut decode_requests)
        };

        for (session_id, lane_session) in lane_sessions {
            self.sessions.insert(session_id, lane_session);
        }
        if result.is_ok() {
            for request in requests {
                self.add_session_tokens(request.session_id, 1);
            }
        }
        result
    }

    pub fn iteration_batch_sampled(
        &mut self,
        requests: &[RuntimeIterationBatchRequest<'_>],
    ) -> Result<IterationBatchOutput> {
        if requests.is_empty() {
            return Ok(IterationBatchOutput {
                request_outputs: Vec::new(),
                samples: Vec::new(),
            });
        }
        let mut unique = std::collections::BTreeSet::new();
        for request in requests {
            if !unique.insert(request.session_id) {
                bail!(
                    "iteration contains duplicate session {}",
                    request.session_id
                );
            }
        }
        let new_session_count = unique
            .iter()
            .filter(|session_id| !self.sessions.contains_key(**session_id))
            .count();
        let available_lanes = (self.lane_count as usize).saturating_sub(self.sessions.len());
        ensure_iteration_session_capacity(new_session_count, available_lanes)?;
        for request in requests {
            self.session(request.session_id)?;
        }

        let mut lane_sessions = Vec::with_capacity(requests.len());
        for request in requests {
            let lane_session = self.sessions.remove(request.session_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "session {} was not active after admission",
                    request.session_id
                )
            })?;
            lane_sessions.push((request.session_id.to_string(), lane_session));
        }

        let result = {
            let mut iteration_requests = lane_sessions
                .iter_mut()
                .zip(requests.iter())
                .map(|((_, lane_session), request)| IterationBatchRequest {
                    session: &mut lane_session.session,
                    token_ids: request.token_ids,
                    positions: request.positions,
                    sampling: request.sampling,
                    input: request.input,
                    sample_last: request.sample_last,
                    phase: request.phase,
                })
                .collect::<Vec<_>>();
            StageSession::iteration_batch_sampled(&mut iteration_requests)
        };

        for (session_id, lane_session) in lane_sessions {
            self.sessions.insert(session_id, lane_session);
        }
        if result.is_ok() {
            for request in requests {
                self.add_session_tokens(request.session_id, request.token_ids.len() as u64);
            }
        }
        result
    }

    pub fn verify_frame(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(Vec<i32>, Option<NativeMtpDraft>, ActivationFrame)> {
        self.verify_frame_sampled(session_id, token_ids, None, input, output_capacity, 0)
    }

    pub(crate) fn canonical_session_position(&self, session_id: &str) -> Result<u64> {
        let tracked_position = self
            .session_token_counts
            .get(session_id)
            .copied()
            .with_context(|| format!("session {session_id} has no tracked position"))?;
        let session = self
            .sessions
            .get(session_id)
            .with_context(|| format!("session {session_id} is not active"))?;
        let rust_position = session.session.token_count();
        let native_position = session.session.native_position()?;
        if tracked_position != rust_position || tracked_position != native_position {
            bail!(
                "session {session_id} position mismatch: tracked={tracked_position}, rust={rust_position}, native={native_position}"
            );
        }
        Ok(native_position)
    }

    pub(crate) fn verify_tokens_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
    ) -> Result<Vec<i32>> {
        let token_count = u64::try_from(token_ids.len())
            .context("linear verification token count exceeds u64")?;
        let session = self.session(session_id)?;
        let predicted = session.verify_tokens_sampled(token_ids, sampling)?;
        self.add_session_tokens(session_id, token_count);
        Ok(predicted)
    }

    /// Verifies a speculative span in one batched forward, returning the target
    /// predictions plus the MTP draft for the branch that was verified.
    pub(crate) fn verify_tokens_sampled_mtp(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        max_draft_tokens: usize,
    ) -> Result<(Vec<i32>, Option<NativeMtpDraft>)> {
        let token_count = u64::try_from(token_ids.len())
            .context("native MTP verification token count exceeds u64")?;
        let session = self.session(session_id)?;
        let (predicted, draft) =
            session.verify_tokens_sampled_mtp(token_ids, sampling, max_draft_tokens)?;
        self.add_session_tokens(session_id, token_count);
        Ok((predicted, draft))
    }

    pub(crate) fn session_token_count(&self, session_id: &str) -> Option<u64> {
        self.session_token_counts.get(session_id).copied()
    }

    pub fn verify_frame_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
        max_draft_tokens: usize,
    ) -> Result<(Vec<i32>, Option<NativeMtpDraft>, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output = session.verify_tokens_frame_sampled(
            token_ids,
            sampling,
            input,
            output_capacity,
            max_draft_tokens,
        )?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(output)
    }

    pub fn verify_frame_sampled_serial(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(Vec<i32>, Option<NativeMtpDraft>, ActivationFrame)> {
        if token_ids.is_empty() {
            bail!("serial verify_frame requires at least one token");
        }
        let input_frames = split_activation_frame(input, token_ids.len())?;
        let mut predicted_tokens = Vec::with_capacity(token_ids.len());
        let mut output_frames = Vec::with_capacity(token_ids.len());
        let mut last_draft = None;
        for (index, token_id) in token_ids.iter().copied().enumerate() {
            let input_frame = input_frames.as_ref().map(|frames| &frames[index]);
            let (predicted, native_mtp, output) = self.decode_frame_sampled_mtp(
                session_id,
                token_id,
                sampling,
                input_frame,
                output_capacity,
                1,
            )?;
            if predicted >= 0 {
                predicted_tokens.push(predicted);
            }
            last_draft = native_mtp;
            output_frames.push(output);
        }
        Ok((
            predicted_tokens,
            last_draft,
            combine_activation_frames(&output_frames)?,
        ))
    }

    pub fn retire_verify_checkpoint(
        &mut self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<()> {
        self.active_session(session_id)?
            .retire_verify_checkpoint(token_start, token_count)
    }

    pub fn trim_session(&mut self, session_id: &str, token_count: u64) -> Result<()> {
        let session = self.session(session_id)?;
        session.trim_session(token_count)?;
        self.session_token_counts
            .insert(session_id.to_string(), token_count);
        self.notify_session_lifecycle(super::lifecycle::SessionLifecycleEvent::SessionTrimmed {
            token_count,
        });
        Ok(())
    }

    pub fn align_session_to_token_count_if_ahead(
        &mut self,
        session_id: &str,
        token_count: u64,
    ) -> Result<Option<RuntimeSessionAlignStats>> {
        let Some(current) = self.session_token_counts.get(session_id).copied() else {
            return Ok(None);
        };
        if current <= token_count {
            return Ok(None);
        }
        self.trim_session(session_id, token_count)?;
        Ok(Some(RuntimeSessionAlignStats {
            before_token_count: current,
            after_token_count: token_count,
        }))
    }

    pub(super) fn session(&mut self, session_id: &str) -> Result<&mut StageSession> {
        if !self.sessions.contains_key(session_id) {
            let lane_session = self.take_idle_session().map(Ok).unwrap_or_else(|| {
                if self.sessions.len() >= self.lane_count as usize {
                    bail!("all execution lanes are busy");
                }
                self.create_lane_session()
            })?;
            self.sessions.insert(session_id.to_string(), lane_session);
            // Every active session must have a tracked position from the
            // moment it activates (both a fresh and a reused idle lane start
            // at native position 0), so `canonical_session_position` is
            // defined even for a generation that completes before its first
            // prefill/decode call -- see the KV-disabled repro this fixes
            // (`session ... has no tracked position`).
            self.session_token_counts
                .entry(session_id.to_string())
                .or_insert(0);
        }
        Ok(&mut self
            .sessions
            .get_mut(session_id)
            .expect("session inserted above")
            .session)
    }

    fn ensure_unique_batch_sessions(requests: &[RuntimeDecodeBatchRequest<'_>]) -> Result<()> {
        let mut seen = BTreeSet::new();
        for request in requests {
            if !seen.insert(request.session_id) {
                bail!("duplicate session {} in decode batch", request.session_id);
            }
        }
        Ok(())
    }

    fn ensure_unique_frame_batch_sessions(
        requests: &[RuntimeDecodeFrameBatchRequest<'_>],
    ) -> Result<()> {
        let mut seen = BTreeSet::new();
        for request in requests {
            if !seen.insert(request.session_id) {
                bail!(
                    "duplicate session {} in decode frame batch",
                    request.session_id
                );
            }
        }
        Ok(())
    }

    pub(super) fn active_session(&mut self, session_id: &str) -> Result<&mut StageSession> {
        self.sessions
            .get_mut(session_id)
            .map(|lane_session| &mut lane_session.session)
            .ok_or_else(|| anyhow::anyhow!("session {session_id} is not active"))
    }
}

#[cfg(test)]
mod tests {
    use super::final_sampled_chunk_start;

    #[test]
    fn sampled_prefill_keeps_the_final_native_chunk_intact() {
        assert_eq!(final_sampled_chunk_start(1, 512), 0);
        assert_eq!(final_sampled_chunk_start(511, 512), 0);
        assert_eq!(final_sampled_chunk_start(512, 512), 0);
        assert_eq!(final_sampled_chunk_start(513, 512), 512);
        assert_eq!(final_sampled_chunk_start(6603, 2048), 6144);
    }
}

fn ensure_iteration_session_capacity(
    new_session_count: usize,
    available_lanes: usize,
) -> Result<()> {
    if new_session_count > available_lanes {
        bail!(
            "iteration requires {new_session_count} new sessions but only {available_lanes} execution lanes are available"
        );
    }
    Ok(())
}

fn split_activation_frame(
    input: Option<&ActivationFrame>,
    token_count: usize,
) -> Result<Option<Vec<ActivationFrame>>> {
    let Some(input) = input else {
        return Ok(None);
    };
    if token_count == 0 {
        bail!("cannot split activation frame for zero tokens");
    }
    if input.desc.token_count as usize != token_count {
        bail!(
            "activation token count mismatch: frame={} tokens={}",
            input.desc.token_count,
            token_count
        );
    }
    let frames = (0..token_count)
        .map(|token_index| slice_activation_frame(input, token_index, 1))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(frames))
}

fn slice_activation_frame(
    input: &ActivationFrame,
    token_start: usize,
    token_count: usize,
) -> Result<ActivationFrame> {
    let source_tokens =
        usize::try_from(input.desc.token_count).context("activation token count exceeds usize")?;
    if token_count == 0
        || token_start
            .checked_add(token_count)
            .is_none_or(|end| end > source_tokens)
    {
        bail!("activation token slice is outside the frame");
    }
    let mut payload = Vec::new();
    let mut parts =
        [skippy_runtime::ActivationPartDesc::default(); skippy_runtime::ACTIVATION_MAX_PARTS];
    for (part_index, source) in input.desc.parts()?.iter().enumerate() {
        let rank = usize::try_from(source.rank).context("activation part rank exceeds usize")?;
        let token_axis =
            usize::try_from(source.token_axis).context("activation part token axis is negative")?;
        if rank == 0 || rank > source.dimensions.len() || token_axis >= rank {
            bail!("activation part has an invalid token axis");
        }
        let inner_bytes = usize::try_from(source.byte_strides[token_axis])
            .context("activation part token stride exceeds usize")?;
        let plane_bytes = inner_bytes
            .checked_mul(source_tokens)
            .context("activation part plane size overflow")?;
        let source_bytes = usize::try_from(source.payload_bytes)
            .context("activation part payload size exceeds usize")?;
        if plane_bytes == 0 || !source_bytes.is_multiple_of(plane_bytes) {
            bail!("activation part is not aligned to its token axis");
        }
        let source_offset = usize::try_from(source.payload_offset)
            .context("activation part offset exceeds usize")?;
        let slice_offset = token_start
            .checked_mul(inner_bytes)
            .context("activation part slice offset overflow")?;
        let slice_bytes = token_count
            .checked_mul(inner_bytes)
            .context("activation part slice size overflow")?;
        let output_offset = payload.len();
        for outer_index in 0..source_bytes / plane_bytes {
            let start = source_offset
                .checked_add(
                    outer_index
                        .checked_mul(plane_bytes)
                        .context("activation part plane offset overflow")?,
                )
                .and_then(|value| value.checked_add(slice_offset))
                .context("activation part slice offset overflow")?;
            let end = start
                .checked_add(slice_bytes)
                .context("activation part slice range overflow")?;
            payload.extend_from_slice(
                input
                    .payload
                    .get(start..end)
                    .context("activation part slice exceeds payload")?,
            );
        }
        let mut output = *source;
        output.dimensions[token_axis] =
            i64::try_from(token_count).context("activation part token count exceeds i64")?;
        for axis in token_axis + 1..rank {
            output.byte_strides[axis] = output.byte_strides[axis - 1]
                .checked_mul(
                    u64::try_from(output.dimensions[axis - 1])
                        .context("activation part dimension is unresolved")?,
                )
                .context("activation part output stride overflow")?;
        }
        output.payload_offset =
            u64::try_from(output_offset).context("activation part output offset exceeds u64")?;
        output.payload_bytes = u64::try_from(payload.len() - output_offset)
            .context("activation part output size exceeds u64")?;
        parts[part_index] = output;
    }
    let mut desc = input.desc;
    desc.token_count = u32::try_from(token_count).context("activation token count exceeds u32")?;
    desc.sequence_count = 1;
    desc.payload_bytes = u64::try_from(payload.len()).context("activation payload exceeds u64")?;
    desc.parts = parts;
    Ok(ActivationFrame { desc, payload })
}

fn combine_activation_frames(frames: &[ActivationFrame]) -> Result<ActivationFrame> {
    let Some(first) = frames.first() else {
        bail!("cannot combine empty activation frames");
    };
    let mut desc = first.desc;
    let mut payload = Vec::new();
    let mut token_count = 0u32;
    for frame in frames {
        if frame.desc.version != desc.version
            || frame.desc.producer_stage_index != desc.producer_stage_index
            || frame.desc.layer_start != desc.layer_start
            || frame.desc.layer_end != desc.layer_end
            || frame.desc.sequence_count != desc.sequence_count
            || frame.desc.frontier_identity != desc.frontier_identity
            || frame.desc.part_count != desc.part_count
        {
            bail!("cannot combine incompatible activation frames");
        }
        token_count = token_count
            .checked_add(frame.desc.token_count)
            .context("combined activation token count overflow")?;
    }
    let mut parts =
        [skippy_runtime::ActivationPartDesc::default(); skippy_runtime::ACTIVATION_MAX_PARTS];
    for (part_index, first_part) in first.desc.parts()?.iter().enumerate() {
        let rank =
            usize::try_from(first_part.rank).context("activation part rank exceeds usize")?;
        let token_axis = usize::try_from(first_part.token_axis)
            .context("activation part token axis is negative")?;
        if rank == 0 || rank > first_part.dimensions.len() || token_axis >= rank {
            bail!("cannot combine incompatible activation parts");
        }
        let inner_bytes = usize::try_from(first_part.byte_strides[token_axis])
            .context("activation part token stride exceeds usize")?;
        let mut frame_parts = Vec::with_capacity(frames.len());
        let mut slab_bytes = Vec::with_capacity(frames.len());
        let mut outer_count = None;
        for frame in frames {
            let part = *frame
                .desc
                .parts()?
                .get(part_index)
                .context("activation frame is missing a part")?;
            if part.identity != first_part.identity
                || part.ggml_type != first_part.ggml_type
                || part.rank != first_part.rank
                || part.token_axis != first_part.token_axis
                || part.flags != first_part.flags
                || part.dimensions[..rank]
                    .iter()
                    .enumerate()
                    .any(|(axis, value)| {
                        axis != token_axis && *value != first_part.dimensions[axis]
                    })
                || part.byte_strides[..=token_axis] != first_part.byte_strides[..=token_axis]
            {
                bail!("cannot combine incompatible activation parts");
            }
            let slab = inner_bytes
                .checked_mul(frame.desc.token_count as usize)
                .context("activation part slab size overflow")?;
            let part_bytes = usize::try_from(part.payload_bytes)
                .context("activation part payload size exceeds usize")?;
            if slab == 0 || !part_bytes.is_multiple_of(slab) {
                bail!("activation part is not aligned to its token axis");
            }
            let current_outer_count = part_bytes / slab;
            if outer_count
                .replace(current_outer_count)
                .is_some_and(|value| value != current_outer_count)
            {
                bail!("activation parts have incompatible outer dimensions");
            }
            frame_parts.push(part);
            slab_bytes.push(slab);
        }
        let output_offset = payload.len();
        for outer_index in 0..outer_count.unwrap_or(0) {
            for (frame_index, frame) in frames.iter().enumerate() {
                let source_offset = usize::try_from(frame_parts[frame_index].payload_offset)
                    .context("activation part offset exceeds usize")?;
                let start = source_offset
                    .checked_add(
                        outer_index
                            .checked_mul(slab_bytes[frame_index])
                            .context("activation part plane offset overflow")?,
                    )
                    .context("activation part plane offset overflow")?;
                let end = start
                    .checked_add(slab_bytes[frame_index])
                    .context("activation part range overflow")?;
                payload.extend_from_slice(
                    frame
                        .payload
                        .get(start..end)
                        .context("activation part exceeds frame payload")?,
                );
            }
        }
        let mut output = *first_part;
        output.dimensions[token_axis] = i64::from(token_count);
        for axis in token_axis + 1..rank {
            output.byte_strides[axis] = output.byte_strides[axis - 1]
                .checked_mul(
                    u64::try_from(output.dimensions[axis - 1])
                        .context("activation part dimension is unresolved")?,
                )
                .context("activation part output stride overflow")?;
        }
        output.payload_offset =
            u64::try_from(output_offset).context("activation part output offset exceeds u64")?;
        output.payload_bytes = u64::try_from(payload.len() - output_offset)
            .context("activation part output size exceeds u64")?;
        parts[part_index] = output;
    }
    desc.token_count = token_count;
    desc.payload_bytes = payload.len() as u64;
    desc.parts = parts;
    Ok(ActivationFrame { desc, payload })
}

#[cfg(test)]
mod iteration_admission_tests {
    use super::*;

    #[test]
    fn rejects_batch_before_partial_session_admission() {
        assert!(ensure_iteration_session_capacity(3, 2).is_err());
        assert!(ensure_iteration_session_capacity(2, 2).is_ok());
    }

    #[test]
    fn rejects_invalid_activation_axis_before_combining() {
        let mut frame = crate::test_activation::f32_frame(1, &[1.0]);
        frame.desc.parts[0].token_axis = 4;

        let error = combine_activation_frames(&[frame]).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("cannot combine incompatible activation parts")
        );
    }
}
