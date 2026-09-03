use super::*;

impl StageOpenAiBackend {
    pub(super) fn record_post_decode_exact_state(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &DecodeState,
    ) -> bool {
        let Some(kv) = self.kv.as_ref() else {
            return false;
        };
        if !kv.payload_is_exact_state() {
            return false;
        }
        let Some(checkpoint_tokens) =
            post_decode_checkpoint_tokens(request.prompt_token_ids, &state.generated_token_ids)
        else {
            return false;
        };
        self.enqueue_exact_state_record_at_tokens(
            session_id,
            request.ids,
            checkpoint_tokens,
            "post_decode_checkpoint",
        )
    }

    pub(in crate::frontend) fn enqueue_exact_state_record_at_tokens(
        &self,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        checkpoint_tokens: Vec<i32>,
        decision_prefix: &'static str,
    ) -> bool {
        let scheduler_backend = self.clone();
        let scheduler_session_id = session_id.to_string();
        let scheduler_ids = ids.clone();
        let enqueue = self.iteration_scheduler.execute_runtime_detached(
            "feature-exact-state-checkpoint",
            move |runtime| {
                scheduler_backend.record_exact_state_at_tokens(
                    runtime,
                    &scheduler_session_id,
                    &scheduler_ids,
                    &checkpoint_tokens,
                    decision_prefix,
                );
            },
        );
        if let Err(error) = enqueue {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!(format!("{decision_prefix}_scheduler_error")),
            );
            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
            return false;
        }
        true
    }

    /// Record a recurrent state only when the native session is at the exact
    /// token boundary named by `checkpoint_tokens`.
    ///
    /// The caller must hold the runtime lock. This check is intentionally
    /// canonical-position based: token text or a caller-supplied count cannot
    /// authorize exporting a state at a different native position.
    pub(super) fn record_exact_state_at_tokens(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        checkpoint_tokens: &[i32],
        decision_prefix: &str,
    ) -> bool {
        let Some(kv) = self.kv.as_ref() else {
            return false;
        };
        if !kv.payload_is_exact_state() {
            return false;
        }
        let Ok(checkpoint_token_count) = u64::try_from(checkpoint_tokens.len()) else {
            return false;
        };
        let runtime_token_count = match runtime.canonical_session_position(session_id) {
            Ok(position) => position,
            Err(error) => {
                let mut attrs = self.openai_attrs(ids);
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!(format!("{decision_prefix}_skipped")),
                );
                attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                self.telemetry
                    .emit("stage.openai_kv_record_decision", attrs);
                return false;
            }
        };
        if runtime_token_count != checkpoint_token_count {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!(format!("{decision_prefix}_skipped")),
            );
            attrs.insert(
                "skippy.kv.checkpoint_token_count".to_string(),
                json!(checkpoint_token_count),
            );
            attrs.insert(
                "skippy.kv.runtime_token_count".to_string(),
                json!(runtime_token_count),
            );
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
            return false;
        }

        let base = self.local_kv_message_base(session_id, ids);
        let identity = kv.prefill_identity(&self.config, &base, 0, checkpoint_tokens);
        match kv.record_exact_state(runtime, session_id, &identity) {
            Ok(Some(record)) => {
                let mut attrs = self.openai_attrs(ids);
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!(format!("{decision_prefix}_recorded")),
                );
                attrs.insert(
                    "skippy.exact_cache.recorded_page_id".to_string(),
                    json!(record.page_id),
                );
                attrs.insert(
                    "skippy.exact_cache.payload_kind".to_string(),
                    json!(record.payload_kind.to_string()),
                );
                attrs.insert(
                    "skippy.exact_cache.recorded_tokens".to_string(),
                    json!(record.token_count),
                );
                attrs.insert("skippy.exact_cache.queued".to_string(), json!(true));
                self.telemetry
                    .emit("stage.openai_kv_record_decision", attrs);
                true
            }
            Ok(None) => false,
            Err(error) => {
                let mut attrs = self.openai_attrs(ids);
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!(format!("{decision_prefix}_error")),
                );
                attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                self.telemetry
                    .emit("stage.openai_kv_record_decision", attrs);
                false
            }
        }
    }
}
