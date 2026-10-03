use super::*;

impl StageOpenAiBackend {
    pub(in crate::frontend) fn try_restore_embedded_split_prefill(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        downstream: &mut TcpStream,
        prefill_tokens: &[i32],
    ) -> OpenAiResult<Option<ChainPrefixRestore>> {
        let Some(kv) = self.kv.as_ref() else {
            return Ok(None);
        };
        if prefill_tokens.is_empty() || !kv.should_lookup() {
            return Ok(None);
        }
        let base = self.local_kv_message_base(session_key, request.ids);
        let identities = kv.lookup_identities(request.config, &base, 0, prefill_tokens);
        let mut restore_stats = StageReplyStats::default();
        let scheduler_kv = Arc::clone(kv);
        let scheduler_session_key = session_key.to_string();
        let scheduler_prefill_tokens = prefill_tokens.to_vec();
        let local_restore = self.iteration_scheduler.execute_runtime(
            "embedded-split-prefix-restore",
            move |runtime| match scheduler_kv
                .restore_exact_state_with_shared_durable_fallback(
                    runtime,
                    &scheduler_session_key,
                    &identities,
                )
                .map_err(openai_backend_error)?
            {
                Some(restored) => Ok(Some(restored.token_count)),
                None => scheduler_kv
                    .restore_resident_prefix(
                        runtime,
                        &scheduler_session_key,
                        &identities,
                        &scheduler_prefill_tokens,
                    )
                    .map_err(openai_backend_error)
                    .map(|restored| restored.map(|restored| restored.token_count)),
            },
        )?;
        let Some(local_restore) = local_restore else {
            return Ok(None);
        };
        if local_restore == 0 {
            return Ok(None);
        }
        let restored_tokens = local_restore.min(prefill_tokens.len());
        restore_stats.kv_lookup_hits += 1;
        restore_stats.kv_imported_pages += 1;
        restore_stats.kv_imported_tokens += restored_tokens as i64;
        restore_stats.kv_hit_stage_mask |= openai_stage_mask(request.config.stage_index);
        let restore = embedded_prefix_cache_message(
            WireMessageKind::TryRestorePrefill,
            &prefill_tokens[..restored_tokens],
            request.ids.request_id,
            request.ids.session_id,
        )?;
        write_stage_message_conditioned(
            &mut *downstream,
            &restore,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let downstream_restore = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        if downstream_restore.kind != WireReplyKind::Ack {
            return Err(OpenAiError::backend(format!(
                "expected prefix try-restore ACK from downstream, got {:?}",
                downstream_restore.kind
            )));
        }
        restore_stats.merge(downstream_restore.stats);
        if restore_stats.kv_lookup_errors > 0
            || restore_stats.kv_lookup_misses > 0
            || downstream_restore.stats.kv_lookup_hits == 0
        {
            self.drop_embedded_split_restore(request, session_key, downstream);
            return Ok(None);
        }
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert("skippy.kv.decision".to_string(), json!("chain_restore_hit"));
        attrs.insert(
            "skippy.kv.restored_tokens".to_string(),
            json!(restored_tokens),
        );
        attrs.insert(
            "skippy.kv.suffix_prefill_tokens".to_string(),
            json!(prefill_tokens.len().saturating_sub(restored_tokens)),
        );
        attrs.insert(
            "skippy.kv.lookup_hits".to_string(),
            json!(restore_stats.kv_lookup_hits),
        );
        attrs.insert(
            "skippy.kv.hit_stage_mask".to_string(),
            json!(restore_stats.kv_hit_stage_mask),
        );
        insert_chain_prefix_cache_savings_attrs(
            &mut attrs,
            chain_prefix_cache_savings(&restore_stats, restored_tokens, request.activation_width),
        );
        self.telemetry
            .emit("stage.openai_kv_lookup_decision", attrs);
        Ok(Some(ChainPrefixRestore {
            restored_tokens,
            stats: restore_stats,
        }))
    }
}
