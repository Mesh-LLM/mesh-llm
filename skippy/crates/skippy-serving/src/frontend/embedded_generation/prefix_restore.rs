use super::*;
use crate::frontend::generation::EmbeddedFusedFirstDecode;
use std::net::TcpStream;
use std::time::Instant;

pub(super) struct EmbeddedPrefixRestore {
    pub(super) allowed: bool,
    pub(super) chain_cache_restored: bool,
    pub(super) chain_restored_tokens: usize,
    pub(super) chain_cache_stats: StageReplyStats,
    pub(super) fused_first_decode: Option<EmbeddedFusedFirstDecode>,
}

impl StageOpenAiBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_embedded_prefix(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        downstream: &mut TcpStream,
        prefill_tokens: &[i32],
        wire_sampling: Option<WireSamplingConfig>,
        cache_stats: &mut GenerationCacheStats,
    ) -> InferenceResult<EmbeddedPrefixRestore> {
        // A native MTP head keeps sidecar state that a restore cannot rebuild,
        // so native MTP prefills in full. A split DFlash draft also rides the
        // native MTP flag, but its final stage drafts again from the rows
        // decoded after the restore point, so it can take a restored prefix.
        let allowed = prefix_restore_allowed(request);
        let started = Instant::now();
        let mut chain_cache_restored = false;
        let mut chain_restored_tokens = 0usize;
        let mut chain_cache_stats = StageReplyStats::default();
        let mut fused_first_decode = None;

        if !allowed && self.kv.is_some() {
            let mut attrs = self.openai_attrs(request.ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("bypass_native_mtp_sidecar"),
            );
            attrs.insert(
                "skippy.kv.prompt_token_count".to_string(),
                json!(prefill_tokens.len()),
            );
            self.telemetry
                .emit("stage.openai_kv_lookup_decision", attrs);
        }

        // The replay, first-token and fused restore paths decode without the
        // split DFlash flag, so a split DFlash request takes the chain prefix
        // restore below and decodes as usual.
        if allowed && request.max_tokens > 0 && request.draft.is_none() && !request.split_dflash() {
            let current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            if let Some(cached) =
                self.try_restore_embedded_split_exact_replay(request, session_key, downstream)?
            {
                chain_cache_restored = true;
                chain_restored_tokens = request
                    .prompt_token_ids
                    .len()
                    .saturating_add(cached.predicted_tokens.len().saturating_sub(1));
                chain_cache_stats = cached.reply_stats;
                cache_stats.cached_prompt_tokens = saturating_u32(request.prompt_token_ids.len());
                cache_stats.matched_prefix_tokens = saturating_u32(request.prompt_token_ids.len());
                cache_stats.suffix_prefill_tokens = 0;
                cache_stats.status = "hit";
                cache_stats.hit_kind = Some("chain_exact_replay");
                fused_first_decode = Some(cached);
            } else if let Some(cached) = self.try_restore_embedded_split_full_prompt_first_token(
                request,
                session_key,
                downstream,
            )? {
                chain_cache_restored = true;
                chain_restored_tokens = request.prompt_token_ids.len();
                chain_cache_stats = cached.reply_stats;
                cache_stats.cached_prompt_tokens = saturating_u32(request.prompt_token_ids.len());
                cache_stats.matched_prefix_tokens = saturating_u32(request.prompt_token_ids.len());
                cache_stats.suffix_prefill_tokens = 0;
                cache_stats.status = "hit";
                cache_stats.hit_kind = Some("chain_full_prompt_first_token");
                fused_first_decode = Some(cached);
            } else if let Some(fused) = self.try_restore_embedded_split_prefill_and_decode(
                request,
                session_key,
                downstream,
                prefill_tokens,
                current,
                wire_sampling,
            )? {
                chain_cache_restored = true;
                chain_restored_tokens = prefill_tokens.len();
                chain_cache_stats = fused.reply_stats;
                cache_stats.cached_prompt_tokens = saturating_u32(prefill_tokens.len());
                cache_stats.matched_prefix_tokens = saturating_u32(prefill_tokens.len());
                cache_stats.suffix_prefill_tokens = 0;
                cache_stats.status = "hit";
                cache_stats.hit_kind = Some("chain_fused_exact_prefix");
                fused_first_decode = Some(fused);
            }
        }

        if !chain_cache_restored
            && allowed
            && let Some(restore) = self.try_restore_embedded_split_prefill(
                request,
                session_key,
                downstream,
                prefill_tokens,
            )?
        {
            chain_restored_tokens = restore.restored_tokens;
            chain_cache_restored = chain_restored_tokens >= prefill_tokens.len();
            chain_cache_stats = restore.stats;
            cache_stats.cached_prompt_tokens = saturating_u32(chain_restored_tokens);
            cache_stats.matched_prefix_tokens = saturating_u32(chain_restored_tokens);
            cache_stats.suffix_prefill_tokens =
                saturating_u32(prefill_tokens.len().saturating_sub(chain_restored_tokens));
            cache_stats.status = "hit";
            cache_stats.hit_kind = Some("chain_prefix");
        }

        if cache_stats.cached_prompt_tokens > 0 {
            cache_stats.restore_ms = started.elapsed().as_secs_f64() * 1_000.0;
        }

        Ok(EmbeddedPrefixRestore {
            allowed,
            chain_cache_restored,
            chain_restored_tokens,
            chain_cache_stats,
            fused_first_decode,
        })
    }
}

/// Whether a split generation may restore a cached prompt prefix.
pub(super) fn prefix_restore_allowed(request: &EmbeddedStageZeroGeneration<'_>) -> bool {
    restore_allowed(request.native_mtp_enabled, request.split_dflash())
}

fn restore_allowed(native_mtp_enabled: bool, split_dflash: bool) -> bool {
    !native_mtp_enabled || split_dflash
}

#[cfg(test)]
mod tests {
    use super::restore_allowed;

    #[test]
    fn split_dflash_restores_a_prefix_and_native_mtp_does_not() {
        assert!(restore_allowed(false, false));
        assert!(
            restore_allowed(true, true),
            "split DFlash rides the MTP flag"
        );
        assert!(!restore_allowed(true, false), "native MTP sidecar state");
    }
}
