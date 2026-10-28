//! Exact recurrent prefill boundaries for growing and shortened chat tails.
use super::*;

impl StageOpenAiBackend {
    pub(super) fn deferred_exact_checkpoint_boundaries(
        &self,
        request: &LocalGeneration<'_>,
        prefill_tokens: &[i32],
        restored_prefill_tokens: usize,
    ) -> Vec<usize> {
        let Some(kv) = self.kv.as_ref().filter(|kv| kv.payload_is_exact_state()) else {
            return Vec::new();
        };
        let shared_boundary = kv
            .exact_shared_checkpoint_token_count(prefill_tokens.len() as u64)
            .and_then(|count| usize::try_from(count).ok());
        let chat_boundary = request
            .recurrent_cache_prefix_token_ids
            .filter(|tokens| !tokens.is_empty() && prefill_tokens.starts_with(tokens))
            .map(<[i32]>::len);
        let selected = select_deferred_checkpoint_boundary(
            shared_boundary,
            chat_boundary,
            restored_prefill_tokens,
            request.prompt_token_ids.len(),
        );
        // Only lane-owned recurrent state has bounded, cheap resident captures.
        // Serialized full-state caches retain their existing single boundary.
        // Four slots leave room for both boundaries, prefill and continuation;
        // smaller configured caches keep their previous capture policy.
        let retain_shortened_tail = kv.payload == StagePrefixCachePayload::KvRecurrent
            && self.config.cache_idle_slots != Some(0)
            && kv.resident_recurrent_limits().0 >= 4;
        checkpoint_boundaries(
            selected,
            chat_boundary,
            restored_prefill_tokens,
            self.config
                .n_ubatch
                .unwrap_or(256)
                .min(self.config.n_batch.unwrap_or(2048))
                .clamp(1, 2048) as usize,
            retain_shortened_tail,
        )
        .into_iter()
        .filter(|&boundary| kv.meets_shared_prefix_min_tokens(boundary))
        .collect()
    }
}

fn select_deferred_checkpoint_boundary(
    shared: Option<usize>,
    chat: Option<usize>,
    restored: usize,
    prompt: usize,
) -> Option<usize> {
    let valid = |boundary: &usize| *boundary > restored && *boundary < prompt;
    chat.filter(valid).or_else(|| shared.filter(valid))
}

pub(super) fn checkpoint_boundaries(
    selected: Option<usize>,
    chat: Option<usize>,
    restored: usize,
    microbatch: usize,
    retain_shortened_tail: bool,
) -> Vec<usize> {
    let Some(boundary) = selected else {
        return Vec::new();
    };
    let mut boundaries = Vec::with_capacity(2);
    if retain_shortened_tail && selected == chat {
        // A changed or shortened last message can diverge before the assistant
        // marker. Keep one additional exact state before the last microbatch.
        // Align to the suffix's existing microbatch boundaries: an arbitrary
        // split changes native prefill shapes before the retained state.
        let microbatch = microbatch.max(1);
        let complete_batches =
            boundary.saturating_sub(restored).saturating_sub(microbatch) / microbatch;
        let earlier = restored + complete_batches * microbatch;
        if earlier > restored {
            boundaries.push(earlier);
        }
    }
    boundaries.push(boundary);
    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_boundary_wins_with_valid_grid_fallback() {
        for (shared, chat, restored, expected) in [
            (Some(8), Some(6), 0, Some(6)),
            (Some(8), None, 0, Some(8)),
            (Some(8), Some(4), 4, Some(8)),
            (Some(8), Some(10), 0, Some(8)),
            (Some(8), Some(99), 0, Some(8)),
            (Some(4), Some(6), 4, Some(6)),
            (Some(4), Some(10), 4, None),
            (Some(4), Some(6), 99, None),
            (None, None, 0, None),
        ] {
            assert_eq!(
                select_deferred_checkpoint_boundary(shared, chat, restored, 10),
                expected
            );
        }
    }

    #[test]
    fn shortened_tail_retains_an_earlier_exact_state_without_recapturing_restored_work() {
        assert_eq!(
            checkpoint_boundaries(Some(107_393), Some(107_393), 105_336, 512, true),
            [106_872, 107_393]
        );
        for restored in [106_881, 107_000] {
            assert_eq!(
                checkpoint_boundaries(Some(107_393), Some(107_393), restored, 512, true),
                [107_393]
            );
        }
        assert_eq!(checkpoint_boundaries(Some(7), Some(7), 0, 512, true), [7]);
        assert_eq!(checkpoint_boundaries(Some(7), Some(7), 0, 0, true), [6, 7]);
        assert_eq!(
            checkpoint_boundaries(Some(188), Some(188), 0, 32, true),
            [128, 188]
        );
        assert_eq!(
            checkpoint_boundaries(Some(188), Some(188), 7, 32, true),
            [135, 188]
        );
    }

    #[test]
    fn serialized_or_small_caches_and_grid_fallback_do_not_add_captures() {
        assert_eq!(
            checkpoint_boundaries(Some(188), Some(188), 0, 32, false),
            [188]
        );
        assert_eq!(checkpoint_boundaries(Some(188), None, 0, 32, true), [188]);
        assert!(checkpoint_boundaries(None, Some(188), 188, 32, true).is_empty());
    }
}
