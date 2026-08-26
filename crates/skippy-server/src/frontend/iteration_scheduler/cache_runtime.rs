use super::RuntimeOperation;
use skippy_scheduler::{CacheAffinity, CacheAwareCandidate, order_cache_aware_candidates};
use std::sync::Arc;

pub(super) struct CacheAwareRuntimeOperation {
    pub(super) operation: RuntimeOperation,
    affinity: CacheAffinity,
    prompt_tokens: Arc<[i32]>,
    priority: u64,
    enqueued_turn: u64,
    order: u64,
}

pub(super) struct CacheRuntimeTelemetry {
    pub(super) matched_tokens: usize,
    pub(super) saved_cost: u64,
    pub(super) age_turns: u64,
    pub(super) stage_hits: usize,
    pub(super) cache_epoch: u64,
}

pub(super) struct CacheRuntimeQueue {
    operations: Vec<CacheAwareRuntimeOperation>,
    order_dirty: bool,
    turn: u64,
    next_order: u64,
    aging_cost_per_turn: u64,
    group_waiting_prefixes: bool,
}

impl CacheRuntimeQueue {
    pub(super) fn new(aging_cost_per_turn: u64, group_waiting_prefixes: bool) -> Self {
        Self {
            operations: Vec::new(),
            order_dirty: false,
            turn: 0,
            next_order: 0,
            aging_cost_per_turn,
            group_waiting_prefixes,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    pub(super) fn enqueue(
        &mut self,
        operation: RuntimeOperation,
        affinity: CacheAffinity,
        prompt_tokens: Arc<[i32]>,
        priority: u64,
    ) {
        let order = self.next_order;
        self.next_order = self.next_order.saturating_add(1);
        self.operations.push(CacheAwareRuntimeOperation {
            operation,
            affinity,
            prompt_tokens,
            priority,
            enqueued_turn: self.turn,
            order,
        });
        self.order_dirty = true;
    }

    pub(super) fn advance_turn(&mut self) {
        self.turn = self.turn.saturating_add(1);
    }

    pub(super) fn pop_next(
        &mut self,
    ) -> Option<(CacheAwareRuntimeOperation, CacheRuntimeTelemetry)> {
        self.reorder_if_dirty();
        let queued = self.operations.pop()?;
        let affinity = &queued.affinity;
        let telemetry = CacheRuntimeTelemetry {
            matched_tokens: affinity.matched_tokens(),
            saved_cost: affinity.estimated_saved_cost(),
            age_turns: self.turn.saturating_sub(queued.enqueued_turn),
            stage_hits: affinity
                .stages
                .iter()
                .filter(|stage| stage.matched_tokens > 0)
                .count(),
            cache_epoch: affinity
                .stages
                .iter()
                .map(|stage| stage.cache_epoch)
                .max()
                .unwrap_or(0),
        };
        Some((queued, telemetry))
    }

    fn reorder_if_dirty(&mut self) {
        if !self.order_dirty {
            return;
        }
        let order = order_cache_aware_candidates(
            self.operations
                .iter()
                .enumerate()
                .map(|(index, queued)| CacheAwareCandidate {
                    index,
                    priority: queued.priority,
                    affinity: &queued.affinity,
                    prompt_tokens: queued.prompt_tokens.as_ref(),
                    enqueued_turn: queued.enqueued_turn,
                    order: queued.order,
                }),
            self.turn,
            self.aging_cost_per_turn,
            self.group_waiting_prefixes,
        );
        let mut pending = self.operations.drain(..).map(Some).collect::<Vec<_>>();
        self.operations = order
            .into_iter()
            .rev()
            .map(|index| {
                pending[index]
                    .take()
                    .expect("ordered cache runtime operation must exist")
            })
            .collect();
        self.order_dirty = false;
    }
}

pub(super) fn should_serve_cache_runtime(
    has_cache_runtime: bool,
    has_iteration: bool,
    last_served_cache_runtime: bool,
) -> bool {
    if has_cache_runtime && has_iteration {
        !last_served_cache_runtime
    } else {
        has_cache_runtime
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_scheduler::StageCacheAffinity;
    use std::sync::mpsc;

    fn operation(selected: &mpsc::Sender<&'static str>, label: &'static str) -> RuntimeOperation {
        let selected = selected.clone();
        RuntimeOperation {
            label,
            run: Box::new(move |_| {
                selected.send(label).unwrap();
            }),
        }
    }

    #[test]
    fn cache_runtime_and_decode_work_alternate_without_starvation() {
        assert!(should_serve_cache_runtime(true, true, false));
        assert!(!should_serve_cache_runtime(true, true, true));
        assert!(should_serve_cache_runtime(true, false, true));
        assert!(!should_serve_cache_runtime(false, true, false));
    }

    #[test]
    fn queue_selects_the_longest_prefix_first() {
        let (selected, selected_rx) = mpsc::channel();
        let mut queue = CacheRuntimeQueue::new(4_096, true);
        queue.enqueue(
            operation(&selected, "cold"),
            CacheAffinity::default(),
            Arc::from([0]),
            0,
        );
        queue.enqueue(
            operation(&selected, "hot"),
            CacheAffinity::from_stage(StageCacheAffinity {
                stage_index: 0,
                matched_tokens: 32,
                prefill_cost_per_token: 1,
                restore_cost: 0,
                cache_epoch: 0,
            }),
            Arc::from([1]),
            0,
        );

        (queue.pop_next().unwrap().0.operation.run)(&fake_runtime());
        (queue.pop_next().unwrap().0.operation.run)(&fake_runtime());

        assert_eq!(selected_rx.recv().unwrap(), "hot");
        assert_eq!(selected_rx.recv().unwrap(), "cold");
    }

    #[test]
    fn queue_keeps_shared_waiting_prefixes_adjacent() {
        let (selected, selected_rx) = mpsc::channel();
        let mut queue = CacheRuntimeQueue::new(4_096, true);
        queue.enqueue(
            operation(&selected, "unique"),
            CacheAffinity::default(),
            Arc::from([9, 9, 9]),
            0,
        );
        queue.enqueue(
            operation(&selected, "shared-a"),
            CacheAffinity::default(),
            Arc::from([1, 2, 3]),
            0,
        );
        queue.enqueue(
            operation(&selected, "shared-b"),
            CacheAffinity::default(),
            Arc::from([1, 2, 4]),
            0,
        );

        (queue.pop_next().unwrap().0.operation.run)(&fake_runtime());
        (queue.pop_next().unwrap().0.operation.run)(&fake_runtime());
        (queue.pop_next().unwrap().0.operation.run)(&fake_runtime());

        assert_eq!(selected_rx.recv().unwrap(), "shared-a");
        assert_eq!(selected_rx.recv().unwrap(), "shared-b");
        assert_eq!(selected_rx.recv().unwrap(), "unique");
    }

    fn fake_runtime() -> std::sync::Arc<std::sync::Mutex<crate::runtime_state::RuntimeState>> {
        std::sync::Arc::new(std::sync::Mutex::new(
            crate::runtime_state::RuntimeState::new_modelless_for_test(1),
        ))
    }
}
