use super::*;

impl IterationScheduler {
    /// Enqueue best-effort runtime work without keeping the request worker
    /// waiting for it. Commands remain FIFO with subsequent iterations, so a
    /// checkpoint queued at an exact session boundary runs before that session
    /// can advance.
    pub(crate) fn execute_runtime_detached(
        &self,
        label: &'static str,
        operation: impl FnOnce(&mut RuntimeState) + Send + 'static,
    ) -> OpenAiResult<()> {
        let (operation, result) = runtime_operation(label, move |runtime| {
            operation(runtime);
            Ok(())
        });
        self.enqueue_command(SchedulerCommand::ExecuteRuntime(operation))?;
        drop(result);
        Ok(())
    }

    /// Deadline-bounded variant of [`Self::execute_runtime_timed`] for
    /// cache-side work (KV record/evict, checkpoint export) that must not
    /// hold a scheduler lane or a caller past the cache operation deadline.
    /// Unlike [`Self::execute_cache_aware_runtime_timed`] the operation does
    /// not join the radix-affinity cache queue; it runs on the plain runtime
    /// command path as soon as the scheduler can take it.
    pub(crate) fn execute_runtime_timed_bounded<T>(
        &self,
        label: &'static str,
        operation_id: String,
        deadline: Instant,
        cancellation: Option<&openai_frontend::CancellationToken>,
        operation: impl FnOnce(&mut RuntimeState) -> OpenAiResult<T> + Send + 'static,
    ) -> OpenAiResult<SchedulerRuntimeOutcome<T>>
    where
        T: Send + 'static,
    {
        let (runtime_operation, result, control) = cache_runtime_operation(
            label,
            operation_id,
            deadline,
            cancellation,
            |runtime, _control| operation(runtime),
        );
        self.enqueue_command(SchedulerCommand::ExecuteRuntime(runtime_operation))?;
        wait_for_cache_runtime(result, &control)
    }
}
