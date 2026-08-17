use super::*;

impl LoggingRuntimeState {
    /// Start the durable persistence worker after the runtime startup boundary
    /// has finished opening the confined store and artifact capture facade.
    ///
    /// `FailOpenArtifactCapture::open` runs its idempotent startup recovery
    /// before this state is constructed, so no producer can hand work to this
    /// service before recovery has completed. The underlying service start is
    /// idempotent, which keeps repeated embedded/runtime entrypoints from
    /// creating duplicate workers for one installed state.
    pub(crate) async fn start_persistence_worker(&self) -> Option<Arc<LoggingService>> {
        // Keep the synchronous activation boundary short. SQLite cleanup runs
        // on the blocking pool and startup must await its result before this
        // service is published as ready, but neither operation may hold this
        // mutex across an await.
        let service = {
            let _activation = self
                .activation_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.retired.load(Ordering::Acquire) {
                return None;
            }
            let service = Arc::clone(self.service.as_ref()?);
            let _ = service.spawn();
            if !service.is_spawned() {
                return None;
            }
            service
        };

        self.start_cleanup_worker(&service).await;
        self.start_webhook_delivery_worker();

        // Retirement can win while the bounded startup cleanup awaits.  Do
        // not hand a caller a service from a displaced runtime state.
        if self.retired.load(Ordering::Acquire) || !service.is_startable() {
            return None;
        }
        Some(service)
    }

    /// Start the opt-in durable-delivery scheduler only after the logging
    /// service has crossed its persistence startup boundary. Configuration or
    /// client construction failures remain local and fail open for serving.
    fn start_webhook_delivery_worker(&self) {
        let Some(config) = self.webhook_config.clone() else {
            return;
        };
        let Some(store) = self.store() else {
            return;
        };
        let Some(metrics) = self.service.as_ref().map(|service| service.metrics()) else {
            return;
        };
        let _activation = self
            .activation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.retired.load(Ordering::Acquire) {
            return;
        }
        let mut installed = self
            .webhook_delivery_worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if installed.is_some() {
            return;
        }
        let transport = match ReqwestWebhookTransport::new() {
            Ok(transport) => Arc::new(transport),
            Err(_) => {
                tracing::warn!(
                    "webhook delivery transport unavailable; continuing without dispatch"
                );
                return;
            }
        };
        let worker = match WebhookDeliveryWorker::from_config(
            store,
            &config,
            transport,
            Arc::new(SystemWebhookWorkerClock),
            Arc::new(RandomWebhookJitter),
        ) {
            Ok(worker) => worker.with_metrics(metrics),
            Err(_) => {
                tracing::warn!(
                    "webhook delivery configuration unavailable; continuing without dispatch"
                );
                return;
            }
        };
        *installed = Some(WebhookDeliveryScheduler::start(worker));
    }

    async fn start_cleanup_worker(&self, service: &Arc<LoggingService>) {
        let Some(store) = self.store() else {
            return;
        };

        let startup_waiter = self.create_and_publish_cleanup_worker(store, service);

        let Some(startup_waiter) = startup_waiter else {
            return;
        };
        // The test hook deliberately pauses only after the scheduler has been
        // published while holding the activation gate. This preserves an
        // observable retirement/cancellation boundary without reopening the
        // concurrent-start candidate race.
        self.pause_cleanup_installation_for_test().await;
        // Cleanup failure is deliberately fail-open: status/audit records the
        // result while the request-serving runtime still comes up.
        let _ = CleanupWorker::wait_for_startup_with(startup_waiter).await;
    }

    /// Return the canonical scheduler's readiness watch. Candidate creation
    /// and handle publication share the activation gate with retirement, so a
    /// concurrent caller can only observe the published worker; it can never
    /// spawn a losing task that races cleanup or overwrites shared status.
    fn create_and_publish_cleanup_worker(
        &self,
        store: Arc<LogStore>,
        service: &Arc<LoggingService>,
    ) -> Option<tokio::sync::watch::Receiver<Option<CleanupOutcome>>> {
        // This is the same gate retirement takes before releasing the state
        // for asynchronous shutdown. There is no await in this critical
        // section: task construction, the ownership check, and publication
        // form one atomic transition.
        let _activation = self
            .activation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.retired.load(Ordering::Acquire) || !service.is_startable() {
            return None;
        }

        let mut installed = self
            .cleanup_worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(worker) = installed.as_ref() {
            return Some(worker.startup_waiter());
        }

        #[cfg(test)]
        self.cleanup_candidate_count.fetch_add(1, Ordering::Relaxed);
        let candidate = CleanupWorker::start(
            store,
            self.artifact_capture.clone(),
            Arc::clone(service),
            self.retention_max_rows,
            self.webhook_dead_letter_retention_secs,
            self.cleanup_cadence,
            Arc::clone(&self.cleanup_status),
        );
        let startup_waiter = candidate.startup_waiter();
        *installed = Some(candidate);
        Some(startup_waiter)
    }

    /// Stop the scheduler before the service persistence worker. This ordering
    /// prevents a late cleanup audit from being offered after service shutdown.
    pub(crate) async fn shutdown_cleanup_worker(&self) {
        let worker = self
            .cleanup_worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(worker) = worker
            && !worker.shutdown().await
        {
            // A timed-out cleanup task retains its exclusive connection
            // and may still be unwinding after interruption. Keep the
            // owner installed so no replacement claims it stopped or
            // starts a concurrent scheduler on this runtime state.
            *self
                .cleanup_worker
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(worker);
        }
    }

    /// Stop the terminal-delivery scheduler only after persistence has drained
    /// its final terminal records. The scheduler's own fixed join bound keeps
    /// shutdown finite; unfinished leased rows remain durable for restart
    /// recovery.
    pub(crate) async fn shutdown_webhook_delivery_worker(&self) {
        let worker = self
            .webhook_delivery_worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(worker) = worker {
            worker.shutdown().await;
        }
    }

    /// Make this installed state permanently non-startable, then stop all of
    /// its background work in the required order. This is used only by the
    /// process-global replacement boundary; ordinary runtime shutdown may use
    /// the individual worker methods while leaving its state inspectable.
    /// Returns false if cleanup did not retire within its fixed bound. In that
    /// case callers must preserve this retired state rather than installing a
    /// replacement scheduler over a still-owned cleanup connection.
    pub(crate) async fn retire_and_shutdown(&self) -> bool {
        {
            let _activation = self
                .activation_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.retired.store(true, Ordering::Release);
            if let Some(service) = self.service.as_ref() {
                service.retire();
            }
        }

        // Cleanup may emit an audit entry, so it must be joined before the
        // persistence drain closes its delivery boundary.
        self.shutdown_cleanup_worker().await;
        if self.status().cleanup_worker_state == "timed_out" {
            self.shutdown_webhook_delivery_worker().await;
            return false;
        }
        if let Some(service) = self.service.as_ref() {
            let _ = service.shutdown().await;
        }
        self.shutdown_webhook_delivery_worker().await;
        true
    }

    #[cfg(test)]
    pub(crate) fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(super) fn install_cleanup_publish_hook_for_test(&self) -> Arc<CleanupInstallHook> {
        let hook = Arc::new(CleanupInstallHook::new());
        *self
            .cleanup_install_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&hook));
        hook
    }

    #[cfg(test)]
    pub(super) fn has_cleanup_worker_for_test(&self) -> bool {
        self.cleanup_worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    }

    #[cfg(test)]
    pub(crate) fn has_webhook_delivery_worker_for_test(&self) -> bool {
        self.webhook_delivery_worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    }

    #[cfg(test)]
    pub(super) fn cleanup_candidate_count_for_test(&self) -> usize {
        self.cleanup_candidate_count.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) async fn pause_cleanup_installation_for_test(&self) {
        let hook = self
            .cleanup_install_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(hook) = hook {
            hook.candidate_created.wait().await;
            hook.resume_install.wait().await;
        }
    }

    #[cfg(not(test))]
    pub(super) async fn pause_cleanup_installation_for_test(&self) {}
}
