//! Serialized verifier calls that outlive the refresh caller's deadline.
use super::*;

impl PluginManager {
    #[cfg(test)]
    pub(crate) async fn verify_provider_evidence(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
    ) -> bool {
        self.verify_provider_evidence_until(
            mesh,
            node,
            evidence,
            tokio::time::Instant::now() + Duration::from_secs(12),
        )
        .await
    }

    pub(crate) async fn verify_provider_evidence_until(
        &self,
        mesh: &str,
        node: &str,
        evidence: &BTreeMap<String, Vec<u8>>,
        deadline: tokio::time::Instant,
    ) -> bool {
        for (plugin, verifier) in &self.inner.peer_evidence.verifiers {
            let bytes = evidence
                .get(&verifier.declaration.namespace)
                .map(Vec::as_slice);
            if !self
                .verify_evidence(plugin, mesh, node, bytes, deadline)
                .await
                .is_ok_and(|entry| entry.accepted)
            {
                return false;
            }
        }
        true
    }

    pub(super) async fn verify_evidence(
        &self,
        plugin: &str,
        mesh: &str,
        node: &str,
        bytes: Option<&[u8]>,
        deadline: tokio::time::Instant,
    ) -> Result<Eligibility> {
        ensure!(
            bytes.is_none_or(|bytes| bytes.len() <= MAX_EVIDENCE_BYTES),
            "oversized peer evidence"
        );
        let registry = self.peer_evidence();
        let key = EvidenceRegistry::lease_key(plugin, mesh, node, bytes);
        if let Some(entry) = registry.cached(&key) {
            return Ok(entry);
        }
        let verifier = registry.verifier(plugin)?;
        let generation = verifier.generation.load(Ordering::Acquire);
        let active = verifier
            .active_generation
            .as_ref()
            .map(|active| active.load(Ordering::Acquire));
        ensure!(active != Some(0), "peer evidence verifier is not active");
        let evidence = match bytes.map(serde_json::from_slice).transpose() {
            Ok(evidence) => evidence,
            Err(error) => {
                return self.cache_verification_result(
                    key,
                    generation,
                    active,
                    Err(anyhow::Error::new(error).context("invalid peer evidence JSON")),
                    false,
                );
            }
        };
        // The actual RPC owns the permit, even after its refresh caller times out.
        let permit =
            tokio::time::timeout_at(deadline, registry.slots.clone().acquire_owned()).await;
        let Ok(permit) = permit else {
            return self.cache_verification_result(
                key,
                generation,
                active,
                Err(anyhow::anyhow!("peer evidence verifier busy")),
                false,
            );
        };
        let permit = permit?;
        if let Some(entry) = registry.cached(&key) {
            return Ok(entry);
        }
        let verifier = self.current_evidence_verifier(plugin).await?;
        let generation = verifier.generation.load(Ordering::Acquire);
        let active = verifier
            .active_generation
            .as_ref()
            .map(|active| active.load(Ordering::Acquire));
        ensure!(active != Some(0), "peer evidence verifier is not active");
        let handler = verifier.declaration.handler.clone();
        let params = serde_json::to_value(PeerEvidenceVerificationRequest {
            mesh_id: mesh.into(),
            node_id: node.into(),
            evidence,
        })?;
        let manager = self.clone();
        let task_key = key.clone();
        let mut invocation = tokio::spawn(async move {
            let _permit = permit;
            let result = manager
                .invoke_service_json(
                    &task_key.plugin,
                    proto::ServiceKind::Operation,
                    &handler,
                    &params,
                )
                .await;
            manager.cache_verification_result(task_key, generation, active, result, true)
        });
        match tokio::time::timeout_at(deadline, &mut invocation).await {
            Ok(result) => result.context("peer evidence verifier task failed")?,
            Err(_) => self.cache_verification_result(
                key,
                generation,
                active,
                Err(anyhow::anyhow!("peer evidence verifier timed out")),
                false,
            ),
        }
    }

    fn cache_verification_result(
        &self,
        key: EvidenceKey,
        generation: u64,
        active: Option<u64>,
        result: Result<PeerEvidenceVerificationResult>,
        replace_current: bool,
    ) -> Result<Eligibility> {
        let registry = self.peer_evidence();
        let verifier = registry.verifier(&key.plugin)?;
        let mut leases = registry.leases.lock().unwrap_or_else(|e| e.into_inner());
        ensure!(
            generation == verifier.generation.load(Ordering::Acquire)
                && verifier
                    .active_generation
                    .as_ref()
                    .is_none_or(|current| { active == Some(current.load(Ordering::Acquire)) }),
            "peer evidence verifier stopped during verification"
        );
        // A caller's timeout must not overwrite a result that won the race.
        if !replace_current
            && let Some(current) = leases
                .get(&key)
                .filter(|entry| entry.current() && entry.runtime_generation == active.unwrap_or(0))
        {
            return Ok(*current);
        }
        let wall_now = SystemTime::now();
        let now = wall_now
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let monotonic_now = Instant::now();
        let (accepted, valid_until, deadline, cache_until_restart) = result
            .and_then(|result| {
                let deadline = if result.cache_until_restart {
                    monotonic_now
                } else {
                    ensure!(result.valid_until > now, "evidence lease expired");
                    let remaining = UNIX_EPOCH
                        .checked_add(Duration::from_secs(result.valid_until))
                        .context("evidence validity exceeds wall clock range")?
                        .duration_since(wall_now)
                        .context("evidence lease expired")?;
                    monotonic_now
                        .checked_add(remaining)
                        .context("evidence validity exceeds monotonic clock range")?
                };
                Ok((
                    result.accepted,
                    result.valid_until,
                    deadline,
                    result.cache_until_restart,
                ))
            })
            .unwrap_or_else(|error| {
                tracing::debug!(plugin = %key.plugin, %error, "peer evidence verifier failed");
                let retry = Duration::from_secs(5);
                (
                    false,
                    now.saturating_add(retry.as_secs()),
                    monotonic_now + retry,
                    false,
                )
            });
        let entry = Eligibility {
            accepted,
            valid_until,
            deadline,
            cache_until_restart,
            last_used: monotonic_now,
            runtime_generation: active.unwrap_or(0),
        };
        EvidenceRegistry::insert(&mut leases, key, entry);
        registry.cache_changed.notify_one();
        Ok(entry)
    }
}
