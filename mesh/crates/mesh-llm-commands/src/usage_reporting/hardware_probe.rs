//! Own the startup survey until it completes or the serving session ends.

use skippy_hardware_profile::probe::ProbeScope;
use std::sync::Arc;

pub struct HardwareReport {
    probes: Arc<ProbeScope>,
    task: tokio::task::JoinHandle<()>,
}

impl HardwareReport {
    pub(super) fn start(collect: impl FnOnce() + Send + 'static) -> Self {
        let probes = Arc::new(ProbeScope::default());
        let worker = Arc::clone(&probes);
        let task = tokio::task::spawn_blocking(move || worker.run(collect));
        Self { probes, task }
    }

    /// Allow a failed startup a bounded chance to report its hardware.
    pub async fn finish_within(&mut self, budget: std::time::Duration) -> anyhow::Result<()> {
        match tokio::time::timeout(budget, &mut self.task).await {
            Ok(result) => result.map_err(Into::into),
            Err(_) => self.shutdown().await,
        }
    }

    /// Cancel child tools and join the blocking collector before dropping Tokio.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        let probes = Arc::clone(&self.probes);
        tokio::task::spawn_blocking(move || probes.cancel()).await??;
        (&mut self.task).await?;
        Ok(())
    }
}

impl Drop for HardwareReport {
    fn drop(&mut self) {
        if let Err(error) = self.probes.cancel() {
            use std::io::Write;
            let _ = writeln!(
                mesh_llm_events::console_err(),
                "failed to stop hardware probe children: {error}"
            );
        }
    }
}

#[cfg(test)]
mod tests;
