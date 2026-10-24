//! Feeds [`super::energy`] with GPU board power read from `nvidia-smi` (#2320).
//!
//! One `nvidia-smi` process streams a reading per GPU every second, and the
//! sampler sums each round into one reading for the node. A GPU that does not
//! report power is left out of the sum. A node without `nvidia-smi` reports no
//! energy at all.

use std::sync::Once;

/// One round of per-GPU readings, summed once every GPU has reported.
#[derive(Debug)]
struct PowerRound {
    gpus: usize,
    seen: usize,
    watts: f64,
    reported: bool,
}

impl PowerRound {
    fn new(gpus: usize) -> Self {
        Self {
            gpus,
            seen: 0,
            watts: 0.0,
            reported: false,
        }
    }

    /// Adds one GPU's reading; returns the node total when the round is
    /// complete and at least one GPU in it reported power.
    fn push(&mut self, watts: Option<f64>) -> Option<f64> {
        self.seen += 1;
        if let Some(watts) = watts {
            self.watts += watts;
            self.reported = true;
        }
        if self.seen < self.gpus {
            return None;
        }
        let total = self.reported.then_some(self.watts);
        *self = Self::new(self.gpus);
        total
    }
}

/// Starts the sampler the first time this node serves a local model, so the
/// idle draw it measures is the node's with a model loaded. Later calls do
/// nothing. The sampler ends quietly when `nvidia-smi` is missing or stops.
pub(crate) fn start_energy_sampler() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        runtime.spawn(async {
            if let Err(error) = nvidia::sample().await {
                tracing::debug!("GPU energy sampling unavailable: {error:#}");
            }
        });
    });
}

mod nvidia {
    use super::PowerRound;
    use anyhow::{Context, Result};
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    const SOURCE: &str = "nvidia-smi";

    pub(super) async fn sample() -> Result<()> {
        let listing = Command::new("nvidia-smi")
            .args(["--query-gpu=index", "--format=csv,noheader"])
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .await
            .context("run nvidia-smi")?;
        let gpus = String::from_utf8_lossy(&listing.stdout)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count();
        if !listing.status.success() || gpus == 0 {
            return Ok(());
        }
        let mut child = Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,power.draw",
                "--format=csv,noheader,nounits",
                "--loop-ms=1000",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("start nvidia-smi power sampling")?;
        // Windows doesn't end a child with its parent, and nvidia-smi keeps
        // looping once its stdout is broken: a killed node would leave it
        // polling the GPU for good.
        #[cfg(windows)]
        if let Some(handle) = child.raw_handle()
            && let Err(error) = mesh_llm_system::process_lifetime::end_with_current_process(handle)
        {
            tracing::debug!("nvidia-smi is not tied to the node's lifetime: {error}");
        }
        let stdout = child
            .stdout
            .take()
            .context("nvidia-smi power sampling has no stdout")?;
        let mut lines = BufReader::new(stdout).lines();
        let mut round = PowerRound::new(gpus);
        while let Some(line) = lines.next_line().await? {
            let Some((_, watts)) = crate::system::hardware::parse_nvidia_power_draw_line(&line)
            else {
                continue;
            };
            if let Some(total) = round.push(watts) {
                super::super::energy::record_power(SOURCE, total);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::PowerRound;

    #[test]
    fn a_round_sums_every_gpu_that_reports_power() {
        let mut round = PowerRound::new(2);
        assert_eq!(round.push(Some(100.0)), None);
        assert_eq!(round.push(Some(50.0)), Some(150.0));
        assert_eq!(round.push(None), None);
        assert_eq!(round.push(Some(40.0)), Some(40.0));
    }

    #[test]
    fn a_round_where_no_gpu_reports_power_has_no_reading() {
        let mut round = PowerRound::new(1);
        assert_eq!(round.push(None), None);
        assert_eq!(round.push(Some(30.0)), Some(30.0));
    }
}
