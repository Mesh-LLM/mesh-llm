//! Feeds [`super::energy`] with GPU board power read from `nvidia-smi` (#2320).
//!
//! One `nvidia-smi` process streams a reading per GPU every second, and the
//! sampler sums each round into one reading for the node. A round only counts
//! when every GPU in it reports power: a partial sum would understate the
//! node, so a round where a GPU misses its reading is a gap. A node without
//! `nvidia-smi` reports no energy at all, and neither does one whose first
//! round misses a reading (Jetson's integrated GPU answers `[N/A]`): the
//! sampler stops there.

use anyhow::Result;
use std::sync::Once;
use tokio::io::{AsyncBufRead, Lines};

/// One round of per-GPU readings, summed once every GPU has reported.
#[derive(Debug)]
struct PowerRound {
    gpus: usize,
    seen: usize,
    watts: f64,
    missing: bool,
}

/// What one GPU's reading did to its round.
#[derive(Debug, PartialEq)]
enum Pushed {
    /// Other GPUs of the round have not reported yet.
    Pending,
    /// The round is complete: the node total, or `None` when a GPU in it
    /// did not report power.
    Complete(Option<f64>),
}

impl PowerRound {
    fn new(gpus: usize) -> Self {
        Self {
            gpus,
            seen: 0,
            watts: 0.0,
            missing: false,
        }
    }

    fn push(&mut self, watts: Option<f64>) -> Pushed {
        self.seen += 1;
        match watts {
            Some(watts) => self.watts += watts,
            None => self.missing = true,
        }
        if self.seen < self.gpus {
            return Pushed::Pending;
        }
        let total = (!self.missing).then_some(self.watts);
        *self = Self::new(self.gpus);
        Pushed::Complete(total)
    }
}

/// Records each round's node total from `nvidia-smi` lines until the stream
/// ends, or right after the first round when it misses a reading: a board
/// that never reports power would otherwise be polled every second for
/// nothing. A later round that misses one is only a gap, which the meter
/// already leaves out.
async fn read_rounds<R: AsyncBufRead + Unpin>(
    lines: &mut Lines<R>,
    gpus: usize,
    mut record: impl FnMut(f64),
) -> Result<()> {
    let mut round = PowerRound::new(gpus);
    let mut measured = false;
    while let Some(line) = lines.next_line().await? {
        let Some((_, watts)) = crate::system::hardware::parse_nvidia_power_draw_line(&line) else {
            continue;
        };
        match round.push(watts) {
            Pushed::Pending => {}
            Pushed::Complete(Some(total)) => {
                measured = true;
                record(total);
            }
            Pushed::Complete(None) if !measured => {
                tracing::info!(
                    "GPU energy sampling stopped: nvidia-smi does not report power draw for every GPU, so /api/status has no energy"
                );
                return Ok(());
            }
            Pushed::Complete(None) => {}
        }
    }
    Ok(())
}

/// Starts the sampler the first time this node serves a local model, so the
/// idle draw it measures is the node's with a model loaded. Later calls do
/// nothing. The sampler ends quietly when `nvidia-smi` is missing or stops,
/// and with one log line when a GPU reports no power.
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
        // Returning drops `child`, and `kill_on_drop` ends nvidia-smi with it.
        super::read_rounds(&mut lines, gpus, |total| {
            super::super::energy::record_power(SOURCE, total);
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::{PowerRound, Pushed, read_rounds};
    use tokio::io::AsyncBufReadExt;

    #[test]
    fn a_round_sums_its_gpus_once_every_one_reports_power() {
        let mut round = PowerRound::new(2);
        assert_eq!(round.push(Some(100.0)), Pushed::Pending);
        assert_eq!(round.push(Some(50.0)), Pushed::Complete(Some(150.0)));
        assert_eq!(round.push(Some(90.0)), Pushed::Pending);
        assert_eq!(round.push(Some(40.0)), Pushed::Complete(Some(130.0)));
    }

    #[test]
    fn a_round_where_a_gpu_misses_its_reading_has_no_reading() {
        // A partial sum would understate the node, so it is not a reading.
        let mut round = PowerRound::new(2);
        assert_eq!(round.push(None), Pushed::Pending);
        assert_eq!(round.push(Some(40.0)), Pushed::Complete(None));
        assert_eq!(round.push(Some(30.0)), Pushed::Pending);
        assert_eq!(round.push(Some(20.0)), Pushed::Complete(Some(50.0)));
    }

    async fn recorded(output: &str, gpus: usize) -> (Vec<f64>, Option<String>) {
        let mut lines = output.as_bytes().lines();
        let mut totals = Vec::new();
        read_rounds(&mut lines, gpus, |total| totals.push(total))
            .await
            .unwrap();
        (totals, lines.next_line().await.unwrap())
    }

    #[tokio::test]
    async fn a_board_without_power_readings_stops_the_sampler_after_one_round() {
        // What a Jetson Orin's nvidia-smi streams: `power.draw` is `[N/A]`.
        let (totals, unread) = recorded("0, [N/A]\n0, [N/A]\n0, 30.0\n", 1).await;
        assert!(totals.is_empty());
        assert_eq!(unread.as_deref(), Some("0, [N/A]"));
    }

    #[tokio::test]
    async fn a_gpu_that_never_reports_power_stops_the_sampler_too() {
        let (totals, unread) = recorded("0, 120.0\n1, [N/A]\n0, 110.0\n1, [N/A]\n", 2).await;
        assert!(totals.is_empty());
        assert_eq!(unread.as_deref(), Some("0, 110.0"));
    }

    #[tokio::test]
    async fn a_gpu_that_misses_a_later_reading_leaves_a_gap_and_sampling_goes_on() {
        let output = "0, 120.0\n1, 5.0\n0, 115.0\n1, [N/A]\n0, 110.0\n1, 20.0\n";
        let (totals, unread) = recorded(output, 2).await;
        assert_eq!(totals, [125.0, 130.0]);
        assert_eq!(unread, None);
    }
}
