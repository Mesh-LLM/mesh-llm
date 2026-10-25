//! Feeds [`super::energy`] with GPU power (#2320): from `nvidia-smi` on any
//! host, and on Apple Silicon from IOReport through macmon, which reads the
//! whole GPU once a second.
//!
//! One `nvidia-smi` process streams a reading per GPU every second, and the
//! sampler sums each round into one reading for the node. A round only counts
//! when every GPU in it reports power: a partial sum would understate the
//! node, so a round where a GPU misses its reading is skipped. A node without
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
/// nothing. A later round that misses one is skipped: the meter integrates
/// the longer interval from the full readings on each side, as it does any
/// interval up to `MAX_SAMPLE_GAP_SECONDS`, and leaves the requests running
/// through it out of the rates only past that.
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
/// nothing. `nvidia-smi` is tried on every host and ends quietly when it is
/// missing or stops, and with one log line when a GPU reports no power. A Mac
/// also starts the Apple Silicon sampler.
pub(crate) fn start_energy_sampler() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        #[cfg(target_os = "macos")]
        apple::start();
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

/// Apple Silicon's GPU power from IOReport, read through macmon without sudo.
#[cfg(target_os = "macos")]
mod apple {
    const SOURCE: &str = "macmon";

    /// macmon blocks for each one-second reading, so it gets its own thread.
    /// It ends quietly where macmon can't read the SoC, as on an Intel Mac.
    pub(super) fn start() {
        let spawned = std::thread::Builder::new()
            .name("gpu-energy-sampler".into())
            .spawn(|| {
                let mut sampler = match macmon::Sampler::new() {
                    Ok(sampler) => sampler,
                    Err(error) => {
                        tracing::debug!("Apple GPU energy sampling unavailable: {error}");
                        return;
                    }
                };
                loop {
                    match sampler.get_metrics(1000) {
                        Ok(metrics) => {
                            super::super::energy::record_power(SOURCE, f64::from(metrics.gpu_power))
                        }
                        Err(error) => {
                            tracing::debug!("Apple GPU energy sampling stopped: {error}");
                            return;
                        }
                    }
                }
            });
        if let Err(error) = spawned {
            tracing::debug!("Apple GPU energy sampler not started: {error}");
        }
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
    async fn a_round_where_a_gpu_misses_a_later_reading_is_skipped_and_sampling_goes_on() {
        let output = "0, 120.0\n1, 5.0\n0, 115.0\n1, [N/A]\n0, 110.0\n1, 20.0\n";
        let (totals, unread) = recorded(output, 2).await;
        assert_eq!(totals, [125.0, 130.0]);
        assert_eq!(unread, None);
    }
}
