//! Phase orchestration. The driver must own a retained server session and expose
//! management only while that owned server is admitted. No callback performs I/O.
use super::l3_contract::{Config, Phase, Run, Status};
use crate::command::DynResult;
use std::path::{Path, PathBuf};

pub(super) struct DiskRoots {
    owner: crate::automation::private_state::PrivateState,
    cache: PathBuf,
    #[cfg(test)]
    next_server: usize,
}
impl DiskRoots {
    pub fn create(parent: &Path) -> DynResult<Self> {
        let owner =
            crate::automation::private_state::PrivateState::create(parent, "replay-disk-l3")?;
        owner.prepare()?;
        let cache = owner.root().join("cache/disk-l3");
        std::fs::create_dir(&cache)?;
        Ok(Self {
            owner,
            cache,
            #[cfg(test)]
            next_server: 0,
        })
    }
    pub fn cache(&self) -> &Path {
        &self.cache
    }
    #[cfg(test)]
    pub fn server_state(&mut self) -> DynResult<PathBuf> {
        self.next_server = self
            .next_server
            .checked_add(1)
            .ok_or("server index overflow")?;
        let path = self
            .owner
            .root()
            .join(format!("state-{}", self.next_server));
        std::fs::create_dir(&path)?;
        Ok(path)
    }
    /// Call only after retained supervisor finalization has joined/reaped all children.
    pub fn finish(self) -> DynResult<()> {
        self.owner
            .finish(Ok::<_, String>(()))
            .map_err(|error| format!("disk-L3 private-state finalization: {error:?}").into())
    }
}
#[derive(Clone, Copy)]
pub(super) enum Mode {
    Final,
    Additional,
    Growth,
    Identical(usize),
    HighLoad(usize),
}
#[derive(Clone, Copy)]
pub(super) enum Disk {
    Off,
    On,
    LowSpace,
}
pub(super) trait Driver {
    /// Starts a fresh server, preserving roots.cache() for On and LowSpace.
    /// Must check endpoint collisions and readiness under retained supervision.
    async fn start(&mut self, disk: Disk, roots: &mut DiskRoots) -> DynResult<()>;
    /// Reap the owned process group and retain native logs before returning.
    async fn stop(&mut self) -> DynResult<()>;
    async fn status(&mut self) -> DynResult<Status>;
    async fn committed(&mut self, minimum_writes: u64) -> DynResult<Status>;
    async fn empty(&mut self) -> DynResult<super::l3_contract::Operation>;
    /// Retain raw JSONL; Final visits every selected source, Identical uses only
    /// the primary final checkpoint. Growth replays every recorded assistant turn.
    async fn measure(&mut self, name: &str, mode: Mode) -> DynResult<Phase>;
    /// Run request and POST prune/DELETE clear concurrently, retain both receipts,
    /// then perform four stable empty polls after the request completes.
    async fn traffic(&mut self) -> DynResult<Phase>;
    fn checkpoint(&mut self, run: &Run) -> DynResult<()>;
}
async fn measured<D: Driver>(
    driver: &mut D,
    name: &str,
    mode: Mode,
    commit: bool,
) -> DynResult<Phase> {
    let before = driver.status().await?;
    let mut phase = driver.measure(name, mode).await?;
    let after = if commit {
        driver
            .committed(
                before
                    .activity
                    .as_ref()
                    .ok_or("missing activity")?
                    .writes
                    .checked_add(1)
                    .ok_or("write counter overflow")?,
            )
            .await?
    } else {
        driver.status().await?
    };
    phase.activity_delta = Some(
        after
            .activity
            .as_ref()
            .ok_or("missing after activity")?
            .delta(before.activity.as_ref().ok_or("missing before activity")?)?,
    );
    phase.status_before = Some(before);
    phase.status_after = Some(after);
    Ok(phase)
}
async fn cold<D: Driver>(driver: &mut D, roots: &mut DiskRoots, run: &mut Run) -> DynResult<()> {
    let mut phase = Phase::default();
    for sample in 0..run.config.cold_samples {
        driver.start(Disk::Off, roots).await?;
        let mut measured = driver
            .measure(&format!("disk_off_cold_{}", sample + 1), Mode::Final)
            .await?;
        for request in &mut measured.requests {
            request.request_id = format!("{}:disk-off:{}", request.session_id, sample + 1);
        }
        phase.requests.extend(measured.requests);
        driver.stop().await?;
    }
    run.phases.insert("disk_off_cold".into(), phase);
    driver.checkpoint(run)
}
async fn populate<D: Driver>(
    driver: &mut D,
    roots: &mut DiskRoots,
    run: &mut Run,
) -> DynResult<()> {
    driver.start(Disk::On, roots).await?;
    run.phases.insert(
        "multi_turn_growth".into(),
        measured(driver, "multi_turn_growth", Mode::Growth, true).await?,
    );
    driver.empty().await?;
    driver.stop().await?;
    driver.start(Disk::On, roots).await?;
    let status = driver.status().await?;
    if status
        .usage
        .as_ref()
        .is_none_or(|u| u.manifests != 0 || u.reserved_inflight_bytes != 0)
    {
        return Err("empty-root phase started nonempty".into());
    }
    // Initial amplification reference is primary only; additional sources are
    // committed separately by the driver, keeping the baseline payload comparable.
    run.phases.insert(
        "disk_on_empty".into(),
        measured(driver, "disk_on_empty", Mode::Identical(1), true).await?,
    );
    run.phases.insert(
        "disk_on_additional_sources".into(),
        measured(driver, "disk_on_additional_sources", Mode::Additional, true).await?,
    );
    run.phases.insert(
        "same_process_l1".into(),
        measured(driver, "same_process_l1", Mode::Final, false).await?,
    );
    driver.stop().await?;
    driver.checkpoint(run)
}
async fn restarts<D: Driver>(
    driver: &mut D,
    roots: &mut DiskRoots,
    run: &mut Run,
) -> DynResult<()> {
    let mut phase = Phase::default();
    for sample in 0..run.config.restart_samples {
        driver.start(Disk::On, roots).await?;
        let mut sample_phase = measured(
            driver,
            &format!("restart_l3_{}", sample + 1),
            Mode::Final,
            false,
        )
        .await?;
        for request in &mut sample_phase.requests {
            request.request_id = format!("{}:restart:{}", request.session_id, sample + 1);
        }
        phase.requests.extend(sample_phase.requests);
        phase
            .activity_deltas
            .push(sample_phase.activity_delta.ok_or("missing restart delta")?);
        driver.stop().await?;
    }
    run.phases.insert("restart_l3".into(), phase);
    driver.start(Disk::On, roots).await?;
    run.phases.insert(
        "concurrent_fill".into(),
        measured(
            driver,
            "concurrent_fill",
            Mode::Identical(run.config.identical_repeats),
            false,
        )
        .await?,
    );
    driver.stop().await?;
    driver.start(Disk::On, roots).await?;
    driver.empty().await?;
    driver.stop().await?;
    driver.start(Disk::On, roots).await?;
    run.phases.insert(
        "concurrent_record".into(),
        measured(
            driver,
            "concurrent_record",
            Mode::Identical(run.config.identical_repeats),
            true,
        )
        .await?,
    );
    driver.stop().await?;
    driver.checkpoint(run)
}
async fn remainder<D: Driver>(
    driver: &mut D,
    roots: &mut DiskRoots,
    run: &mut Run,
) -> DynResult<()> {
    driver.start(Disk::On, roots).await?;
    run.phases
        .insert("lifecycle_under_traffic".into(), driver.traffic().await?);
    driver.stop().await?;
    driver.start(Disk::LowSpace, roots).await?;
    run.phases.insert(
        "low_space".into(),
        measured(driver, "low_space", Mode::Identical(1), false).await?,
    );
    driver.stop().await?;
    for (disk, label) in [(Disk::Off, "off"), (Disk::On, "on")] {
        driver.start(disk, roots).await?;
        for concurrency in &run.config.concurrency {
            let name = format!("high_load_{label}_c{concurrency}");
            let phase = driver.measure(&name, Mode::HighLoad(*concurrency)).await?;
            run.phases.insert(name, phase);
            driver.checkpoint(run)?;
        }
        driver.stop().await?;
    }
    Ok(())
}
pub(super) async fn execute<D: Driver>(
    driver: &mut D,
    roots: &mut DiskRoots,
    run: &mut Run,
) -> DynResult<()> {
    run.config.validate(true)?;
    let result = async {
        cold(driver, roots, run).await?;
        populate(driver, roots, run).await?;
        restarts(driver, roots, run).await?;
        remainder(driver, roots, run).await
    }
    .await;
    // stop is idempotent and must finalize even on request, status or checkpoint errors.
    let stopped = driver.stop().await;
    result?;
    stopped?;
    run.completed_at = Some(crate::ci_operations::ci_metrics_time::isoformat(
        crate::ci_operations::ci_metrics_time::Instant::now(),
    ));
    run.gates = Some(super::l3_gates::evaluate(run)?);
    driver.checkpoint(run)?;
    if run.gates.as_ref().is_some_and(|g| g.passed) {
        Ok(())
    } else {
        Err("disk-L3 lifecycle gates failed; evidence retained".into())
    }
}
pub(super) fn plan(config: &Config) -> DynResult<serde_json::Value> {
    config.validate(false)?;
    Ok(
        serde_json::json!({"schema_version":1,"kind":"disk-l3-lifecycle","config":config,
        "build_commands":[["just","release-host-build"],["just","release-runtime-build",config.backend]],
        "persistent_cache_across_restarts":true,
        "phases":["disk_off_cold","multi_turn_growth","disk_on_empty","disk_on_additional_sources","same_process_l1","restart_l3","concurrent_fill","concurrent_record","lifecycle_under_traffic","low_space"],
        "high_load_pairs":config.concurrency,"outputs":["raw request JSONL","phase status and counters","server/build logs","run.json","REPORT.md","artifact-sha256.txt"]}),
    )
}
