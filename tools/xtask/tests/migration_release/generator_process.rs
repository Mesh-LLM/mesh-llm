//! Unix fixture cleanup for descendants that stay in the generator's group.
//! Polling and cleanup have separate budgets; spawn and filesystem operations
//! are synchronous, so this is not a hard wall-clock or task19 supervisor API.
//! Keep the leader unreaped until the last group signal, reserving its PGID.

use crate::support::Stage;
use std::error::Error;
use std::fs;
use std::os::unix::process::CommandExt as _;
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[path = "generator_process_tests.rs"]
mod tests;

pub(super) const GENERATOR_DEADLINE: Duration = Duration::from_secs(90);
const CONTROL_BUDGET: Duration = Duration::from_secs(1);
const CLEANUP_BUDGET: Duration = Duration::from_secs(2);
const POLL: Duration = Duration::from_millis(10);
type FixtureResult<T> = Result<T, Box<dyn Error>>;

fn reap_until(child: &mut Child, until: Instant) -> FixtureResult<ExitStatus> {
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= until {
            return Err(format!(
                "PID {} was not reaped before the polling deadline",
                child.id()
            )
            .into());
        }
        thread::sleep(POLL);
    }
}

fn control_output(command: &mut Command, stage: &Stage, until: Instant) -> FixtureResult<Output> {
    let stdout = stage.path().join("control.stdout");
    let stderr = stage.path().join("control.stderr");
    let mut child = command
        .stdout(fs::File::create(&stdout)?)
        .stderr(fs::File::create(&stderr)?)
        .stdin(Stdio::null())
        .spawn()?;
    let status = match reap_until(&mut child, until) {
        Ok(status) => status,
        Err(error) => {
            let cleanup = child
                .kill()
                .map_err(Into::into)
                .and_then(|()| reap_until(&mut child, Instant::now() + CLEANUP_BUDGET));
            return match cleanup {
                Ok(_) => Err(format!(
                    "control command {:?}: {error}; child stopped and reaped",
                    command.get_program()
                )
                .into()),
                Err(cleanup) => Err(format!(
                    "control command {:?}: {error}; CLEANUP FAILED: {cleanup}",
                    command.get_program()
                )
                .into()),
            };
        }
    };
    Ok(Output {
        status,
        stdout: fs::read(stdout)?,
        stderr: fs::read(stderr)?,
    })
}

fn group_has_live_members(group: u32, stage: &Stage, until: Instant) -> FixtureResult<bool> {
    let output = control_output(
        Command::new("/bin/ps").args(["-A", "-o", "pid=,pgid=,stat="]),
        stage,
        until.min(Instant::now() + CONTROL_BUDGET),
    )?;
    if !output.status.success() {
        return Err(format!(
            "group probe failed: {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let mut live = false;
    let mut leader_seen = false;
    for line in std::str::from_utf8(&output.stdout)?.lines() {
        let mut fields = line.split_whitespace();
        let pid: u32 = fields.next().ok_or("ps omitted PID")?.parse()?;
        let pgid: u32 = fields.next().ok_or("ps omitted PGID")?.parse()?;
        let state = fields.next().ok_or("ps omitted process state")?;
        if pgid == group {
            leader_seen |= pid == group;
            live |= !state.starts_with('Z');
        }
    }
    if !leader_seen {
        return Err(format!("unreaped generator leader {group} missing from group probe").into());
    }
    Ok(live)
}

struct GeneratorChild<'stage> {
    child: Child,
    stage: &'stage Stage,
    armed: bool,
}

impl GeneratorChild<'_> {
    fn finish(&mut self, until: Instant) -> FixtureResult<ExitStatus> {
        loop {
            if Instant::now() >= until {
                return Err("generator polling deadline exceeded".into());
            }
            if !group_has_live_members(self.child.id(), self.stage, until)? {
                let status = reap_until(&mut self.child, until)?;
                self.armed = false;
                return Ok(status);
            }
            thread::sleep(POLL);
        }
    }

    fn stop(&mut self) -> FixtureResult<()> {
        self.stop_with("/bin/kill").map_err(|error| {
            format!(
                "generator group {} cleanup failed: {error}",
                self.child.id()
            )
            .into()
        })
    }

    fn stop_with(&mut self, killer: &str) -> FixtureResult<()> {
        if !self.armed {
            return Ok(());
        }
        let until = Instant::now() + CLEANUP_BUDGET;
        let group = format!("-{}", self.child.id());
        let output = control_output(
            Command::new(killer).args(["-KILL", "--", &group]),
            self.stage,
            until.min(Instant::now() + CONTROL_BUDGET),
        )?;
        if !output.status.success() {
            return Err(format!(
                "could not stop generator group {group}: {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        self.finish(until)?;
        Ok(())
    }
}

impl Drop for GeneratorChild<'_> {
    fn drop(&mut self) {
        if self.armed
            && let Err(error) = self.stop()
        {
            eprintln!("generator fixture CLEANUP FAILED: {error}");
            if !thread::panicking() {
                panic!("generator fixture cleanup failed: {error}");
            }
        }
    }
}

pub(super) fn run_generator(
    command: &mut Command,
    stage: &Stage,
    deadline: Duration,
) -> FixtureResult<Output> {
    let until = Instant::now() + deadline;
    let stdout = stage.path().join("generator.stdout");
    let stderr = stage.path().join("generator.stderr");
    command.stdout(Stdio::from(fs::File::create(&stdout)?));
    command.stderr(Stdio::from(fs::File::create(&stderr)?));
    command.process_group(0);
    let mut child = GeneratorChild {
        child: command.spawn()?,
        stage,
        armed: true,
    };
    let status = match child.finish(until) {
        Ok(status) => status,
        Err(error) => {
            let context = format!(
                "generator group {} failed with {}ms polling budget: {error}",
                child.child.id(),
                deadline.as_millis()
            );
            return match child.stop() {
                Ok(()) => Err(format!("{context}; group stopped and leader reaped").into()),
                Err(cleanup) => Err(format!("{context}; CLEANUP FAILED: {cleanup}").into()),
            };
        }
    };
    Ok(Output {
        status,
        stdout: fs::read(stdout)?,
        stderr: fs::read(stderr)?,
    })
}
