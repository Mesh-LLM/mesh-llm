use super::*;
use crate::support::TestResult;

#[test]
fn migration_release_generator_stops_hung_nested_child_without_touching_sentinel() -> TestResult {
    let stage = Stage::new("generator-nested-timeout")?;
    let sentinel_stage = Stage::new("generator-sentinel")?;
    let mut sentinel = GeneratorChild {
        child: Command::new("sleep").arg("10").process_group(0).spawn()?,
        stage: &sentinel_stage,
        armed: true,
    };
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg("sleep 3 & printf '%s' \"$!\" > \"$1/nested.pid\"; exit 0")
        .arg("sh")
        .arg(stage.path());
    let started = Instant::now();

    let result = run_generator(&mut command, &stage, Duration::from_millis(200));

    let elapsed = started.elapsed();
    let sentinel_alive = group_has_live_members(
        sentinel.child.id(),
        &sentinel_stage,
        Instant::now() + CONTROL_BUDGET,
    )?;
    sentinel.stop()?;
    let error = result.expect_err("hung nested child must time out");
    assert!(
        error.to_string().contains("200ms polling budget"),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("group stopped and leader reaped"),
        "{error}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "nested child ran for {elapsed:?}"
    );
    let nested = fs::read_to_string(stage.path().join("nested.pid"))?;
    let nested_status = control_output(
        Command::new("/bin/kill").args(["-0", nested.trim()]),
        &stage,
        Instant::now() + CONTROL_BUDGET,
    )?;
    assert!(
        !nested_status.status.success(),
        "nested child {nested} survived timeout"
    );
    assert!(sentinel_alive, "unrelated sentinel was stopped");
    eprintln!(
        "fixture receipt: child={nested} stopped; sentinel={} alive before cleanup; elapsed={elapsed:?}; {error}",
        sentinel.child.id()
    );
    Ok(())
}

#[test]
fn migration_release_generator_keeps_leader_unreaped_until_group_cleanup() -> TestResult {
    let stage = Stage::new("generator-leader-ownership")?;
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(
            "parent=$$; (sleep 0.1; /bin/ps -p \"$parent\" -o stat= > \"$1/leader.stat\") & exit 0",
        )
        .arg("sh")
        .arg(stage.path());

    let output = run_generator(&mut command, &stage, Duration::from_secs(2))?;

    assert!(output.status.success());
    let leader = fs::read_to_string(stage.path().join("leader.stat"))?;
    assert!(
        leader.trim_start().starts_with('Z'),
        "leader must remain an unreaped zombie while its child lives: {leader:?}"
    );
    Ok(())
}

#[test]
fn migration_release_generator_preserves_exit_and_output_without_live_descendants() -> TestResult {
    let stage = Stage::new("generator-output")?;
    let mut command = Command::new("sh");
    command.args([
        "-c",
        "printf 'fixture stdout'; printf 'fixture stderr' >&2; exit 7",
    ]);

    let output = run_generator(&mut command, &stage, Duration::from_secs(2))?;

    assert_eq!(output.status.code(), Some(7));
    assert_eq!(output.stdout, b"fixture stdout");
    assert_eq!(output.stderr, b"fixture stderr");
    Ok(())
}

#[test]
fn migration_release_generator_control_command_is_stopped_when_polling_expires() -> TestResult {
    let stage = Stage::new("generator-control-timeout")?;
    let mut command = Command::new("/bin/sleep");
    command.arg("3");
    let started = Instant::now();

    let result = control_output(&mut command, &stage, started + Duration::from_millis(100));

    let error = result.expect_err("stalled control command must fail");
    assert!(
        error.to_string().contains("child stopped and reaped"),
        "{error}"
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    Ok(())
}

#[test]
fn migration_release_generator_reports_control_command_failure() -> TestResult {
    let stage = Stage::new("generator-control-failure")?;
    let mut command = Command::new("/bin/kill");
    command.arg("invalid-pid");

    let output = control_output(&mut command, &stage, Instant::now() + CONTROL_BUDGET)?;

    assert!(!output.status.success());
    assert!(!output.stderr.is_empty());
    Ok(())
}

#[test]
fn migration_release_generator_reports_failed_cleanup_without_blocking_drop() -> TestResult {
    let stage = Stage::new("generator-failed-cleanup")?;
    let mut child = GeneratorChild {
        child: Command::new("sleep").arg("5").process_group(0).spawn()?,
        stage: &stage,
        armed: true,
    };
    let started = Instant::now();

    let result = child.stop_with("/usr/bin/false");

    assert_drop_cleans_after_failure(child)?;
    let error = result.expect_err("failed kill must be visible");
    assert!(
        error.to_string().contains("could not stop generator group"),
        "{error}"
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    Ok(())
}

#[test]
fn migration_release_generator_rejects_successful_kill_command_with_survivors() -> TestResult {
    let stage = Stage::new("generator-surviving-cleanup")?;
    let mut child = GeneratorChild {
        child: Command::new("sleep").arg("5").process_group(0).spawn()?,
        stage: &stage,
        armed: true,
    };
    let started = Instant::now();

    let result = child.stop_with("/usr/bin/true");

    let elapsed = started.elapsed();
    assert_drop_cleans_after_failure(child)?;
    assert!(
        result.is_err(),
        "a successful kill command is not proof the group stopped"
    );
    assert!(
        elapsed < Duration::from_secs(4),
        "cleanup blocked for {elapsed:?}"
    );
    Ok(())
}

#[test]
fn migration_release_generator_reports_missing_cleanup_command() -> TestResult {
    let stage = Stage::new("generator-missing-cleanup")?;
    let mut child = GeneratorChild {
        child: Command::new("sleep").arg("5").process_group(0).spawn()?,
        stage: &stage,
        armed: true,
    };
    let missing = stage.path().join("absent-kill");

    let result = child.stop_with(missing.to_str().ok_or("fixture path is not UTF-8")?);

    assert_drop_cleans_after_failure(child)?;
    let error = result.expect_err("missing kill must be visible");
    assert_eq!(
        error
            .downcast_ref::<std::io::Error>()
            .map(std::io::Error::kind),
        Some(std::io::ErrorKind::NotFound)
    );
    Ok(())
}

#[test]
fn migration_release_generator_skips_group_signal_after_leader_reap() -> TestResult {
    let stage = Stage::new("generator-reaped-cleanup")?;
    let mut child = GeneratorChild {
        child: Command::new("sleep").arg("5").process_group(0).spawn()?,
        stage: &stage,
        armed: true,
    };
    child.stop()?;

    let result = child.stop_with("/usr/bin/false");

    result?;
    assert!(!child.armed, "successful cleanup must disarm the guard");
    Ok(())
}

fn assert_drop_cleans_after_failure(mut child: GeneratorChild<'_>) -> TestResult {
    let group = child.child.id();
    let stage = child.stage;
    let retained_ownership = child.armed;
    if !retained_ownership {
        child.child.kill()?;
        reap_until(&mut child.child, Instant::now() + CLEANUP_BUDGET)?;
    }

    drop(child);

    let probe = control_output(
        Command::new("/bin/kill").args(["-0", &group.to_string()]),
        stage,
        Instant::now() + CONTROL_BUDGET,
    )?;
    assert!(
        !probe.status.success(),
        "owned leader {group} survived Drop"
    );
    eprintln!(
        "cleanup receipt: group={group}; retained_ownership={retained_ownership}; leader absent after Drop"
    );
    assert!(
        retained_ownership,
        "cleanup failure disarmed live group {group}; red-run rescue was required"
    );
    Ok(())
}
