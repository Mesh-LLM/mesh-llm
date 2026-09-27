#!/usr/bin/env python3
"""Run a command with a portable wall-clock limit and process-group cleanup."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


class CleanupError(RuntimeError):
    """The supervised command finished, but live descendants could not be stopped."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cleanup-on-exit", action="store_true",
                        help="terminate remaining group members even when the command completes")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


def live_group_members(pgid: int) -> int:
    """Count non-zombie members that can still mutate the handed-off workspace."""
    try:
        rows = subprocess.check_output(["ps", "-axo", "pgid=,stat="], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise CleanupError(f"could not inspect process group {pgid}: {error}") from error
    members = 0
    for row in rows.splitlines():
        fields = row.split()
        if len(fields) == 2 and int(fields[0]) == pgid and not fields[1].startswith("Z"):
            members += 1
    return members


def cleanup_completed_group(process: subprocess.Popen[bytes]) -> None:
    """Stop descendants before returning a completed agent's workspace to its caller."""
    term_error: PermissionError | None = None
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as error:
        term_error = error
    # The group leader has already exited; waiting on it cannot wait for its
    # descendants. Give those children a short grace, then kill survivors.
    time.sleep(0.1)
    inspection_error: CleanupError | None = None
    try:
        members = live_group_members(process.pid)
    except CleanupError as error:
        # Some sandboxes deny process-table inspection. A successful SIGKILL
        # still gives a bounded handoff; an EPERM below remains fail-closed.
        inspection_error = error
        members = -1
    if not members:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError as error:
        try:
            members = live_group_members(process.pid)
        except CleanupError as inspect_error:
            raise CleanupError(
                f"permission denied stopping the process group and inspection failed: {inspect_error}"
            ) from error
        if not members:
            return
        prior = f" after SIGTERM was denied ({term_error})" if term_error else ""
        raise CleanupError(
            f"permission denied stopping {members} live process-group member(s){prior}"
        ) from error
    if inspection_error is not None:
        # SIGKILL was accepted for the complete process group. Without process
        # table access there is nothing more the supervisor can observe.
        time.sleep(0.1)
        return
    deadline = time.monotonic() + 10
    while True:
        if not live_group_members(process.pid):
            return
        if time.monotonic() >= deadline:
            raise CleanupError("command process group did not stop before workspace handoff")
        time.sleep(0.05)


def main() -> int:
    """Supervise one process group, preserving completed status and bounded cancellation."""
    args = parse_args()
    received_signal: int | None = None

    def request_termination(signum: int, _frame: object) -> None:
        """Record the first signal without reentering process construction or wait locks."""
        # A handler can interrupt Popen construction or wait's internal lock.
        # Only record intent here; never wait, print, or clean up reentrantly.
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum

    previous_handlers = {
        signum: signal.signal(signum, request_termination)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        # Commands are argument-driven: inheriting a manifest loop's stdin
        # could silently consume later planned rows, so children receive EOF.
        process = subprocess.Popen(
            args.command,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + args.seconds
        while received_signal is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                returncode = process.poll()
                if returncode is not None:
                    if args.cleanup_on_exit:
                        cleanup_completed_group(process)
                    return returncode
                print(
                    f"{args.label} timed out after {args.seconds}s; terminating process group",
                    file=sys.stderr,
                )
                terminate_group(process)
                return 124
            try:
                returncode = process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
            if received_signal is None:
                if args.cleanup_on_exit:
                    cleanup_completed_group(process)
                return returncode

        print(
            f"{args.label} received signal {received_signal}; terminating process group",
            file=sys.stderr,
        )
        terminate_group(process)
        return 128 + received_signal
    except CleanupError as error:
        print(f"{args.label} infrastructure cleanup failed: {error}", file=sys.stderr)
        return 125
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
