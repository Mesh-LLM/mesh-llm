#!/usr/bin/env python3
"""Run a command with a portable wall-clock limit and process-group cleanup."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


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


def cleanup_completed_group(process: subprocess.Popen[bytes]) -> None:
    """Stop descendants before returning a completed agent's workspace to its caller."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    # The group leader has already exited; waiting on it cannot wait for its
    # descendants. Give those children a short grace, then kill survivors.
    time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 10
    while True:
        # Orphaned zombies cannot mutate files and may await init's reaper.
        rows = subprocess.check_output(["ps", "-axo", "pgid=,stat="], text=True)
        if not any(int(fields[0]) == process.pid and not fields[1].startswith("Z")
                   for row in rows.splitlines() if len(fields := row.split()) == 2):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("command process group did not stop before workspace handoff")
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
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
