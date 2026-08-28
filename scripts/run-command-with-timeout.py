#!/usr/bin/env python3
"""Run a command with a portable wall-clock limit and process-group cleanup."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--label", required=True)
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


def main() -> int:
    args = parse_args()
    # The wrapper is commonly invoked from manifest-reading shell loops. A
    # child must never inherit and consume the loop's stdin, because doing so
    # can silently drop later planned rows. Commands in this harness are fully
    # argument-driven, so EOF is the only valid stdin contract.
    process = subprocess.Popen(
        args.command,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    def terminate_on_signal(signum: int, _frame: object) -> None:
        # Prevent a second cancellation signal from interrupting cleanup and
        # leaving descendants behind on the persistent runner.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print(
            f"{args.label} received signal {signum}; terminating process group",
            file=sys.stderr,
        )
        terminate_group(process)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, terminate_on_signal)
    signal.signal(signal.SIGTERM, terminate_on_signal)
    try:
        return process.wait(timeout=args.seconds)
    except subprocess.TimeoutExpired:
        print(
            f"{args.label} timed out after {args.seconds}s; terminating process group",
            file=sys.stderr,
        )
        terminate_group(process)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
