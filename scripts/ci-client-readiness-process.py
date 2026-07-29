#!/usr/bin/env python3
"""Windows process-group support for the client-readiness smoke."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from pathlib import Path
from typing import Sequence


def creationflags_for_platform(is_windows: bool) -> int:
    if not is_windows:
        return 0
    return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def launch(
    command: Sequence[str], pid_file: Path, log_file: Path, *, is_windows: bool
) -> int:
    with log_file.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags_for_platform(is_windows),
        )
        pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        return process.wait()


def request_ctrl_break(pid: int) -> None:
    ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
    if ctrl_break is None:
        raise RuntimeError("CTRL_BREAK_EVENT is only available on Windows")
    os.kill(pid, ctrl_break)


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def force_stop(pid: int) -> None:
    subprocess.run(
        ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run")
    run.add_argument("--pid-file", type=Path, required=True)
    run.add_argument("--log", type=Path, required=True)
    run.add_argument("program", nargs=argparse.REMAINDER)

    for name in ("ctrl-break", "is-running", "force-stop"):
        command = subcommands.add_parser(name)
        command.add_argument("--pid", type=int, required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        if not args.program:
            raise SystemExit("run requires a program after --")
        return launch(args.program, args.pid_file, args.log, is_windows=os.name == "nt")
    if args.command == "ctrl-break":
        request_ctrl_break(args.pid)
        return 0
    if args.command == "is-running":
        return 0 if is_running(args.pid) else 1
    if args.command == "force-stop":
        force_stop(args.pid)
        return 0
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
