"""Resolve serving arguments for evaluation binaries from different revisions.

Eval scripts load this file with runpy.run_path beside their own __file__. That
works both for direct script execution and importlib-loaded tests without adding
the eval directory to sys.path. This compatibility fallback belongs only to
evaluation tooling; it does not add aliases to the product CLI.
"""

from __future__ import annotations

from pathlib import Path
import subprocess


HELP_TIMEOUT_SECONDS = 5.0


def serve_args(
    binary: Path,
    *,
    binary_transport: bool = False,
    worker_only: bool = False,
) -> list[str]:
    """Probe this executable and return its subcommand and serving-mode flags.

    Probe per launch rather than caching by path: an evaluation may rebuild a
    binary in place between cells, and an old baseline may already use `serve`.
    Legacy binary serving derives worker behavior from the stage configuration.
    """
    if worker_only and not binary_transport:
        raise ValueError("worker-only evaluation requires binary transport")
    legacy = "serve-binary" if binary_transport else "serve-openai"
    failures = []
    for command in ("serve", legacy):
        try:
            result = subprocess.run(
                [str(binary), command, "--help"],
                capture_output=True,
                text=True,
                timeout=HELP_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{command}: help timed out after {HELP_TIMEOUT_SECONDS:g}s")
            continue
        except OSError as error:
            raise RuntimeError(f"cannot probe Skippy evaluation binary {binary}: {error}") from error
        if result.returncode == 0:
            if command != "serve":
                return [command]
            args = [command]
            if binary_transport:
                args.extend(["--stage-transport", "binary"])
            if worker_only:
                args.append("--worker-only")
            return args
        detail = (result.stderr or result.stdout or "no diagnostic").strip()[:500]
        failures.append(f"{command}: exit {result.returncode}: {detail}")
    raise RuntimeError(
        f"Skippy evaluation binary {binary} supports neither serve nor {legacy}: "
        + "; ".join(failures)
    )
