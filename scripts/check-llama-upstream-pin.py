#!/usr/bin/env python3
"""Reject PRs that move the pinned llama.cpp revision backwards.

The script is intended to run from a protected workflow checkout.  It reads
the pin files from the immutable base and head commits, then builds a fresh
blobless mirror of the upstream repository.  Fetching both pins without a
depth limit is deliberate: a shallow or incomplete local upstream checkout
must never turn an unknown ancestry result into a passing check.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PIN_PATH = "third_party/llama.cpp/upstream.txt"
DEFAULT_UPSTREAM_URL = "https://github.com/ggml-org/llama.cpp.git"
UPSTREAM_FETCH_TIMEOUT_SECONDS = 300


class PinGuardError(RuntimeError):
    """Raised when the pin comparison cannot be proved safe."""


def git(
    repo: Path,
    *args: str,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        timeout_detail = f" after {timeout:g} seconds" if timeout is not None else ""
        raise PinGuardError(
            f"git {' '.join(args)} timed out{timeout_detail}; "
            "the guard cannot prove llama.cpp upstream ancestry and will fail closed"
        ) from error
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no git output"
        raise PinGuardError(f"git {' '.join(args)} failed: {detail}")
    return result


def validate_revision(value: str, label: str) -> str:
    if not SHA_RE.fullmatch(value):
        raise PinGuardError(f"{label} must be a lowercase 40-character SHA: {value!r}")
    return value


def read_pin(repo: Path, revision: str, label: str) -> str:
    tree = git(repo, "ls-tree", "-z", revision, "--", PIN_PATH)
    entries = [entry for entry in tree.stdout.split("\0") if entry]
    if len(entries) != 1:
        raise PinGuardError(
            f"{label} commit {revision} must contain exactly one {PIN_PATH} entry"
        )
    metadata, path = entries[0].split("\t", 1)
    metadata_parts = metadata.split()
    if (
        path != PIN_PATH
        or len(metadata_parts) != 3
        or metadata_parts[0] != "100644"
        or metadata_parts[1] != "blob"
    ):
        raise PinGuardError(
            f"{label} commit {revision} {PIN_PATH} must be a regular 100644 blob"
        )
    result = git(repo, "show", f"{revision}:{PIN_PATH}")
    pin = result.stdout.strip()
    if not SHA_RE.fullmatch(pin):
        raise PinGuardError(
            f"{label} commit {revision} has an invalid {PIN_PATH} value: {pin!r}"
        )
    return pin


def fetch_upstream(upstream_url: str, base_pin: str, proposed_pin: str) -> tempfile.TemporaryDirectory[str]:
    checkout = tempfile.TemporaryDirectory(prefix="mesh-llm-llama-pin-")
    upstream = Path(checkout.name) / "upstream.git"
    upstream.mkdir()
    try:
        git(upstream, "init", "--bare", "--quiet")
        git(upstream, "remote", "add", "origin", upstream_url)
        fetched = git(
            upstream,
            "-c",
            "protocol.version=2",
            "fetch",
            "--no-tags",
            "--filter=blob:none",
            "origin",
            base_pin,
            proposed_pin,
            check=False,
            timeout=UPSTREAM_FETCH_TIMEOUT_SECONDS,
        )
        if fetched.returncode != 0:
            detail = fetched.stderr.strip() or fetched.stdout.strip() or "no git output"
            raise PinGuardError(
                "unable to fetch both llama.cpp upstream pins; "
                "the guard cannot prove ancestry and will fail closed: "
                f"{detail}"
            )

        shallow = git(upstream, "rev-parse", "--is-shallow-repository").stdout.strip()
        if shallow == "true":
            # The fetch above intentionally has no depth limit.  Keep this
            # fallback for mirrors configured as shallow by their server.
            unshallow = git(
                upstream,
                "fetch",
                "--no-tags",
                "--unshallow",
                "origin",
                check=False,
                timeout=UPSTREAM_FETCH_TIMEOUT_SECONDS,
            )
            if unshallow.returncode != 0 or git(
                upstream, "rev-parse", "--is-shallow-repository"
            ).stdout.strip() == "true":
                detail = unshallow.stderr.strip() or unshallow.stdout.strip() or "no git output"
                raise PinGuardError(
                    "llama.cpp upstream history is shallow after fetch; "
                    f"cannot prove ancestry: {detail}"
                )

        for pin in (base_pin, proposed_pin):
            if git(upstream, "cat-file", "-e", f"{pin}^{{commit}}", check=False).returncode != 0:
                raise PinGuardError(
                    f"llama.cpp upstream object {pin} is unavailable after fetch; "
                    "cannot prove ancestry"
                )

        missing = git(
            upstream,
            "rev-list",
            "--missing=print",
            base_pin,
            proposed_pin,
            check=False,
        )
        if missing.returncode != 0:
            detail = missing.stderr.strip() or missing.stdout.strip() or "no git output"
            raise PinGuardError(f"cannot walk llama.cpp upstream history: {detail}")
        if any(line.startswith("?") for line in missing.stdout.splitlines()):
            raise PinGuardError(
                "llama.cpp upstream history is incomplete after fetch; "
                "cannot prove ancestry"
            )
        return checkout
    except Exception:
        checkout.cleanup()
        raise


def check_pin(repository: Path, base_revision: str, head_revision: str, upstream_url: str) -> int:
    base_revision = validate_revision(base_revision, "base revision")
    head_revision = validate_revision(head_revision, "head revision")
    for revision in (base_revision, head_revision):
        if git(repository, "cat-file", "-e", f"{revision}^{{commit}}", check=False).returncode != 0:
            raise PinGuardError(f"repository is missing {revision}; cannot inspect PR pin")

    merge_base_result = git(
        repository,
        "merge-base",
        base_revision,
        head_revision,
        check=False,
    )
    if merge_base_result.returncode != 0:
        detail = merge_base_result.stderr.strip() or merge_base_result.stdout.strip() or "no git output"
        raise PinGuardError(f"cannot determine the PR merge base: {detail}")
    merge_base = merge_base_result.stdout.strip()
    if not SHA_RE.fullmatch(merge_base):
        raise PinGuardError(f"git returned an invalid PR merge base: {merge_base!r}")

    base_pin = read_pin(repository, merge_base, "PR merge-base")
    proposed_pin = read_pin(repository, head_revision, "head")
    print(f"PR merge-base commit:   {merge_base}")
    print(f"merge-base llama.cpp pin: {base_pin}")
    print(f"proposed llama.cpp pin: {proposed_pin}")

    if base_pin == proposed_pin:
        print("llama.cpp upstream pin is unchanged")
        return 0

    checkout = fetch_upstream(upstream_url, base_pin, proposed_pin)
    try:
        upstream = Path(checkout.name) / "upstream.git"
        proposed_ancestor = git(
            upstream,
            "merge-base",
            "--is-ancestor",
            proposed_pin,
            base_pin,
            check=False,
        )
        if proposed_ancestor.returncode == 0:
            raise PinGuardError(
                "PR moves third_party/llama.cpp/upstream.txt backward: "
                f"{proposed_pin} is an ancestor of the base pin {base_pin}"
            )
        if proposed_ancestor.returncode != 1:
            detail = proposed_ancestor.stderr.strip() or proposed_ancestor.stdout.strip() or "no git output"
            raise PinGuardError(f"cannot compare llama.cpp upstream pins: {detail}")

        base_ancestor = git(
            upstream,
            "merge-base",
            "--is-ancestor",
            base_pin,
            proposed_pin,
            check=False,
        )
        if base_ancestor.returncode == 0:
            print("llama.cpp upstream pin moves forward")
            return 0
        if base_ancestor.returncode != 1:
            detail = base_ancestor.stderr.strip() or base_ancestor.stdout.strip() or "no git output"
            raise PinGuardError(f"cannot compare llama.cpp upstream pins: {detail}")
        raise PinGuardError(
            "PR proposes a divergent llama.cpp upstream history; "
            "the guard cannot prove a forward pin update"
        )
    finally:
        checkout.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_revision", help="PR base repository commit")
    parser.add_argument("head_revision", help="PR head repository commit")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="mesh-llm checkout containing the base and head commits",
    )
    parser.add_argument(
        "--upstream-url",
        default=DEFAULT_UPSTREAM_URL,
        help="llama.cpp git URL (for hermetic tests, a local repository URL)",
    )
    args = parser.parse_args()
    try:
        return check_pin(args.repository.resolve(), args.base_revision, args.head_revision, args.upstream_url)
    except PinGuardError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
