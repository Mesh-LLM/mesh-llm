#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_runtime(
    root: Path,
    os_name: str,
    arch: str,
    backend: str,
    cuda_major: str = "",
) -> Path:
    expected_kind = {"cuda-blackwell": "cuda", "hip": "rocm"}.get(backend, backend)
    matches = []
    if root.is_dir():
        for manifest_path in sorted(root.glob("*/manifest.json")):
            runtime = json.loads(manifest_path.read_text(encoding="utf-8"))["runtime"]
            platform = runtime["platform"]
            runtime_backend = runtime["backend"]
            if platform["os"] != os_name or platform["arch"] != arch:
                continue
            if runtime_backend["kind"] != expected_kind:
                continue
            if expected_kind == "cuda" and cuda_major:
                if str(runtime_backend.get("cuda", {}).get("toolkit_major", "")) != cuda_major:
                    continue
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        rendered = ", ".join(str(path) for path in matches) or "none"
        raise ValueError(
            f"expected exactly one native runtime for {os_name}/{arch}/{backend}; "
            f"found {rendered} under {root}"
        )
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--os", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--cuda-major", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(select_runtime(args.root, args.os, args.arch, args.backend, args.cuda_major))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
