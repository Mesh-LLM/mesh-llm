#!/usr/bin/env python3
"""Resolve protected pre-extraction package batches in a candidate executor.

This runs only after candidate source checkout, where Cargo tests already run.
It does not change the protected planner, its catalogs, or matrix worker count.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

# Temporary source migration: preserve every predecessor's test responsibility.
# In particular skippy-model-package is reused for acquisition after extraction.
SUCCESSORS = {
    'mesh-llm-gpu-bench': ['skippy-gpu-bench'],
    'mesh-llm-guardrails': ['skippy-guardrails'],
    'mesh-llm-hardware-profile': ['skippy-hardware-profile'],
    'mesh-llm-native-runtime': ['skippy-native-runtime'],
    'mesh-llm-runtime-install': ['skippy-runtime-install'],
    'model-artifact': ['skippy-model-artifact'],
    'model-hf': ['skippy-model-hf', 'skippy-hf-hub'],
    'model-package': ['skippy-model-package'],
    'model-ref': ['skippy-model-ref'],
    'model-resolver': ['skippy-model-resolver'],
    'openai-frontend': ['skippy-openai-frontend'],
    'skippy-model-package': ['skippy-package-builder'],
    'skippy-server': [
        'skippy-serving', 'skippy-api', 'skippy-cli', 'skippy-commands',
        'skippy-config', 'skippy-events',
    ],
    'mesh-llm-host-runtime': [
        'mesh-llm-host-runtime', 'mesh-llm-skippy-adapter', 'mesh-llm-control-api',
        'mesh-llm-membership', 'mesh-llm-transport',
    ],
}


def names(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError('package list must be a nonempty array')
    if any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) for name in value):
        raise ValueError('invalid Cargo package name')
    if len(set(value)) != len(value):
        raise ValueError('duplicate Cargo package name')
    return value


def resolve(requested: list[str], planned: list[str], available: set[str], generation: str) -> list[str]:
    requested = names(requested)
    planned = names(planned)
    if not set(requested).issubset(planned):
        raise ValueError('requested batch contains packages outside the protected plan')
    if generation not in {"legacy", "current"}:
        raise ValueError("unsupported planner package generation")
    migrating = generation == "legacy" and "skippy-package-builder" in available
    result = []
    for name in requested:
        candidates = SUCCESSORS.get(name, [name]) if migrating else [name]
        missing = set(candidates) - available
        if missing:
            raise ValueError(f'planned package {name!r} has missing source owners: {sorted(missing)}')
        for candidate in candidates:
            if candidate in result:
                raise ValueError(f'package resolves more than once: {candidate}')
            result.append(candidate)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', choices=['legacy', 'current'], required=True,
                        help='Package naming generation of the protected plan, not the candidate')
    parser.add_argument('--crates', required=True, help='JSON package array for this batch')
    parser.add_argument('--batches', help='Complete protected JSON batch matrix; defaults to this batch')
    args = parser.parse_args()
    requested = names(json.loads(args.crates))
    planned = requested
    if args.batches is not None:
        batches = json.loads(args.batches)
        if not isinstance(batches, list) or not batches:
            raise ValueError('batch matrix must be a nonempty array')
        planned = names([name for batch in batches for name in names(batch['crates'])])
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--locked', '--no-deps', '--format-version=1'], text=True,
    ))
    members = set(metadata['workspace_members'])
    available = {package['name'] for package in metadata['packages'] if package['id'] in members}
    print(json.dumps(resolve(requested, planned, available, args.generation)))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f'CI package resolution failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error
