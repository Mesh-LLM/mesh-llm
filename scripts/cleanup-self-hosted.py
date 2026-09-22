#!/usr/bin/env python3
"""Remove only known job outputs after self-hosted artifact upload attempts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess


def targets(env, job, evidence_uploaded, package_uploaded):
    workspace = Path(env['GITHUB_WORKSPACE']).resolve()
    temporary = Path(env['RUNNER_TEMP']).resolve()
    if job not in ('build', 'family'):
        return other_targets(env, job, evidence_uploaded, workspace, temporary)
    root = Path(env['CANARY_SOURCE_ROOT']).absolute()
    if root not in (workspace, workspace / 'canary-source'):
        raise ValueError('source root must be the controller or selected-source checkout')
    run, attempt, pass_id = (env[k] for k in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'CANARY_PASS_ID'))
    if not re.fullmatch(r'[0-9]+', run) or not re.fullmatch(r'[0-9]+', attempt):
        raise ValueError('invalid run identity')
    if not re.fullmatch(r'(repair|verify)-[1-3]', pass_id):
        raise ValueError('invalid pass identity')
    key = f'{run}-{attempt}'
    paths = [(workspace, root / 'target/debug'),
             (workspace, root / '.deps/llama.cpp')]
    if job == 'build':
        native = workspace / f'.deps/llama-{key}-{pass_id}'
        for suffix in ('', '-workloads', f'-verification-{key}', f'-verification-{key}-workloads'):
            paths.append((workspace, Path(str(native) + suffix)))
        paths.extend((temporary, temporary / name) for name in
                     (f'canary-previous-{pass_id}', f'canary-feedback-{pass_id}'))
        if package_uploaded:
            paths.append((temporary, temporary / f'canary-export-{key}-{pass_id}'))
        if evidence_uploaded:
            paths.append((workspace, root / f'.deps/llama-canary-state-{key}-{pass_id}'))
    else:
        shard = env['CANARY_SHARD_INDEX']
        if not re.fullmatch(r'[0-9]+', shard):
            raise ValueError('invalid shard identity')
        paths.extend([(workspace, workspace / f'.deps/canary-input-{key}-{pass_id}-{shard}'),
                      (workspace, root / '.deps/canary-workload-oracles'),
                      (workspace, workspace / 'ci/canary-python/.venv')])
        if evidence_uploaded:
            paths.append((workspace, workspace / f'target/canary-evidence-{key}/{pass_id}-{shard}'))
    return paths


def other_targets(env, job, uploaded, workspace, temporary):
    paths = []
    if job == 'replay':
        paths = [(workspace, workspace / 'ci/agentic-replay-nightly/.venv'),
                 (temporary, temporary / 'agentic-replay-history'),
                 (temporary, temporary / 'agentic-replay-worktrees')]
        if uploaded:
            paths.append((temporary, temporary / 'agentic-replay-artifacts'))
    elif job == 'cuda-release':
        # Native build is a declared Actions cache: its explicit save precedes cleanup.
        paths = [(workspace, workspace / name) for name in
                 ('target', '.deps/llama.cpp', '.deps/llama-build')]
        if uploaded:
            paths.append((workspace, workspace / 'dist/native-runtimes'))
    elif job == 'smoke':
        for key in ('CLEANUP_ARTIFACT_PATH', 'CLEANUP_BINARY_PATH'):
            value = Path(env[key])
            if '..' in value.parts:
                raise ValueError('parent traversal in smoke output')
            path = value if value.is_absolute() else workspace / value
            # Only conventional generated output trees are eligible.
            relative = path.relative_to(workspace)
            if not relative.parts or relative.parts[0] not in ('target', 'ci-artifacts'):
                raise ValueError('smoke output is not in a generated output tree')
            paths.append((workspace, path))
            if key == 'CLEANUP_BINARY_PATH':
                paths.append((workspace, path.parent / 'native-runtimes'))
    elif job == 'runner-contract':
        paths = [(workspace, workspace / 'target')]
    else:
        raise ValueError('unknown cleanup profile')
    return paths


def validate_path(base, path):
    """Reject symlinked parents; unlink a leaf symlink without following it."""
    if not path.is_relative_to(base) or path == base:
        raise ValueError(f'cleanup escaped its owning directory: {path}')
    for parent in path.parents:
        if parent == base:
            break
        if parent.is_symlink():
            raise ValueError(f'cleanup parent is a symlink: {parent}')


def cleanup(paths):
    # Validate the complete list before deleting anything.
    for base, path in paths:
        validate_path(base, path)
    for _, path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
        print(f'Cleaned job output: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('build', 'family', 'replay', 'cuda-release', 'smoke', 'runner-contract'), required=True)
    parser.add_argument('--evidence-uploaded', choices=('true', 'false'), required=True)
    parser.add_argument('--package-uploaded', choices=('true', 'false'), default='false')
    args = parser.parse_args()
    cleanup(targets(os.environ, args.job, args.evidence_uploaded == 'true', args.package_uploaded == 'true'))
    if args.job == 'replay':
        subprocess.run(['git', '-C', os.environ['GITHUB_WORKSPACE'], 'worktree', 'prune'], check=True)


if __name__ == '__main__':
    main()
