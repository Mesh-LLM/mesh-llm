# split-frontier jobs

Every-frontier split verification, stored and run like the agentx jobs
(`jy repos run mesh-llm split-frontier-smoke --ref pr:N --node <host>`).
Results land in the mesh-llm Benchmarks series per job.

## Why

The canary certifies families at fixed shapes; nothing systematically
exercises every possible stage frontier of one model. Frontier-local defects
(misplaced graph helpers, dropped state halves, per-layer API drift) are
exactly the class that survived certification on the v0.7.0 upgrade and cost
days of CI archaeology afterwards.

## What runs

`just skippy-frontier-matrix <model>` drives the typed facade
(`skippy/skippy-api-rs`, see PR #2481): a whole-model reference activation
frame, then every 2-stage split `[0..k / k..N)` for k in 1..N-1 plus 3-stage
chains at a coarse stride, each compared against the reference within a small
f32 tolerance. Per-frontier state-effect counts come from the typed plan, so a
frontier that silently drops recurrent/SWA state is visible in the row.

## Models

- Smoke: SmolLM2-135M (dense, seconds per matrix).
- Certify: SmolLM2-135M plus granite-4.0-h-350m — the hybrid/recurrent leg is
  mandatory; every historical frontier defect lived there.

## Backends

Templates build for the machine's backend (Metal on macOS, CUDA on aarch64
Linux, CPU elsewhere). Run the job on one host per backend family; per-backend
series keep a backend-specific regression visible instead of masked by an
aggregate.

## Registrations

Configuration-only registrations in general-inference/jianyang mirror
agentx #209/#210 and are filed as a follow-up; until then these templates are
the source of truth and can be run ad hoc with `jy repos run`.
