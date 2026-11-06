# Local and CI llama upgrade repair

From a clean trusted checkout with the canary prerequisites, configured Goose,
and the complete prewarmed immutable model cache:

```sh
just llama-upgrade <40-hex-upstream-sha> metal /absolute/model-cache
# Linux native producer:
just llama-upgrade <40-hex-upstream-sha> cuda /absolute/model-cache
```

This enters the same typed repair/build owner used by CI, freezes the Mesh and
controller revisions, and writes its request, recovery source and diagnostics
under a new temporary directory outside the checkout. The child receives no
GitHub publication token, uses a private home and Cargo home, and disables Git
credential helpers. It retains the installed rustup toolchain location. Provide
Goose provider authentication through its environment.
A successful local build exports an **uncertified** candidate bundle;
the complete distributed family pass and fresh independent verification are
still required for publication. It never downloads a missing model.

Use `cargo xtool automation canary-receipts local-upgrade --help` for a separate
controller checkout or evidence parent. `--plan` prints the frozen request without
starting Goose, building, downloading dependencies or creating evidence.

During repair, local developers and the agent use the same checks:

```sh
just llama-upgrade-check environment metal
just llama-upgrade-check fast metal
just llama-upgrade-check full metal
```

`environment` restores the controller's locked native-test Python project with
an existing Python 3.12+ interpreter. The agent inherits that interpreter; system
pip installs cannot repair this environment. `fast` replays the durable patch queue
at the requested pin, builds the native link closure/probes and Python test driver,
checks upstream Python tests, builds self-contained Rust binaries, verifies
their defined native model-open symbols, and records source and binary identities,
including matching Rust/native ABI versions. The identity includes committed
and uncommitted Mesh and llama.cpp source, so a later source edit requires
recompilation before smoke checks. It then runs dense split/chain and recurrent
state checks,
with GPU layers and Flash Attention enabled on the native producer backend.
The controller-owned focused smoke explicitly targets the selected checkout, so
historical revisions do not need the new smoke switches. Focused and canonical
smokes reuse the fingerprinted executables instead of rebuilding them.

`full` repeats those focused checks, builds every upstream test executable, then validates generated equality and
idempotence, runs the unfiltered native suite, canonical Skippy smoke, CPU workload
oracles, qualified System One and Laya checks. Existing early real-model and
distributed family gates follow in the canary build owner. Focused success is
never a certification result. These commands do not change the upstream pin.

Live output contains the active step and elapsed time. Complete bounded stdout
and stderr logs and JSON results live in the state's `checks/` directory.
`last-check.json` records the failed command, working directory, failure class
and log paths, plus the controlled environment and removed variables for reproduction. Dependency or process-supervision failures stop
with status 125 rather than consuming another coding turn. An isolated Metal
fixture SIGBUS gets one diagnostic fixture retry and a complete unfiltered native
rerun. Ordinary assertions and mixed failures are not retried.

A downloaded GitHub job log alone does not establish a stall: the incident that
prompted this change returned an incomplete snapshot. Correlate the active step,
heartbeat and runner process state. If runner access is unavailable, report
liveness as unconfirmed rather than inferring a hang from missing log lines.

Each coding turn retains an explicitly unverified source checkpoint. After
supervised termination, including cancellation, the outer owner also attempts a
final checkpoint after child cleanup. Recovery includes the Mesh diff and native
changes against their recorded upstream base, including native commits not yet
exported to the durable patch queue. If the native marker is missing or its commit
is unavailable, recovery records the whole tracked native tree as a patch from
an empty tree and archives untracked files separately. The manifest marks this
fallback with `full_tracked_tree: true`; it remains unverified diagnostic source.
Recovery failure never changes the original
exit status. Force termination or runner loss can still prevent final recovery;
earlier checkpoints remain available.

CI uploads the small recovery package before verbose diagnostics. The restored
Python environment is excluded from evidence uploads. Recovery artifacts never
authorize candidate admission or publication.
