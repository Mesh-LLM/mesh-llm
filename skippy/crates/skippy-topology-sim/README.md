# skippy-topology-sim

Scenario-driven placement simulator for the performance-aware topology planner.

A scenario is a TOML file describing nodes, directed links, a model package,
and a workload intent. The simulator feeds the scenario into
`skippy_coordinator::topology::plan_topology` and scores the resulting plan with
the same cost model the planner uses, so a planner decision can be asserted
against an expectation ("a faster node receives as much work as its capacity
allows", "a slow link rejects the TPOT target") in CI without a cluster.
The `execution` layer adds a discrete pipeline model over a chosen plan —
per-stage service times, per-hop latency and activation transfer, and the
serial vs pipelined decode regimes — calibrated against
[`mesh/docs/BENCHMARKS.md`](../../../mesh/docs/BENCHMARKS.md).

This crate owns scenario parsing, simulation, and the calibrated cost model
used to check the planner. It does not own the planner itself (that is
[`skippy-coordinator`](../skippy-coordinator/README.md)), and it loads no
models, starts no servers, and opens no sockets.

Primary consumers are the planner's scenario tests and CI, plus contributors
reasoning about a placement before renting hardware.

## Usage

Scenarios live in `scenarios/`; each one is also exercised by the tests:

```bash
cargo test -p skippy-topology-sim
```

Design rationale is in
[`PERFORMANCE_AWARE_TOPOLOGY_PLANNER.md`](../../../mesh/docs/design/PERFORMANCE_AWARE_TOPOLOGY_PLANNER.md).
