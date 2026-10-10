# Canary receipt fixture freeze

Status: source-derived inputs and expected decisions frozen before the Rust
implementation. Executable legacy oracle captures are PENDING global queue
assignment. These files are not recorded execution results.

Base: `a4e04070db2c6b8e2644a40f5c4f207789df6c2d`.
Legacy receipt/results/aggregate source:
`scripts/llama-canary-family-evidence.py`, SHA-256
`89dc3f313c95482e7022b58d7a274eb724e18eb4aca20ae9a8907ae45245e5c2`.
Legacy test source: `scripts/tests/test_llama_canary_family_evidence.py`, SHA-256
`280bb2ac4b504713da600b2d11cb86ec43bf970fe6826892eef16372f63c0f05`.

`plan.json` reproduces the two-family setup at test lines 36-42 with matrix
order deliberately reversed. This is an input projection, not planner output.
`identity.json` is a synthetic already-verified producer identity projection;
its schema 3 is NOT a worker receipt version. Package verification is an
explicit upstream prerequisite, not simulated by this fixture.
`dense.jsonl`, `hybrid.jsonl` derive from test lines 69-71.
`pretty.jsonl`, `embedding.jsonl`, `battery.jsonl`, and `projector.jsonl` derive
from the tests at lines 129-153, 300-331 and the result contract at legacy
lines 514-556. Bytes are hand-authored independent inputs, never port output.

`cases.json` freezes expected acceptance/rejection and selected attempts from
legacy lines 559-616 and tests 155-298. Mutations apply to disposable fixture
copies only. A current attempt of 4, producer attempt of 2, and newer worker
attempt of 3 distinguish producer identity, worker freshness, and aggregate
reruns. Every variant leaves an unchanged hybrid sibling unless stated.

The tests must construct worker receipt JSON independently of the Rust receipt
writer. Raw result SHA-256 comes from fixture bytes, not serialized parsed JSON.
The writer's independent expected receipt bytes will be frozen after hashing
these inputs and before implementation. No Python helper may be added.

Pending oracle capture: run the existing named legacy FamilyEvidenceTests in
the assigned serial validation slot with PYTHONDONTWRITEBYTECODE=1 and retained
stdout/stderr/status. Those tests create their own package fixtures. Do not run
pack, restore, certification, repair, planners, or model-dependent checks.
Exact new-fixture differential capture remains pending; source-derived verdicts
must never be relabelled as observed legacy stdout or status.
