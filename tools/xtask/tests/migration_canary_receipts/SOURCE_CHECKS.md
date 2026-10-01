# Read-only correction checks, 2026-09-28

These checks identify bytes and source shape. They do not execute the candidate
or establish behavior, compilation, formatting, lint success or parity.

| Check | Observed result |
| --- | --- |
| Original `CANDIDATE.sha256`, before edits | All 29 entries OK, exit 0 |
| `FROZEN.sha256` and `EXPECTED.sha256`, after edits | All 11 entries OK, exit 0 |
| `od -An -tx1` on four separator fixtures | Actual 0x1c, 0x1d, 0x1e and 0x1f bytes observed at document boundaries |
| `wc -c` on those four fixtures | 331 bytes each, 1324 total |
| `rg -c '^fn migration_canary_receipts'` | Original files retain 6 + 7 + 10 + 13 = 36 tests; new files add 10 + 5 = 15 |
| Original test/support SHA-256 values | All five unchanged from historical CANDIDATE.sha256 |
| `GIT_MASTER=1 GIT_OPTIONAL_LOCKS=0 git diff --exit-code HEAD --` | Exit 0, no tracked-file changes |
| `GIT_MASTER=1 GIT_OPTIONAL_LOCKS=0 git diff --cached --exit-code --` | Exit 0, no staged changes |
| Candidate status | Only candidate-local untracked source, tests, fixtures and handoff records |

The scoped production delta is two adapter names and their serde references,
plus the result-stream whitespace predicate import/call. The test entry adds
explicit shared-text dependency wiring and the two new test modules. New files
are two test modules, six byte fixtures and correction records. CANDIDATE.md and
REGISTRATION.md have correction headers; their historical bodies are retained.
SOURCE_REVIEW.md, the old hash manifest and frozen fixture evidence are unchanged.

No executable validation ran. The exact pending commands, expected selected
counts, differential controls and unresolved top-level duplicate decision are
in CORRECTION.md. Workload queue ownership is unchanged. There are no external
state changes or newly asserted size-limit, package or publication authorities.
