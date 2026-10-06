# Existing optional Python research and upstream references

These sixteen existing paths are optional research, task fixtures or upstream
compatibility references. They do not provide generic required validation.
`BOUNDARIES.json` records each exact path, dependency imports, purpose and named
caller. `pyproject.toml` declares the separate interpreter and empty external
package dependency set for fifteen stdlib paths. Use an already installed
CPython 3.11 through 3.14 for those paths; they need no package synchronization.
The quantizer comparison is separate: `evals/quantizer-reference/pyproject.toml`
declares its actual local numpy and gguf imports on CPython 3.12. Its optional
upstream-conversion group records the prepared pinned converter requirements.
No lock resolution, installed environment or converter execution is claimed. This
source declaration does not establish execution on every interpreter version.

Preserve the documented working directory for the MoA scripts because local
imports and relative corpus/output paths are part of their existing behavior.
For example, from `evals/moa-openrouter`, `python3 make_fixture.py` converts
existing captured JSONL files into the native test fixture. Recording commands
require an explicit OpenRouter key and spend money; immutable Rust trace replay
requires neither Python nor remote capture. Raw `tool_calls`, `finish_reason`,
usage and latency remain in the captured reference data.

The latency README and injection script describe historical upstream RPC/hook
experiments with external `rpc-server` or `llama-server`. They are not current
MeshLLM build, serving or release instructions. Retaining them preserves that
experiment; it does not authorize new machines, builds, downloads or runtime
modernization. Virtual LLM ablation similarly targets an already-running
research endpoint. Its old startup example is historical.

The three scenario files are agent task inputs. Their intentional bugs and
refactoring opportunities remain. They are not generic repository automation.
The pinned LMCache scalar codec generator writes compatibility fixture bytes;
normal native tests consume checked bytes and do not regenerate them with
Python. It is an upstream codec reference, not a migration differential oracle.
The quantizer comparison preserves supplied reference-tool conversion and
all-mode output comparisons. Its explicit --python/--python-converter child must use the matching optional
upstream-conversion environment. Different upstream revisions need their own
dependency/provenance check. This project does not install that environment.

The four actual third-party SDK clients retain their required cadence and
separate locked environments. SDK execution remains conditional and unqualified
until actual endpoint/model/platform evidence exists. Reader and Granite
exceptions keep their own existing declarations. This document does not admit
additional Python into required PR/main setup. Exception admission and actual
research execution qualification are separate from this isolation proposal.
