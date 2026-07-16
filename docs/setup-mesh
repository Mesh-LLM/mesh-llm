# Set up Mesh LLM with an assistant

This is an agent-facing installation and setup runbook. Give this file to an
assistant running on the machine that will be the main Mesh LLM node and ask it
to follow the instructions. The assistant may operate this machine, but Mesh
LLM itself is not being granted remote-execution authority.

The intended result is a private mesh containing this machine and one or more
additional machines, followed by a practical model plan and an optional local
agent-harness configuration.

## Instructions for the assistant

Work interactively. Explain each material choice, ask before installing,
updating, downloading a large model, changing a service, connecting with SSH,
or launching an agent harness. Do not claim success from process startup alone:
observe the mesh, model, and inference paths that the user intends to use.

Treat this as a conditional runbook, not a script that always starts at step 1.
Determine the current state, then enter the relevant section: Mesh absent,
installed but stale, hardware unknown, remote inventory unknown, topology
known, peer missing, model loading, exact inference failing, or harness absent.
After every action, verify its success condition before advancing.

When the user must do something on another machine, first say:

- which machine and shell the command is for;
- what it will install, change, or start;
- why it is the next action; and
- what output or state will mean it worked.

Then provide exactly one complete copy/paste block for that machine. Resolve
all placeholders before showing it. A multiline shell block is acceptable when
it is one paste-and-run operation. Offer to copy it with `pbcopy`, `wl-copy`, or
PowerShell `Set-Clipboard` when available, but only after asking; always display
the same block, and warn that an enrollment block contains an invite token.

Treat the installed CLI as the authority for its version. Run relevant
`--help` commands before composing flags. Use these repository documents for
the concepts, but do not assume that a newer document describes an older
installed binary:

- [Mesh workflows](docs/MESHES.md)
- [Skippy split serving](docs/SKIPPY_SPLITS.md)
- [Agent harnesses](docs/AGENTS.md)
- [Configuration](website/src/docs/pages/config-reference.md)

Do not make a mesh public unless the user explicitly asks. Do not post an
invite token, credentials, host inventory, or private paths to a public service.
An ordinary private invite is connectivity material, not a strong
identity/admission policy. For an untrusted network or controlled membership,
stop and discuss the owner identity and trust-policy options in `docs/MESHES.md`
before enrolling nodes.

### 1. Ask what the user is building

Begin with a short Q&A. Ask only what is not already known, and allow the user
to say “explore this machine” when they do not know an answer.

Establish:

1. The goal: chat, coding agent, several independent models, one model split
   across machines, experimentation, or simply proving that a mesh works.
2. The number of machines and, when known, each OS, architecture, RAM/unified
   memory, GPU, GPU memory, and network location.
3. Whether the machines are on the same fast LAN or separated by the internet.
4. Whether this assistant may use an existing SSH host alias for a remote
   machine. SSH is optional and must be explicitly approved before connecting.
5. Whether the user wants temporary foreground processes first or a persistent
   service after the experiment is proven.
6. Which local agent or chat harness they want to use, if any.
7. Whether stable releases are required or the user intentionally wants a
   prerelease/development build.

Default to one additional node, a private mesh, foreground processes, and
preflighted final models. Use a small starter model only when the additional
machine cannot be inventoried before mesh creation. Do not require an agent on
the other machine.

### 2. Establish the installed release

On this main machine, inspect without changing anything:

```sh
command -v mesh-llm || true
mesh-llm --version 2>/dev/null || true
mesh-llm --help 2>/dev/null || true
```

Determine how the binary was installed when possible. Do not replace a source
or development build with a release binary without asking.

Inspect all resolved installations before adding another one:

```sh
type -a mesh-llm 2>/dev/null || true
```

If a working release binary already exists, update it in place rather than
creating a duplicate. If Mesh is absent, use the official per-user default
unless the user has a preferred executable directory: `~/.local/bin` on
macOS/Linux and `%LOCALAPPDATA%\mesh-llm\bin` on Windows. After installation,
verify the exact resolved path and version. A binary file left from a source
build is not an installed release merely because it exists.

Check the current stable GitHub release separately from the installed version:

```sh
curl -fsSL https://api.github.com/repos/Mesh-LLM/mesh-llm/releases/latest |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])'
```

If Python is unavailable, inspect the API response with another local JSON
tool. In PowerShell use:

```powershell
(Invoke-RestMethod https://api.github.com/repos/Mesh-LLM/mesh-llm/releases/latest).tag_name
```

Compare semantic versions, ignoring a leading `v`. Do not interpret a failed
network check as proof that the installation is current. Once a node is
running, `/api/status` may also expose `version` and `latest_version`; use that
as a second signal.

If the user explicitly wants a prerelease, do not compare it with the stable
`releases/latest` endpoint. Inspect the published releases and select the first
entry where `prerelease` is true and `draft` is false:

```sh
curl -fsSL 'https://api.github.com/repos/Mesh-LLM/mesh-llm/releases?per_page=20' |
  python3 -c 'import json,sys; print(next(r["tag_name"] for r in json.load(sys.stdin) if r["prerelease"] and not r["draft"]))'
```

Use `mesh-llm update --version '<tag>'` only after the user confirms that exact
prerelease.

If Mesh is absent, offer the official installer. The Unix installer supports
Apple Silicon macOS, Linux x86_64, and Linux aarch64:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash
```

Windows x64 PowerShell:

```powershell
irm https://meshllm.cloud/install.ps1 | iex
```

The installer normally runs `mesh-llm setup` when interactive. Afterwards,
open a new shell if necessary and verify `mesh-llm --version` and
`mesh-llm --help`. Do not recommend Homebrew unless its current availability
has been independently verified.

If an installed release is behind, explain the difference and ask before
running:

```sh
mesh-llm update
```

After an install or update, verify the version and required commands again.
If `setup`, `doctor`, or another documented command is missing, treat that as a
binary/version mismatch, not user error. Do not continue with flags the binary
does not advertise. Use the official installer/update path, with approval, to
obtain a release that contains the required surface.

Run machine setup if it has not completed:

```sh
mesh-llm setup
```

For an assistant-controlled, temporary experiment, prefer explicit
non-service setup after approval:

```sh
mesh-llm setup --yes --no-service
```

Do not install a service yet unless the user asks. A service runs
`mesh-llm serve`, which requires startup models in config; proving a foreground
topology first produces much clearer diagnostics.

### 3. Survey this machine before choosing models

Discover capabilities from Mesh rather than estimating from a product name:

```sh
mesh-llm doctor --json
mesh-llm gpus --json
mesh-llm models installed --json
mesh-llm models recommended --json
```

If `gpus --json` reports zero/blank capacity, stale values, or an incomplete
fingerprint on a machine known to have an accelerator, do not plan from it.
Explain that detection refreshes Mesh's cached hardware fingerprint, then run
with approval:

```sh
mesh-llm gpus detect --json
```

Also inspect the live help for the operations likely to be used:

```sh
mesh-llm serve --help
mesh-llm models search --help
mesh-llm models show --help
mesh-llm doctor split --help
```

Record detected backend, usable capacity, installed models, version, OS, and
warnings. On Apple Silicon, unified memory is useful model capacity, but it is
not all safely allocatable to model weights and KV cache. On discrete-GPU
systems, Mesh may report serving capacity differently from raw GPU VRAM. Use
Mesh's detected/advertised capacity and model-fit data; leave headroom for the
OS, context/KV cache, and runtime overhead.

Inventory every proposed serving machine before choosing a model whenever
possible. There are two supported preflight paths:

1. With approved SSH, run install/version checks, `doctor`, `gpus`, disk space,
   and `models installed` on the named host directly.
2. Without SSH, give the user one copy/paste inventory block that installs Mesh
   only if absent, verifies/updates only after consent, and prints a compact
   receipt containing version, hardware, free disk, and installed-model JSON.
   Ask the user to paste that receipt back.

Do not infer a runnable model from a Hugging Face cache directory name. A repo
may contain only metadata, one multipart shard, or a few layer-package files.
Prefer exact entries returned by `mesh-llm models installed --json`, inspect
their reported path and total size, and use `models show` for fit/capabilities.
Prefer complete cached models before proposing any large download.

If remote inventory is unavailable and the initial goal is only to create a
mesh, select a small starter model from the current catalog or local inventory.
Prefer a well-supported model that is a small fraction of local capacity and
quick to download. This is the fallback, not the default. Inspect exact
candidates instead of inventing a model id:

```sh
mesh-llm models search coding --catalog --json
mesh-llm models show '<exact-model-ref>' --json
```

For a coding-agent goal, require advertised tool-use capability where the CLI
provides it. Present the candidate, artifact size, capability evidence, and why
it fits before starting a large download.

### 4. Choose the launch path and start the private mesh

Bare `mesh-llm serve` is not an empty mesh creator in current releases: it
needs at least one configured or explicit startup model. If all machine
inventories are known, choose the final independent/replica/split topology now
and start this machine with its final model. Do not load a disposable starter
first merely to obtain an invite.

If inventories are not available, use the chosen starter as a temporary
bootstrap. Start either model in a supervised foreground terminal and use JSON
events so the assistant can capture exact values:

```sh
mesh-llm serve --model '<selected-model-ref>' --name '<main-node-name>' --log-format json
```

Do not detach a TUI process with `nohup`. Keep it in a terminal or a supervised
PTY that remains observable. `--headless` disables the embedded web UI, not the
terminal UI, and is not a general background-mode flag.

Capture the complete invite token from the structured startup event. Never ask
the user to retype it from a clipped display. Keep the token out of the setup
journal and redact it from summaries. It must appear in the generated enrollment
block because it is what allows the other node to connect.

Confirm on the main node:

```sh
curl -fsS http://127.0.0.1:3131/api/status
curl -fsS http://127.0.0.1:9337/v1/models
```

### 5. Generate one enrollment block per additional node

The other machine does not need an agent and the main machine does not need
direct access to it. Generate one copy/paste block containing install, setup,
and join. Substitute the exact invite and a unique node name. Quote values for
the target shell and never use placeholders in the final block.

For macOS or Linux:

```sh
set -eu
curl -fsSL https://meshllm.cloud/install.sh | bash -s -- --no-setup
MESH="$HOME/.local/bin/mesh-llm"
"$MESH" setup --yes --no-service
"$MESH" doctor --json
exec "$MESH" serve --join '<complete-invite-token>' \
  --model '<exact-cached-model-ref>' --name '<node-name>' --log-format json
```

For Windows x64 PowerShell:

```powershell
& ([scriptblock]::Create((irm https://meshllm.cloud/install.ps1))) -NoSetup
$Mesh = Join-Path $env:LOCALAPPDATA 'mesh-llm\bin\mesh-llm.exe'
& $Mesh setup --yes --no-service
& $Mesh doctor --json
& $Mesh serve --join '<complete-invite-token>' `
  --model '<exact-cached-model-ref>' --name '<node-name>' --log-format json
```

The templates show the preferred preflighted final-model path. Resolve the
exact cached model ref before presenting the block. Omit `--model` only when
inventory was unavailable or the user explicitly wants standby/unallocated
capacity. Neither form gives the mesh permission to run arbitrary commands on
that machine.

Tell the user that the final command remains in the foreground and should keep
running. If a block fails before the node joins, the main node cannot see its
local error. Ask for the terminal output, or offer SSH-based inspection.

If the user approved SSH, first inspect the named target rather than scanning
the network. Explain the exact remote commands, then use SSH to install, survey,
join, and observe logs. Do not copy private SSH keys, alter SSH configuration,
or connect to other discovered hosts without separate approval. For a long-lived
remote serve, use a held TTY and interactive login shell. Quote the entire
login-shell `-c` payload as one argument and verify the remote PID/ports
immediately; if it prints top-level help and exits, treat that as command
construction failure, not a Mesh failure.

### 6. Supervise joining and report state changes

Poll the main management API while the user runs each block:

```sh
curl -fsS http://127.0.0.1:3131/api/status
curl -fsS http://127.0.0.1:9337/v1/models
```

During active setup, check every few seconds and give the user a concise update
whenever a node changes from absent to connected, standby, loading, or serving.
During a long model download/load, reduce polling and report at least once a
minute. Do not spam unchanged JSON.

For each expected node, verify:

- the peer appears and remains connected;
- its hostname/node label, Mesh version, hardware/capacity, and available or
  hosted models are present when enumeration is enabled;
- the mesh id is consistent and publication remains private;
- versions are compatible (prefer the same current release for a first test);
- the expected model eventually appears in `/v1/models`;
- a real chat completion succeeds and returns text.

Use a bounded request for the end-to-end test:

```sh
curl -fsS --max-time 120 http://127.0.0.1:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer mesh' \
  -d '{"model":"<exact-id-from-v1-models>","messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":16}'
```

Do not call a topology healthy merely because `llama_ready` is true or a model
is listed. The completion test catches routing, backend, and split-stage
failures that status alone can miss.

Use a model-appropriate bounded output allowance. Reasoning models may consume
dozens or hundreds of tokens before emitting visible answer text. A response
that reaches the right model but ends at `finish_reason: length` with no useful
text proves transport only, not usable inference. Retry once with a reasonable
bounded allowance and inspect the full assistant message before diagnosing the
route as broken.

### 7. Plan what the machines should serve

Use this section before launch when preflight/SSH has supplied all inventories;
otherwise use it after standby nodes join and advertise theirs. Summarize the
topology in a compact table: node, OS/backend, advertised usable capacity,
network latency when known, free disk, complete installed models, current
state, and version. Then offer no more than three plans, ordered from lowest
risk to most ambitious. When planning before launch, return to section 4 after
the user chooses.

Prefer these shapes:

1. **Independent models (default).** Put one model that fits comfortably on
   each node. This gives the mesh model diversity, avoids stage latency, and
   lets exact model ids or `auto` route requests. It is usually the best first
   useful two-node test.
2. **Replicas.** Run the same independently loadable model on both nodes when
   concurrency or resilience matters more than diversity.
3. **One split model (advanced).** Use a published Skippy layer package only
   when the desired model cannot fit comfortably on one node, the combined
   usable capacity has headroom, and the node-to-node link is fast and stable.

For each candidate, query current evidence:

```sh
mesh-llm models search '<family-or-use-case>' --catalog --json
mesh-llm models show '<exact-model-ref>' --json
```

Do not equate parameter count with resident bytes. Compare the exact artifact
or package size with advertised usable capacity and leave meaningful headroom.
Prefer catalog entries and certified families over arbitrary large GGUFs.

Useful planning examples, not fixed recommendations:

- For a 128 GB Apple Silicon machine plus a 64 GB Apple Silicon machine, first
  consider a strong model that fits the larger node and a smaller, faster or
  differently capable model on the smaller node. Consider a split only when an
  exact layer package is clearly too large for 128 GB but comfortably below the
  mesh's combined usable capacity after overhead.
- For a machine above 256 GB plus a 128 GB Apple Silicon machine on a fast LAN,
  it can be reasonable to evaluate a much larger package-backed split model.
  This remains an advanced plan: inspect the package, certify it, and let the
  split doctor validate actual peer eligibility before downloading hundreds of
  gigabytes.

Do not hard-code a model recommendation from these memory figures. Catalog
contents, quants, package certification, context requirements, and runtime
support change. Show the live search/show evidence and ask the user which plan
to enact.

For independent models, the main node can add a model to its active serving
runtime with the supported local lifecycle command:

```sh
mesh-llm load '<exact-model-ref>'
```

A standby remote node is not a general remote-execution target. To change what
it serves, either use approved SSH or give the user a new exact foreground
command that restarts it with `serve --join ... --model ...`. Do not imply that
the coordinator can install or launch arbitrary models remotely.

For a split, read `docs/SKIPPY_SPLITS.md`, ensure every serving node requests
the same layer-package model with `--split`, and restart the relevant nodes with
the generated exact commands. Then run:

```sh
mesh-llm doctor split --model-ref '<exact-layer-package-ref>' --port 3131 --json
```

Require all intended stages to be ready and a real completion to succeed. If a
worker remains “standing by for stage assignment”, the coordinator reports
ready alone, or inference hangs, treat the split as failed and collect a doctor
bundle rather than waiting indefinitely:

```sh
mesh-llm doctor split --model-ref '<exact-layer-package-ref>' \
  --port 3131 --output-dir '<diagnostic-directory>'
```

### 8. Diagnose failures from the main node

Start with evidence available locally:

```sh
mesh-llm --version
mesh-llm doctor --json
curl -fsS http://127.0.0.1:3131/api/status
curl -fsS http://127.0.0.1:9337/v1/models
```

Classify before changing anything:

- **Install/CLI mismatch:** installed version lacks a documented command or
  flag. Verify the binary path and release; update/reinstall only with approval.
- **Node never appears:** obtain the remote foreground output. Check that the
  complete token was pasted, versions satisfy mesh requirements, and the node
  can reach the internet/relay or direct peer path. Use `--bind-ip` only for an
  identified multi-interface/bridge-address problem. Do not switch to mDNS as
  a generic fix; it may require local mDNS services such as Avahi and still
  requires the invite.
- **Node joins as standby:** this is expected after the enrollment block. It
  advertises capacity but serves no new model until demand/local inventory can
  promote it or it is restarted with an explicit model.
- **Model absent:** distinguish downloading, resolving, loading, insufficient
  capacity, and unsupported backend. Check local inventory and exact model ref.
- **Reported ready but inference crashes or hangs:** inspect the foreground
  logs and backend/runtime selection. A load-only success does not prove that
  the first GPU kernel or distributed stage works.
- **Download/cache failure:** inspect the paths reported by `doctor`, free
  space, permissions, `HF_HOME`, and temporary-directory settings. Do not erase
  caches as a first response.
- **Split failure:** use `doctor split`, stage/runtime status, exact package ref,
  peer latency, and a diagnostic bundle. Fall back to independent models when
  the split cannot be proven healthy.

If the cause is remote-only, offer two choices: the user pastes the relevant
remote output, or the user authorizes SSH to the named machine. With SSH
approval, inspect version, `doctor --json`, process state, and logs. For a
systemd user service use `journalctl --user -u mesh-llm.service`; for a macOS
service, obtain the exact log paths from `mesh-llm setup --verbose` or the
launchd plist. Redact invite tokens and credentials from anything retained or
shared.

### 9. Connect a local chat or agent harness

Only after exact-model inference succeeds, inspect which harnesses exist on the
main machine:

```sh
for cmd in goose claude opencode pi; do command -v "$cmd" || true; done
```

If no suitable harness exists, offer one and explain the external download
before installing it. Goose is a self-contained CLI and its current official
stable installer is documented by the
[AAIF Goose repository](https://github.com/aaif-goose/goose):

```sh
curl -fsSL https://github.com/aaif-goose/goose/releases/download/stable/download_cli.sh | bash
```

Re-check the upstream instruction at install time, ask permission, and verify
`goose --version`; do not silently install a harness as a side effect of Mesh
setup.

Ask which installed harness the user wants. Prefer Mesh's built-in launcher,
which owns the current provider configuration:

```sh
mesh-llm goose --model '<exact-model-id>'
mesh-llm claude --model '<exact-model-id>'
mesh-llm opencode --model '<exact-model-id>'
mesh-llm pi --model '<exact-model-id>'
```

For OpenCode or Pi, `--write` can configure without launching. Check each
launcher's live `--help` before using it. Do not overwrite an existing harness
configuration without explaining the change and obtaining approval.

Choose model ids deliberately:

- start with a proven exact, tool-capable model for coding agents;
- try `auto` after exact routing is known-good and the user wants Mesh to pick;
- use the virtual `mesh` model only when multiple suitable models are healthy
  and the user explicitly wants inter-model collaboration. It is not the
  default health check or a substitute for a working exact route.

Explain how to chat manually as a universal fallback: OpenAI-compatible base
URL `http://127.0.0.1:9337/v1`, any non-empty API key, and an exact id returned
by `/v1/models`.

For a coding harness, validate tool calls rather than stopping at plain chat:

```sh
scripts/qa-agent-tool-call-reliability.py \
  --base-url http://127.0.0.1:9337/v1 \
  --models '<exact-model-id>' --attempts 2 --print-plan
```

If this repository script is not present on the user's machine, perform a
small tool-call test through the chosen harness instead of assuming it exists.

### 10. Keep a resumable, secret-free journal

Maintain `~/.mesh-llm/installation-notes.md` on the main machine, mode `0600`
where supported. This is a concise human-readable working log, not raw terminal
output. Update sections such as Goal, Machines, Installation, Inventory,
Topology, Verification, Harness, Problems, and Next action after meaningful
state changes. Store:

- timestamp and installed Mesh version;
- user goal and approved access method;
- node labels and non-secret hardware/status summaries;
- chosen topology and exact model refs;
- completed checks and their outcomes;
- harness choice and non-secret endpoint/model configuration;
- the next action or diagnosed blocker.

Never store the raw invite token, SSH credentials, API keys, owner keys, or
full unredacted command lines containing secrets. Before resuming, re-check
live process/API state rather than assuming the journal is current.

### 11. Completion report

Finish with:

- installed and latest-known Mesh versions;
- nodes expected, connected, and healthy;
- the chosen topology and why it fits;
- exact models serving and where;
- a successful inference result, including whether it was exact, `auto`, or
  `mesh` routing;
- harness configuration and the command the user should run to chat;
- any remaining warnings, especially version skew, capacity headroom, model
  download size, trust policy, or unproven split behavior.

If any required end-to-end check failed, say the mesh is partially configured,
not complete, and give the smallest next diagnostic action.

## Real-world failure references

These reports inform the checks above and may help when symptoms match:

- [Outdated binary missing setup/config commands](https://github.com/Mesh-LLM/mesh-llm/issues/961)
- [Invite token clipped in terminal output](https://github.com/Mesh-LLM/mesh-llm/issues/964)
- [Split coordinator ready while a worker stage is never activated](https://github.com/Mesh-LLM/mesh-llm/issues/951)
- [Model download resolving to a read-only path](https://github.com/Mesh-LLM/mesh-llm/issues/980)
- [First inference crashes despite successful model load on an unsupported ROCm target](https://github.com/Mesh-LLM/mesh-llm/issues/966)
- [Detailed newcomer feedback on install, mDNS, model choice, and split ergonomics](https://github.com/Mesh-LLM/mesh-llm/discussions/978)
