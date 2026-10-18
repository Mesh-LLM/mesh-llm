# Mesh Setup Installer

This spec defines the v1 install and setup boundary for Mesh LLM.

## User Flow

The public install flow is two-stage:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash
mesh-llm setup
```

On Windows:

```powershell
irm https://meshllm.cloud/install.ps1 | iex
mesh-llm.exe setup
```

The bootstrap scripts may run setup automatically when attached to an
interactive terminal. In non-interactive contexts they print the exact setup
command instead.

## Bootstrap Scripts

`install.sh` and `install.ps1` own only first-stage product installation:

- parse minimal bootstrap flags
- detect the host platform and architecture
- download the matching release archive
- verify checksum sidecars when available
- warn and continue when a `.sha256` checksum sidecar is missing by default;
  fail closed only when `MESH_LLM_REQUIRE_CHECKSUM=1` is set
- require the product-v2 manifest and selected `native-runtimes/` tree
- install the `mesh-llm` executable, product manifests, and adjacent runtime
  tree into the selected product directory
- update or print PATH guidance
- run or print `mesh-llm setup`

They select the published archive flavor for the detected platform, but must
not reimplement runtime candidate scoring, copy backend libraries beside the
host, install the bundled runtime into the user cache, prune cached runtimes,
own service setup, interpret doctor/readiness, or mutate GitHub accounts.

Legacy script service/runtime flags may emit compatibility warnings or forward
direct setup flags, but they must not reintroduce shell or PowerShell-owned
runtime or service policy.

## Setup Command

`mesh-llm setup` owns machine setup after the executable exists. CLI parsing
lives in `mesh/crates/mesh-llm-cli`, binary dispatch in `mesh/crates/mesh-llm`, and setup
orchestration in `mesh/crates/mesh-llm-commands`.

Supported flags:

- `--yes`: accept recommended core setup defaults without prompting.
- `--no-interactive`: never prompt; use non-interactive defaults.
- `--service`: request background service setup where supported.
- `--no-service`: skip background service setup.
- `--skip-runtime`: skip native-runtime install and prune.
- `--verbose`: print detailed setup paths, commands, log locations, and
  status. Default setup output is concise and formatted for interactive use.

Unsupported in v1:

- `--skip-doctor`
- setup-owned doctor/readiness flow
- a machine-level Windows service (the Windows service is per-user)
- startup model, public/private mesh, or telemetry prompts
- a separate installer binary

## Runtime Setup

Unless `--skip-runtime` is passed, setup installs the recommended or configured
native runtime and then prunes inactive runtime artifacts.

Runtime selection must reuse `skippy-runtime-install` and the existing
`runtime_native` resolver/config-selection plumbing. Setup and scripts must not
duplicate native runtime candidate scoring or hardware detection.

Runtime install failure is a core setup failure. Runtime prune failure after a
successful install is reported as a warning because the installed runtime is
usable.

## Service Setup

Service setup is supported for:

- Linux systemd user units
- macOS launchd agents
- Windows per-user scheduled tasks

On Windows, setup registers a scheduled task named `mesh-llm` for the current
user: a logon trigger, an interactive token, least privilege, and no elevation.
The task runs `conhost.exe --headless` from the binary's directory with the
hidden `mesh-llm service-host` subcommand, so no console window shows, at task
priority 4, which gives the node the CPU, I/O and memory priority of a `serve`
started from a terminal (the default, 7, would run it below normal). The
service host runs `mesh-llm serve` with its output copied into
`%LOCALAPPDATA%\mesh-llm\logs\service.log`, restarts it after a failure,
including a start that fails, with an exponential backoff (5 seconds, doubling,
five restarts in a row, counted afresh after ten minutes of running), and stops
it when the task ends. The log moves to `service.previous.log` at each start
and once it would pass 10 MiB, so the two stay under 20 MiB. Ending a
task only ends its own process, so the service host watches that console host
and holds the node in a kill-on-close job. The node reads its TOML config and
`invite.token` beside it; Windows has no `service.env`. When the user has no
desktop session, as when setup runs over SSH, the task cannot start until the
next logon; setup reads the task's last result and reports that, instead of
reporting a start that did not happen.

Interactive setup prompts for service installation and defaults to Yes.
`--yes` accepts service setup unless `--no-service` is also present.
Non-interactive setup prints service opt-in guidance unless `--service` is
explicitly supplied.

When service setup is requested or accepted, service installation, enablement,
and startup failures are core setup failures. Setup must not silently escalate
privileges; it writes per-user service files and surfaces permission or command
failures.

## GitHub Star Prompt

After successful core setup, an interactive setup may offer to star
`Mesh-LLM/mesh-llm` using the already-authenticated GitHub CLI account.

Eligibility:

- `gh` is on PATH
- `gh api --hostname github.com /user --silent` succeeds using the account selected by `gh`
- the authenticated viewer has not already starred the repo
- a visible interactive prompt is shown

The prompt defaults to Yes:

```text
Star Mesh-LLM/mesh-llm on GitHub? [Y/n]
```

`--yes`, `--no-interactive`, hidden prompts, missing `gh`, unauthenticated
`gh`, eligibility-check failures, already-starred state, and star-request
failures must not fail core setup. No hidden account mutation is allowed:
starring only happens after the visible prompt is accepted.

## Final Summary

Default setup output must provide a concise formatted completion summary. The
summary must separate:

- runtime installed, skipped, or installed-with-prune-warning
- service installed, skipped, guidance-only, or failed
- GitHub star completed, already present, or non-fatally failed when those
  outcomes occur

Detailed paths, generated service commands, log commands, and follow-up
diagnostic status, including exact GitHub skip reasons, belong behind
`--verbose`.

`mesh-llm doctor` remains a separate troubleshooting command and is not part of
normal setup.

## Uninstall Command

`mesh-llm uninstall` owns automated cleanup after the executable has been
installed. It is intentionally implemented in Rust, not in the bootstrap
scripts, so it can reuse the same platform path decisions as setup and remove
the running binary last.

Supported flags:

- `--yes`: run without confirmation.
- `--dry-run`: print the planned cleanup without changing the machine.
- `--json`: print dry-run plans and outcomes as JSON.
- `--verbose`: print detailed cleanup steps and removed paths. Default text
  output is concise and formatted for interactive use.
- `--keep-cache`: preserve downloaded native runtimes.
- `--keep-service-files`: preserve setup-owned service helper files.
- `--purge-config`: remove `~/.mesh-llm` configuration and identity data.
- `--keep-config`: explicitly preserve configuration and identity data.
  `--purge-config` and `--keep-config` are mutually exclusive; CLI parsing must
  reject invocations that provide both flags.
- `--binary-path <PATH>`: remove a specific executable path.

Default uninstall behavior stops tracked `mesh-llm` processes, disables and
removes the per-user Linux systemd unit or macOS launchd agent when present,
removes setup-owned service helper files, removes the native-runtime cache, and
then removes the executable. On Windows it first ends and deletes the per-user
scheduled task, so its service host does not restart the node being stopped,
then removes the task definition and the service logs. It preserves `~/.mesh-llm` by default so keys,
identity, and user configuration are not deleted accidentally.

The command removes only known setup-owned service files. If the setup service
configuration directory contains unrelated files, uninstall leaves that
directory in place and reports a warning instead of recursively deleting it.
