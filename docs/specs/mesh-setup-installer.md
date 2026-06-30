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

`install.sh` and `install.ps1` own only first-stage executable installation:

- parse minimal bootstrap flags
- detect the host platform and architecture
- download the matching release archive
- verify checksum sidecars when available
- install the `mesh-llm` executable into the selected directory
- update or print PATH guidance
- run or print `mesh-llm setup`

They must not own native-runtime flavor selection, GPU/backend detection,
runtime install or prune, service setup, doctor/readiness interpretation, or
GitHub account mutations.

Legacy script service/runtime flags may emit compatibility warnings or forward
direct setup flags, but they must not reintroduce shell or PowerShell-owned
runtime or service policy.

## Setup Command

`mesh-llm setup` owns machine setup after the executable exists. CLI parsing
lives in `crates/mesh-llm-cli`, binary dispatch in `crates/mesh-llm`, and setup
orchestration in `crates/mesh-llm-commands`.

Supported flags:

- `--yes`: accept recommended core setup defaults without prompting.
- `--no-interactive`: never prompt; use non-interactive defaults.
- `--service`: request background service setup where supported.
- `--no-service`: skip background service setup.
- `--skip-runtime`: skip native-runtime install and prune.

Unsupported in v1:

- `--skip-doctor`
- setup-owned doctor/readiness flow
- Windows service creation
- startup model, public/private mesh, or telemetry prompts
- a separate installer binary

## Runtime Setup

Unless `--skip-runtime` is passed, setup installs the recommended or configured
native runtime and then prunes inactive runtime artifacts.

Runtime selection must reuse `mesh-llm-runtime-install` and the existing
`runtime_native` resolver/config-selection plumbing. Setup and scripts must not
duplicate native runtime candidate scoring or hardware detection.

Runtime install failure is a core setup failure. Runtime prune failure after a
successful install is reported as a warning because the installed runtime is
usable.

## Service Setup

Service setup is supported for:

- Linux systemd user units
- macOS launchd agents

Windows `mesh-llm setup --service` is a hard unsupported-service error. Default
Windows setup and `--no-service` continue without service setup.

Interactive Unix setup prompts for service installation and defaults to Yes.
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
- `gh auth status --active --hostname github.com` succeeds
- the authenticated viewer has not already starred the repo
- a visible interactive prompt is shown

The prompt defaults to Yes:

```text
Star Mesh-LLM/mesh-llm on GitHub using your authenticated GitHub CLI account? [Y/n]
```

`--yes`, `--no-interactive`, hidden prompts, missing `gh`, unauthenticated
`gh`, eligibility-check failures, already-starred state, and star-request
failures must not fail core setup. No hidden account mutation is allowed:
starring only happens after the visible prompt is accepted.

## Final Summary

Setup output must separate:

- runtime installed, skipped, or installed-with-prune-warning
- service installed, skipped, guidance-only, or failed
- GitHub star completed, skipped, already present, or non-fatally failed

`mesh-llm doctor` remains a separate troubleshooting command and is not part of
normal setup.
