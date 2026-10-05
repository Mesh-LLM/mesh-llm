# mesh-llm-commands

`mesh-llm-commands` owns command handlers that can run without depending on
`mesh-llm-host-runtime`.

This crate is part of the host-runtime decomposition: command handlers move
here first when they can be expressed in terms of lower-level domain crates.
The shipped `mesh-llm` binary can dispatch these handlers directly, while
`mesh-llm-host-runtime` keeps temporary compatibility shims until command
dispatch fully leaves the host runtime.

## Wallet CLI

For the current external `lexe-wallet` installation and the normal balance,
funding, policy and send commands, see the
[wallet operator guide](../../docs/plugins/wallet/SETUP.md). It also explains
why every wallet command should use the node's explicit `--config` path and how
to preserve existing state before the first wallet open.
