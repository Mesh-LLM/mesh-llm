# mesh-llm-commands

`mesh-llm-commands` owns command handlers that can run without depending on
`mesh-llm-host-runtime`.

This crate is part of the host-runtime decomposition: command handlers move
here first when they can be expressed in terms of lower-level domain crates.
The shipped `mesh-llm` binary can dispatch these handlers directly, while
`mesh-llm-host-runtime` keeps temporary compatibility shims until command
dispatch fully leaves the host runtime.
