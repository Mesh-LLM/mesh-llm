# skippy-scheduler

`skippy-scheduler` provides iteration-level scheduling policy for concurrent
Skippy staged serving. It admits and preempts sequences, budgets prefill and
decode work against runtime memory, and produces shared iteration plans for
`skippy-server` to execute through the native Skippy ABI.

The crate owns scheduling policy only; model execution and transport remain in
the Skippy runtime and server crates.

## License

Licensed under Apache-2.0.
