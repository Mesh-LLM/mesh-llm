# skippy-commands

Standalone Skippy command execution and output formatting. Argument parsing lives in `skippy-cli`; this crate executes the parsed commands against the shared Skippy API and renders their JSON output through a single console facility.

`models` resolves Hub references, downloads verified artifacts with size/SHA-256 verification and manages the local model cache. `runtime` lists, installs from explicit catalogs, imports and migrates native runtime caches. `split` plans and admits direct GGUF splits against the same release-bound certification roster Mesh uses, publishing stage configs and admission descriptors only after every stage is admitted. `console` installs the standalone diagnostics sink and writes JSON documents.

Commands are expressed as plain typed actions (`ModelAction`, `RuntimeAction`, `PlanSplitCommand`), deliberately decoupled from Clap. The crate has no dependency on `skippy-server`, adds no serving options types, and reads only the documented `SKIPPY_*` environment variables through `skippy-config` path policy.
