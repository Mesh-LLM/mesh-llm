# skippy-model-package

`skippy-model-package` supports the Hugging Face job workflow that prepares
and publishes Skippy layer-package repositories. It discovers source GGUF
quantizations and projectors, checks publishing permissions, prepares job
specifications, submits and inspects Hugging Face Jobs, and carries the
versioned split-job script. The Mesh model commands are its current consumer.

This is job orchestration, not the package-v2 manifest contract or the local
package writer. [`skippy-package-format`](../skippy-package-format/README.md)
owns validated package manifests and identities;
[`skippy-package-builder`](../skippy-package-builder/README.md) builds a local
source-complete package. Runtime loading and materialization belong to the
Skippy runtime and lifecycle API.

The job workflow currently targets the MeshLLM Hugging Face organization and
catalog where applicable. Its `queue-unsloth-layer-packages` binary is a
specialized batch tool, not the general package-builder command.
See the [model onboarding guide](../../docs/NEW_MODEL_ONBOARDING.md) for a
dry-run example. The binary defaults to dry-run; `--confirm` submits jobs and
can incur Hugging Face Jobs charges.
