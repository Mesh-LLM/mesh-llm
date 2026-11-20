# skippy-model-resolver

`skippy-model-resolver` turns a user-supplied model name, path, or catalog
reference into ranked artifact candidates. It checks local GGUF files and layer
packages, searches configured model directories, and matches catalog variants.
The returned candidates distinguish local and remote GGUFs from local and
remote layer packages; a local candidate wins over a remote one.

The crate owns candidate selection and the catalog-provider interface. It does
not download model bytes, inspect a remote repository's file set, validate a
package manifest, or load a model. Those are separate responsibilities of
[`skippy-model-hf`](../skippy-model-hf/README.md),
[`skippy-model-artifact`](../skippy-model-artifact/README.md),
[`skippy-package-format`](../skippy-package-format/README.md), and the runtime.

Mesh's host runtime and standalone Skippy commands are consumers. Callers provide a catalog (an
in-memory catalog or an `entries/` checkout through `HfCatalogProvider`) and
the local search directories; network refresh remains the caller's job.

The `startup` module owns startup source values and the pure explicit-source
and missing-source policy. This is separate from `ModelResolver` ranking:
startup checks the catalog before the installed cache for bare names. The
ordered asynchronous lookup lives in `skippy-model-hf::startup`.
