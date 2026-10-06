The native generic Jobs interface is `model-package-generic-jobs`. Build it with `just generic-conversion-jobs-release-build`. This is an isolated model-package helper; xtask remains pure and uses the supplied image's pinned runner.

Prepare `delivery.json` with schema_version1, namespace, worker_input, mounts and cpu_plan. `cpu_plan` is the existing CpuJobPlan serialized from the reviewed CPU hardware planner; its cost and source/image provisioning are declarations, not observed hosted proof. `worker_input` is the closed generic-conversion envelope: schema_version1, workflow generic-conversion, timeout_secs259200 for the original72h job, pinned image-supplied runner, operator{schema_version1,bootstrap,conversion}, receipt_export. The bootstrap supplies selected mesh/git-tree/llama revisions, recipe pin, exact eight tools, CPU static profile, image digest, reviewed PATH and matching plan/cost. The job image must already supply the pinned Linux xtask, Git/Just/compiler tools, private publisher helper/source and writable /work parent. Nothing constructs an image or installs arbitrary toolchains.

`mounts` contains readonly model repo+immutable commit+container path tuples. Conversion source files pin the full mounted checkpoint. For upload-only, set conversion.upload_only=true and supply upload_artifact with immutable converted repo/revision/source_directory, fresh work_directory, target_prefix, output_basename, requested_splits, complete flat name/SHA/byte_size roster and matching timeout_seconds. Conversion source equals mounted artifact directory; original BF16 source_repo remains provenance. Manifest/card/status bytes are copied unchanged into work/target/prefix; no build or conversion occurs. Effective emitted count may exceed requested count. Whole-folder publication retains128shard/32sidecar admission before provisioning and one immutable verified commit.

The evidence repo must already exist; receipt_export binds its current parent commit and unique path. The image-supplied publisher helper/source is pinned. Publish-confirmed model output additionally requires the existing full model/repository helper contracts. Explicit private credential file is the only facade token source. Its Unix owner must match current user and mode must exclude group/other bits. The facade never prints the token or worker_input.

Read-only preparation performs no remote requests:

```sh
target/release/model-package-generic-jobs prepare --input /absolute/delivery.json --output-directory /absolute/fresh-prepared
```

Only an explicitly authorized submission uses confirmation. It returns after submission acknowledgment, corresponding to original detached submission; SUBMITTED never means converted or certified:

```sh
target/release/model-package-generic-jobs submit --confirm-submission --input /absolute/delivery.json --credential-file /absolute/private-token --output-directory /absolute/fresh-submitted --timeout-seconds 300
```

Archive original delivery.json and submitted.json together. Collection rebinds the full declaration/request/job and uses existing bounded Jobs monitoring plus immutable receipt bytes, with10second polls and28800 maximum polls (80h capacity):

```sh
target/release/model-package-generic-jobs collect --input /absolute/delivery.json --submitted-file /absolute/fresh-submitted/submitted.json --credential-file /absolute/private-token --output-directory /absolute/fresh-collected --timeout-seconds 259200
```

Failed native Jobs can produce correlated immutable observations with conversion_admitted=false. Public bytes, locator hashes and native success are required for admission; CERTIFIED is rejected. `mesh-llm models package --status JOB_ID`, `--logs JOB_ID` and `--cancel JOB_ID` remain the existing inspect/log/cancel surfaces. Local signal/timeout stops monitoring or an in-flight POST; it sends no automatic remote cancel and does not prove remote acceptance absent. On unconfirmed submission, inspect authorized Jobs status before deciding whether to retry. Receipt publication and actual cloud job/model/image/tool qualification remain distinct.
