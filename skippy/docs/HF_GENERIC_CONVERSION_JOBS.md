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

## Prepare the Linux helpers once

Build these commands in the Linux image's selected mesh checkout. A macOS build cannot supply Linux executables. These recipes build tools; they do not construct a container image, acquire a model or submit a Job.

```sh
just automation-bootstrap
just generic-conversion-jobs-release-build
just model-publisher-release-build
just layer-job-helper-release-build
just mtp-checkpoint-helper-build
```

| Recipe | Output | Image destination used below |
| --- | --- | --- |
| automation-bootstrap | target/debug/xtask, printed as binary_path | /opt/mesh/xtask |
| generic-conversion-jobs-release-build | target/release/model-package-generic-jobs | /opt/mesh/model-package-generic-jobs |
| model-publisher-release-build | target/release/model-package-publish | /opt/mesh/model-package-publish |
| layer-job-helper-release-build | target/release/model-package-layer-job | /opt/mesh/model-package-layer-job |
| mtp-checkpoint-helper-build | target/debug/model-package-mtp-checkpoint | /opt/mesh/model-package-mtp-checkpoint |

The staging helper is needed for default MTP composition. Copy the selected source records beside the helpers too. Publisher/helper source pins are regular source artifacts, not a claim that a single source file proves executable reproducibility. Record the commit and build provenance separately.

During image preparation, copy regular files to the final destinations, retain execute permission and hash the final copied bytes. For example, inside the Linux image preparation environment:

```sh
install -d /opt/mesh
install -m 0755 target/debug/xtask /opt/mesh/xtask
install -m 0755 target/release/model-package-generic-jobs /opt/mesh/model-package-generic-jobs
install -m 0755 target/release/model-package-publish /opt/mesh/model-package-publish
install -m 0755 target/release/model-package-layer-job /opt/mesh/model-package-layer-job
install -m 0755 target/debug/model-package-mtp-checkpoint /opt/mesh/model-package-mtp-checkpoint
sha256sum /opt/mesh/xtask /opt/mesh/model-package-generic-jobs \
  /opt/mesh/model-package-publish /opt/mesh/model-package-layer-job \
  /opt/mesh/model-package-mtp-checkpoint
/opt/mesh/model-package-generic-jobs --help
/opt/mesh/model-package-publish --help
/opt/mesh/model-package-layer-job --help
```

The image must already contain the exact eight bootstrap tools: git, just, cargo, rustc, cmake, c++, ld.lld and curl. Give each an absolute regular executable path and SHA-256. The declared PATH directories must select those same tools by name. Probe each with `--version`. Pin the completed image by registry digest. A tool probe and image digest establish declarations and local availability; they do not certify the cloud runtime or native ABI.

Normal conversion and default MTP composition still build a fresh CPU converter in the Job. The existing bootstrap clones https://github.com/Mesh-LLM/mesh-llm.git, checks out the declared mesh commit, checks its tree and upstream pin, runs `just llama-prepare`, then `just skippy-quantize-standalone-release-build cpu`. It observes the resulting target/release/skippy-quantize before and after use. Supplying a prebuilt converter does not bypass that branch. Upload-only uses its separately admitted immutable artifact roster and omits the build.

Prepare an existing writable /work parent and readonly immutable model mounts under /models, for example /models/checkpoint or /models/target. Keep helpers outside /work and /models. Both bootstrap and child evidence directories are fresh. No setup command here installs a compiler, chooses an image or grants publication authority.

## Assemble and validate one request

Keep the model-specific operator request separate from the common delivery envelope. `operator.json` is the existing generic operator object with schema_version, bootstrap and conversion. For default MTP it is the compose-default operator object with bootstrap, immutable checkpoint/tokenizer file maps, an explicitly source-bound tokenizer profile, ordered target parts, helper pins and publication settings. Set its credential_file to null for Jobs. `receipt-export.json` supplies the existing evidence repository, its immutable parent commit, unique path ending /native-job.json, publisher helper/source pins, credential_environment:true and credential_file:null. Reserve its export_budget_secs within the whole budget.

Use `model-package-generic-jobs plan` as described below to produce cpu-plan.json and bootstrap-resources.json from a supplied hardware snapshot. Apply the returned resource fields to the operator before preparing delivery; do not reproduce the typed receipt digest by hashing jq or pretty JSON.

With those three source-bound documents and `mounts.json`, the outer configuration is small. This command only projects JSON. It does not establish model pins, costs, credentials or image custody:

```sh
jq -n --arg namespace "$HF_NAMESPACE" --arg runner_sha "$LINUX_XTASK_SHA256" \
  --slurpfile op operator.json --slurpfile plan cpu-plan.json \
  --slurpfile mounts mounts.json --slurpfile export receipt-export.json '
  {schema_version:1,namespace:$namespace,cpu_plan:$plan[0],mounts:$mounts[0],
   worker_input:{schema_version:1,workflow:"generic-conversion",
     timeout_secs:$plan[0].timeout_seconds,
     runner:{path:"/opt/mesh/xtask",sha256:$runner_sha},
     operator:$op[0],receipt_export:$export[0]}}' > delivery.json
```

A mount row has only repo, revision and mount_path, for example `{"repo":"owner/checkpoint","revision":"<actual 40-hex commit>","mount_path":"/models/checkpoint"}`. Generic conversion source_files must cover the supplied checkpoint files within its mount. Default MTP mounts cover the complete ordered target roster; its checkpoint staging helper separately acquires the pinned checkpoint and tokenizer file maps. Never guess a tokenizer profile from architecture.

For default MTP Jobs, use workflow `default-mtp-composition` with its compose-default operator. It uses the same build outputs, envelope and prepare/submit/collect grammar.

Set cpu_plan.timeout_seconds, worker_input.timeout_secs, operator/bootstrap timeout and conversion timeout or MTP overall_seconds to the same admitted budget. The original 72-hour budget is 259200 seconds. Default MTP Jobs also admits up to 259200 seconds. Keep all source/helper/image revisions immutable. The CPU planner's estimate and bootstrap maximum cost must correlate; pricing is declared evidence until actual hosted observation.

Read-only facade preparation makes no remote call:

```sh
/opt/mesh/model-package-generic-jobs prepare \
  --input "$PWD/delivery.json" --output-directory "$PWD/fresh-prepared"
jq '{status,submitted,conversion_admitted,image_observed,cost_observed}' \
  fresh-prepared/result.json
```

Expect PREPARED, submitted:false, conversion_admitted:false and image_observed:false. Archive declaration.json, result.json and the original delivery.json. Prepare checks the delivery envelope, mounts, plan/budget correlation and pin grammar. It does not execute the full worker's filesystem admission, verify mounted bytes or prove the image contains its declared tools. Those checks run in the owning worker/bootstrap. Do not change the request between prepare, submit and collect.

Submission and collection commands below consume an explicit private credential file and fresh output directories. On Unix its owner must match the current user and its mode must exclude group/other access. Submit requires authorized cloud spending and publication; a SUBMITTED acknowledgment is not conversion completion. The inline worker request is limited to 64 KiB. Use the mounted request transport below for larger complete inventories. Existing image construction and full model/native/cloud acceptance remain operator requirements.

## Generate the resource plan offline

Build the existing facade with `just generic-conversion-jobs-release-build`. Supply an explicit hardware snapshot from the approved Jobs hardware declaration. The planner makes no network requests. For a finite example, this declared row is a fixture price, not a current HF quote:

```json
{
  "schema_version": 1,
  "hardware": [{"name":"cpu-upgrade","cpu":"8 vCPU","ram":"32 GB","unitCostUSD":0.01,"unit_label":"minute"}],
  "requested_flavor": "auto",
  "requested_timeout_seconds": 259200,
  "model_size_bytes": 1024,
  "max_cost_usd": 50.0
}
```

Replace the snapshot and model_size_bytes with the supplied complete source size. `auto` preserves the existing CPU baseline preference. A named flavor selects that explicit CPU row. The existing planner can raise a short requested timeout to its model-size minimum; use the returned timeout throughout the worker/operator/bootstrap request. The declared maximum must cover the computed estimate.

```sh
target/release/model-package-generic-jobs plan \
  --input "$PWD/hardware-request.json" --output-directory "$PWD/fresh-resource-plan"
jq '{status,submitted,hardware_observed,cost_observed}' fresh-resource-plan/result.json
```

The output is cpu-plan.json, bootstrap-resources.json and result.json. The resources document supplies timeout_seconds, cpu_plan_receipt_sha256, declared_estimate_usd and max_cost_usd. Its hash comes from serializing the typed CpuJobPlan with the existing Rust owner; hashing the pretty file gives a different digest.

For the generic operator, apply those returned fields before composing delivery.json:

```sh
jq --slurpfile resource fresh-resource-plan/bootstrap-resources.json '
  .bootstrap += $resource[0] |
  .conversion.timeout_seconds = $resource[0].timeout_seconds
' operator.json > planned-operator.json
```

For default MTP, use `.bootstrap += $resource[0] | .overall_seconds = $resource[0].timeout_seconds` instead. Keep publication/export reserves within that returned whole budget. Use fresh-resource-plan/cpu-plan.json and planned-operator.json in the envelope example. The plan does not construct the immutable model roster, source/profile pins or tool/image declarations.

`plan` refuses credentials, confirmation, submitted acknowledgments, invalid hardware/size/timeout declarations and insufficient maximum cost before creating output. `PLANNED_OFFLINE` means the supplied snapshot was planned. It never means submitted, current pricing observed or model conversion admitted. Actual source, image and cloud acceptance remain separately qualified.

## Export larger requests without a provider-secret limit

The facade accepts worker_input JSON inline up to 64 KiB. For a larger complete source inventory, `export-request` creates the exact canonical worker bytes and derives the mount locator before calling the existing mounted workflow admission. The worker byte limit is 8 MiB; the outer facade input limit is 16 MiB. Generic conversion and default MTP composition use the same transport. Keep the complete roster rather than trimming pins to fit inline transport.

Create the export input from your delivery.json. Add a separate readonly request-repository mount under /models/requests and choose worker-input.json within it. Use an existing immutable repository commit for the initial offline declaration:

```sh
jq --arg repo "$REQUEST_REPO" --arg revision "$REQUEST_PARENT_COMMIT" '
  .mounts += [{repo:$repo,revision:$revision,mount_path:"/models/requests"}] |
  {schema_version:1,delivery:.,request_destination:{
    path:"/models/requests/worker-input.json",repo:$repo,revision:$revision}}
' delivery.json > request-export.json
target/release/model-package-generic-jobs export-request \
  --input "$PWD/request-export.json" --output-directory "$PWD/fresh-export"
```

The output contains private regular worker-input.json, mounted-request.json, delivery.json and result.json. The result binds the original export request, final delivery declaration and exact worker bytes by SHA-256 and byte size. `REQUEST_EXPORTED_OFFLINE` does not mean those bytes are remotely present. No token, upload or Jobs submission is consumed by export.

Publish worker-input.json unchanged to the selected request repository through your authorized repository publication operation. After obtaining the actual immutable commit that contains the bytes, update both request_destination.revision and the matching /models/requests mount revision in request-export.json, then run export-request into another fresh directory. The canonical worker bytes stay the same because the transport locator/mount envelope is outside worker_input. Rebind the actual commit before submission; an existing parent declaration alone is not custody of the new file. Verify the new worker SHA matches the exact published bytes. Request-repository provisioning/publication remains a separate operation; this facade doesn't upload its own request.

Use the final exported delivery.json directly:

```sh
target/release/model-package-generic-jobs prepare \
  --input "$PWD/final-export/delivery.json" --output-directory "$PWD/fresh-prepared"
```

Prepare exports worker-input.json for small inline requests too, alongside declaration.json and result.json. `worker_input_sha256` and `worker_input_byte_size` identify those exact private output bytes. Do not hash the pretty outer delivery.json as the worker locator. The worker reads the readonly mounted regular file, verifies the exact byte count/hash and only then admits its typed request. The provider receives the small locator, not the large worker document. The full original delivery input and submission acknowledgment still belong together for collection correlation.
