# Client-verifiable TEE inference

Status: draft Intel TDX single-host implementation. The Mesh client has an
opt-in fail-closed route, but this has not yet been certified by a fresh
hardware run. Other TEE families and confidential GPU routing remain plans,
not supported capabilities.

## What a client must establish

A client may discover peers through an untrusted mesh. Before sending a prompt,
it must obtain a fresh hardware quote for the selected peer, appraise it against
policy chosen by the client, and connect to the *same* authenticated iroh
`EndpointId`. A self-advertised `tee` bit, an operator badge, encrypted QUIC,
and successful inference are not evidence of TEE execution.

The proof has four separate links:

1. **Hardware:** verify the quote signature and certificate chain against the
   hardware vendor root, current signed collateral, expiry, TCB status, and
   security-relevant configuration such as debug mode. Collateral can come from
   a cache or Intel PCS; the verifier checks its signatures. Intel documents
   these checks in its [TDX enabling guide][intel-tdx]. Intel requires cached
   collateral for production use of PCS; the direct PCS call below is for
   low-frequency trials.
2. **Workload:** compare quoted launch/runtime measurements with a trusted
   allowlist for the guest OS, measured loader, Mesh deployment, and its
   configuration. For dstack, replay the [runtime event log][dstack-tdx] to
   the quoted RTMR3 and compare the measured application Compose hash with an
   independently approved deployment. A quote for an arbitrary VM does not
   establish that Mesh runs inside it.
3. **Peer and freshness:** the client chooses a fresh nonce. Quote `report_data`
   binds that nonce and the serving iroh public identity. The Mesh endpoint
   private key must be generated and retained inside the measured guest, by
   code covered by the workload policy. The authenticated QUIC connection then
   proves possession of that key. Merely letting an attestor quote a caller
   supplied public key would allow a real TEE to endorse an outside server.
4. **Inference path:** the approved runtime must keep prompt processing and
   any prompt-derived data inside approved participants. The quote does not
   prove this from network behavior alone. Model weights loaded after boot need
   a measured or trusted loader that checks their digest. The client must
   reverify on key, workload, model epoch, or policy change.

Thus the truthful initial claim is *this endpoint key belongs to an approved,
currently attested TEE workload*. Claiming a particular model produced the
answer requires the model-load and forwarding controls above.

## Binary signing and trust policy

Signing a downloadable Mesh binary is useful for provenance and updates, but
is not a prerequisite for hardware attestation. A client can approve an exact
measured workload digest without a separate release signature. For a scalable
release process, a trusted signer can certify which digest corresponds to
reviewed Mesh source; the client still has to match that digest to the quoted
running workload, including the native runtime libraries it loads. Mesh's
existing `require_release_attestation` policy is build provenance and does
not make this runtime link. See [Mesh workflows](../MESHES.md#immutable-mesh-requirements).

The party selecting the approved measurements or signer is part of the trust
model. Local quote verification avoids trusting a Mesh operator's assertion,
but still trusts the hardware vendor, verifier implementation, measured code,
and the client's policy. A configured third-party verifier can instead issue a
short-lived signed result bound to the peer; that is a distinct trust mode and
must be shown as such.

## Existing Mesh foothold

The route table currently carries model and peer identities without runtime
TEE evidence (`mesh/node.rs`, `mesh/peer_state.rs`). `x-mesh-target` can pin a
single-host request to one peer and refuses fallback if it is unavailable
(`network/openai/ingress.rs`). That was used in a manual Intel TDX trial. A
`model=mesh` MoA committee and split inference may involve more than the
pinned host, so pinning alone cannot secure those paths.

[`scripts/verify-mesh-tdx-peer.py`](../../scripts/verify-mesh-tdx-peer.py)
turns the trial's manual quote check into a repeatable, fail-closed command.
It expects a dstack-style `/attest?nonce=...` response. The user supplies an
independently approved policy with exact `mr_td`, `rt_mr0` through `rt_mr3`,
and `app_compose_sha256` values. The tool gets Intel-signed collateral from
Intel PCS, verifies the quote locally with [dcap-qvl][dcap-qvl], rejects
non-current TCB, advisories and debug mode, and checks the nonce, selected
peer, measurements, Compose hash, and RTMR3 event-log replay. It does not trust a Phala verifier
verdict. It does not verify model bytes or enable automatic routing.

Example, after obtaining trusted measurement values from the approved build
and deployment process:

```json
{
  "mr_td": "<96 hex characters>",
  "rt_mr0": "<96 hex characters>",
  "rt_mr1": "<96 hex characters>",
  "rt_mr2": "<96 hex characters>",
  "rt_mr3": "<96 hex characters>",
  "app_compose_sha256": "<64 hex characters>"
}
```

```bash
python3 -m venv /tmp/mesh-tee-verifier
/tmp/mesh-tee-verifier/bin/pip install 'dcap-qvl==0.6.5'
/tmp/mesh-tee-verifier/bin/python scripts/verify-mesh-tdx-peer.py \
  --policy approved-tdx-policy.json \
  --peer PEER_ENDPOINT_ID \
  --url https://attestor.example:8080
```

On success from a **live** challenge, use the returned `endpoint_id` as
`x-mesh-target` for one ordinary, single-host request with an exact model ID,
and check `x-mesh-served-by` matches. A request must stop if the selected peer
changes. Saved evidence can be inspected with `--evidence`
and the original `--nonce`, but that does not create a fresh proof for a new
request. The verifier needs an accurate local clock to appraise collateral
expiry. The Phala trial used a development guest OS, and its model bytes were
not independently tied to the quote; its old measurements are not a production
allowlist.

## Draft in-Mesh Intel TDX path

On an approved dstack TDX guest, set `MESH_LLM_EPHEMERAL_KEY=1` before starting
Mesh and point `MESH_TEE_DSTACK_SOCKET` at the guest-only dstack API socket.
The TEE stream handler takes a client nonce, derives report data from its own
iroh `EndpointId`, obtains raw `/GetQuote` and `/Info` data, and returns a
bounded evidence frame. A node without that configuration cannot answer.
This only creates a useful endpoint binding if the independently approved
workload measurement covers the running Mesh code, its native runtime, key
handling, and socket access policy. Merely setting these environment variables
on an arbitrary host proves nothing.

On the client, set `MESH_TEE_TDX_POLICY` to a JSON file in the format above.
Add `x-mesh-require-tee: true` to a local `/v1/chat/completions` request with
an exact model ID; `MESH_TEE_REQUIRE_ALL=1` makes this the default for local
inference ingress. The client obtains a **new** quote on an authenticated iroh
connection to each candidate, checks the Intel signature/collateral and
policy itself, and selects only a passing peer. It sends no prompt to a
candidate that fails. The selected peer receives a local-serving-only marker
and rejects re-routing, MoA, plugins, and an unavailable local model. The
marker is a route restriction, **not** a hardware proof by itself. Non-TEE
requests retain existing behavior. The temporary bootstrap tunnel cannot
perform TEE appraisal and returns 503 for TEE-required or local-only requests
until the full router is ready. This first route verifies **remote** peers; a
local serving node is not automatically self-certified merely because its
client ingress has TEE mode enabled.

This first path deliberately rejects `model=auto`, `model=mesh`, split work,
and other OpenAI endpoints. It has no verified model-byte digest or
serving-state epoch, no proof cache, no offline collateral cache, and no
confidential GPU proof. It proves only the selected endpoint and approved
measured workload. The trial was deleted; archived evidence is useful for
regression checking, not for claiming a live machine is currently attested.

## Platform adapter boundary

The challenge stream and strict single-host route stay shared. The client
chooses a platform verifier through its own policy, then appraises a bounded
evidence envelope before prompt dispatch. The existing Intel TDX/dstack
response retains `format: "mesh-tee-endpoint-v1"` and its original flat fields.
The new envelope shape is:

```json
{
  "format": "mesh-tee-endpoint-v2",
  "platform": "snp-direct",
  "endpoint_id": "<32 raw iroh public-key bytes, hex encoded>",
  "nonce": "<32 client-chosen bytes, hex encoded>",
  "payload": { "<platform-specific signed evidence and collateral>": "..." }
}
```

`platform` names a verifier candidate; the peer cannot choose the client's
trusted root or policy. The parser accepts the v2 envelope only as input to a
matching registered verifier. With only the TDX v1 verifier configured, every
v2 platform is rejected. Unknown formats, unexpected v2 fields, and mismatched
nonce or authenticated `EndpointId` also fail closed. The outer endpoint and
nonce fields are untrusted echoes; the selected verifier must find their
binding in signed hardware/provider evidence.

All new adapters use the same 64-byte endpoint-binding value:

```text
B_v2 = SHA-512("mesh-tee-endpoint-v2\0" || endpoint_id[32 raw bytes] || client_nonce[32 raw bytes])
```

The fixed-length byte strings are concatenated in that order, without hex
text or length prefixes. For a guest-controlled direct SNP or TDX report, the
entire value goes into its signed 64-byte `REPORT_DATA`/`REPORTDATA`. AWS Nitro
can carry it in signed `user_data` while also signing the original challenge
in `nonce`. Managed VM/vTPM deployments require a profile-specific signed
chain from this binding to the hardware report and measured workload; field
placement and trust roots must be verified for each provider. The current TDX
v1 binding uses `SHA-256` over its v1 domain, nonce, and endpoint, followed by
32 zero bytes. It is valid only for v1 and cannot be used as a v2 fallback.
Neither binding makes a model-byte claim.

Cross-adapter vector: endpoint bytes `00 01 ... 1f`, nonce bytes
`20 21 ... 3f`, and expected `B_v2` hex
`25931c725b7ac216a56a0f2a7b6a2c71706fbb3fb57ddcc90923532158295435a32937fd60e564ef9f94306e9d1ba962dd8fd65adf62a642ffc7ef04074f1b6a`.
See the [AMD SNP ABI](https://docs.amd.com/v/u/en-US/56860_PUB_SEV_SNP),
[Intel TDX ABI](https://cdrdv2-public.intel.com/853289/intel-tdx-module-abi-spec-348551006.pdf),
and [AWS Nitro document specification](https://docs.aws.amazon.com/enclaves/latest/user/verify-root.html)
for the signed fields. The digest construction and envelope are Mesh protocol
choices.

## Platform coverage plan

Keep one Mesh challenge/routing contract but make the evidence producer and
verifier explicit per platform. A TEE-only request must name an allowed
platform policy. Unknown evidence formats, downgraded claims, a cloud-only
software assertion when hardware proof is required, or any missing workload
binding fail closed. Model bytes and every additional worker need their own
measured or attested chain before a stronger claim is offered.

| Family | Hardware evidence and workload policy to add | Mesh milestone |
|---|---|---|
| Intel TDX CVM | DCAP quote, Intel root/collateral/TCB, debug flag, MRTD/RTMR event replay and exact workload; bind nonce and iroh key in report data. | Draft single-host path here; fresh TDX end-to-end certification next. |
| AMD SEV-SNP CVM | [SNP report][amd-snp] and AMD ARK/ASK plus VCEK/VLEK chain, TCB/security policy and launch measurement; attest the later boot/userspace chain too. [Google notes][google-roots] its vTPM measurements for these later stages are a separate, provider-controlled trust boundary. | Next CPU-TEE adapter and negative corpus. |
| AWS Nitro Enclave | Verify [AWS-signed COSE attestation document][nitro], PCRs, nonce and public key; package the inference server and networking proxy so the key/prompt stay in the enclave. | Separate deployment and verifier adapter; no GPU claim. |
| Cloud-managed SEV/vTPM | Verify provider-signed vTPM evidence and measured boot under an explicitly **provider-trusting** policy. Do not label this equivalent to vendor-root SNP/TDX evidence. | Separate trust-mode label and verifier. |
| Arm CCA Realm | Verify [platform/Realm token][arm-cca] chain, challenge, RIM/REM measurements and workload/endpoint binding against trusted reference values. | Adapter when suitable Mesh hosting is available. |
| Intel SGX enclave | Verify enclave quote, TCB/debug state, MRENCLAVE/MRSIGNER policy and endpoint binding; all code and prompt-bearing I/O must stay within its smaller enclave boundary. | Optional distinct runtime, not an interchangeable CVM flag. |
| NVIDIA confidential GPU with a CVM | Verify **both** CPU CVM and [NVIDIA GPU device evidence][nvidia-gpu], supported confidential-computing mode and firmware, and the protected CPU–GPU session/device binding; attest every GPU/switch in multi-GPU mode. RTX PRO 6000 Blackwell Server Edition is a single-GPU CC option in [NVIDIA's current matrix][nvidia-platforms], but a GPU model name is never evidence. [Google G4][google-g4] uses AMD SEV and Google-managed vTPM evidence for its VM, so that deployment has a different CPU trust root from TDX/SNP. | CPU verifier first, then GPU verifier and binding test; fail closed meanwhile. |

For NVIDIA, [local verification and Ready state][nvidia-cpp-sdk] are separate
steps. Two independently valid quotes alone do not prove the verified GPU
served this request. [dstack security guidance][dstack-security] explicitly
describes a device-relay limitation for its Hopper/Blackwell deployments;
Mesh must not claim a CPU–GPU binding until the deployment can establish it.
This is an extensible platform plan, not a promise that every vendor or product
marketed as a TEE has the evidence and deployment controls Mesh requires.

## Remaining production Mesh contract

The draft path above implements the first challenge, TDX appraisal and
single-host route. The following work remains before it is production-ready
and before other platform families can be selected:

1. **Certify the first path on live hardware.** Rebuild an independently
   approved, digest-pinned TDX workload; verify a fresh challenge, then make
   real single-host inference to that same identity. Confirm the serving
   process and QUIC key stay inside the measured guest. Record a sanitized
   quote, policy, server identity, response identity, and failure cases.
2. **Bind model state.** Have the measured loader verify exact model bytes and
   an epoch/digest. Extend the quote's domain-separated report-data binding
   and client policy with this claim, and invalidate any proof on load, unload,
   update, key change, or policy change. Today the route proves the workload,
   not the specific model weights.
3. **Harden verification and operations.** Cache signed collateral without
   trusting the cache; appraise expiry and revocation, add proof-rate limits,
   concurrency limits and structured failure reasons. A vendor-neutral
   verified-peer result can then support several verifier adapters. Discovery
   remains an untrusted candidate source.
4. **Extend execution topology.** Add platform adapters from the table above.
   Permit MoA/split only after every prompt-bearing worker and link is
   independently checked. For confidential GPU inference, prove the
   CPU–GPU protected-session binding as well as each device quote. An
   attestation-required mesh admission rule is useful but does not replace
   client-local checks against a dishonest mesh operator.

Negative tests must cover a forged discovery hint, wrong nonce, swapped peer
key, altered quote, stale or advisory TCB, debug mode, wrong workload/model,
expired proof, route fallback, and unsupported multi-host paths. The final
integration test is a fresh quote followed by inference to that exact peer;
mixed-version meshes must continue working for ordinary requests.

[intel-tdx]: https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/02/infrastructure_setup/
[dcap-qvl]: https://github.com/Phala-Network/dcap-qvl/blob/master/README.md
[dstack-tdx]: https://github.com/Dstack-TEE/dstack/blob/next/docs/attestation-tdx.md
[dstack-security]: https://github.com/Dstack-TEE/dstack/blob/next/docs/security/security-model.md
[nvidia-gpu]: https://docs.nvidia.com/attestation/attestation-client-tools-sdk/latest/gpu_and_switch_attestation.html
[nvidia-cpp-sdk]: https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/overview.html
[amd-snp]: https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58217_amd-epyc-9004-ug-platform-attestation-using-virtee-snp.pdf
[google-roots]: https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/attestation-overview
[google-g4]: https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/create-a-confidential-vm-instance-with-gpu
[nitro]: https://docs.aws.amazon.com/enclaves/latest/user/verify-root.html
[arm-cca]: https://learn.arm.com/learning-paths/servers-and-cloud-computing/cca-veraison/attestation-token/
[nvidia-platforms]: https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/latest/supported-platforms.html
