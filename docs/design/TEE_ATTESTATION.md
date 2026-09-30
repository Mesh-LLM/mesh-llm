# Client-verifiable TEE inference

Status: design and first manual Intel TDX verifier. Mesh does not yet offer a
`TEE only` client setting or automatic attestation-aware routing.

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

## Production Mesh contract

The following work is needed before a client can select `TEE only` without a
manual verifier:

1. **Challenge service and evidence.** Add an optional, bounded attestation
   request/response over an authenticated iroh peer connection. The request
   carries a protocol version, 32-byte random nonce, and intended model or
   serving-state epoch. The response carries raw evidence, platform type,
   measured workload/model claims, and the peer ID. Generate the endpoint key
   in the guest. The attested code must derive the peer ID from its own key,
   not copy a value supplied by the caller. Use a versioned, domain-separated
   `report_data` digest over nonce, peer ID, and serving-state claim. Bound
   evidence size and verification time. Unknown/old peers may omit this
   optional protocol; they must be ineligible for TEE-only requests.
2. **Client-local appraisal.** Define a vendor-neutral verified-peer result
   issued only after a platform verifier checks signed evidence and a local
   allowlist. Intel TDX is the first verifier. Keep a short proof lifetime;
   invalidate on key, model epoch, TCB, or policy changes. Discovery may carry
   a cheap hint, but cannot issue a verified-peer result.
3. **Fail-closed routing.** Add an explicit `require_tee` request/client policy.
   Filter model/capacity candidates against fresh verified-peer results
   *before* any prompt or tool content leaves the client. Connect to the
   verified iroh identity and check the serving identity on the response.
   No verified candidate, quote failure, expiry, or target substitution is an
   error. Initially allow only single-host inference; reject MoA, split, and
   plugin paths until every participant that can see prompt-derived data is
   verified under the same policy. A TEE-only mesh admission rule can reuse
   the verifier, but does not replace client-local appraisal when the client
   distrusts the mesh operator.
4. **Additional hardware.** Keep the challenge/routing contract and add an AMD
   SEV-SNP or other CVM verifier with its own vendor roots and measurement
   policy. For confidential GPU inference, appraise *both* the CVM and GPU
   evidence, require GPU confidential-computing mode, and establish that the
   GPU serving this session is the one appraised. An RTX PRO 6000 name alone
   is not a claim. NVIDIA documents GPU evidence and local verification in
   its [attestation SDK][nvidia-gpu]. Current
   [dstack security guidance][dstack-security] says its Hopper/Blackwell
   deployments cannot rule out a live relay to another genuine GPU without a
   CPU-TEE-verifiable device
   binding; that assurance must be treated as a platform capability, not
   assumed from two passing quotes.

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
