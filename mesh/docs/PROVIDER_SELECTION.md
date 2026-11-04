# Choose a specific inference provider

MeshLLM normally routes an exact model ID to an eligible provider. To select one
provider explicitly, send the HTTP header `x-mesh-target` with the provider's
**full 64-character hex endpoint ID**, while keeping the exact model ID in the
request body. This is a MeshLLM HTTP extension, not an OpenAI model-name suffix.

## Discover providers and call one

Start your local client with `mesh-llm client --auto`, then query its API:

```bash
BASE_URL=http://127.0.0.1:9337/v1
curl -fsS http://127.0.0.1:9337/mesh/v1/models | jq '
  .data[] | {id, providers: [.providers[] | {provider_id, available, requires}]}'
```

For the desired model, choose a provider's `provider_id`. A gated provider
lists under `requires` the plugin channels it needs, with the terms its plugin
attached (a price, for a payment plugin), republished verbatim; see
[Mesh model discovery](MODEL_DISCOVERY.md). Discovery is not a binding price
quote and does not prove that a provider is currently healthy. Do not use a shortened node
ID displayed by a status table, an owner ID, or a wallet ID.

```bash
MODEL='exact-model-id-from-models'
PROVIDER='full-64-character-provider-id-from-the-selected-offer'

curl -sS -D response-headers.txt "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "x-mesh-target: $PROVIDER" \
  --data "$(jq -nc --arg model "$MODEL" '{
    model: $model,
    messages: [{role: "user", content: "Reply with a short greeting."}],
    max_tokens: 64,
    stream: false
  }')"

grep -i '^x-mesh-served-by:' response-headers.txt
```

The routing response's `x-mesh-served-by` identifies the resolved provider. Check
the HTTP status and completion body as well: receiving this header on an error
is not proof of successful inference or payment settlement.

## SDKs and command-line clients

Use a client that lets you add request headers. For example, the OpenAI Python
SDK accepts `extra_headers` on a chat-completion request:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9337/v1", api_key="mesh")
response = client.chat.completions.with_raw_response.create(
    model=model_id,
    messages=[{"role": "user", "content": "Reply with a short greeting."}],
    max_tokens=64,
    extra_headers={"x-mesh-target": provider_id},
)
print(response.headers.get("x-mesh-served-by"))
print(response.parse().choices[0].message.content)
```

Here `model_id` and `provider_id` come from discovery above. `api_key="mesh"` is
an SDK placeholder, not payment authorization. A separately authenticated HTTP
frontend may require its own credentials.

The `mesh-llm client --auto` CLI starts the routing API; it does not itself select
an inference provider. There is no general `--provider <endpoint-id>` inference
flag in the MeshLLM CLI. Harness provider names (such as `mesh` in an agent's
configuration) select a configured API backend, not a mesh peer. If a harness
cannot attach this HTTP header, selecting only a model name does not pin a peer.

## Fail-closed behavior and exclusions

- Omitting `x-mesh-target` retains ordinary routing and eligible-provider retries.
- A target is a hard constraint, not a preference. A provider that is absent,
  blocked, excluded, or not serving the model is rejected; MeshLLM does not
  silently fall back to a different provider.
- Malformed IDs or multiple target headers return HTTP 400. Conflicting or
  unavailable selections return HTTP 409 at selection time; later execution or
  transport failures may return other errors.
- Use a concrete model ID. Virtual models (including plugin-backed automatic or
  mixture routing), pipeline dispatch, and model-less requests do not support
  this provider constraint and reject it rather than silently ignoring it.
- `x-mesh-exclude` accepts comma-separated full endpoint IDs to exclude peers
  while leaving ordinary routing among the remaining candidates. Excluding the
  explicit target is an error.

These headers constrain local-user ingress routing. They never authorize an
incoming peer request to hop to another peer.

## Require verifiable receipts

Inference receipts are experimental. Read
[what a pass proves](design/RECEIPTS.md#what-a-pass-proves) and the
[open limitations](design/RECEIPTS.md#open-limitations) before you rely on
them.

Send `x-mesh-trust: receipts` to serve a chat completion only on a runtime
that issues [inference receipts](../../skippy/docs/RECEIPTS.md): this node's
own, or a peer that advertises receipts for the model. Your node then audits
about one turn in ten from those peers, also when `[receipts] audit` is off.
It blocks a peer for 7 days when an audit fails in something that the peer
signed. It also checks the receipt of every turn. A peer that advertises
receipts and returns no usable receipt gets a strike, and three strikes in
24 hours block the peer for 7 days. The other results of an audit are in
[Audit policy in Mesh](design/RECEIPTS.md#audit-policy-in-mesh).

Your node audits a turn only when it can tie the model name of the request
to a Hugging Face file. Name the model as `owner/repo:selector`, as
`owner/repo@revision/path/file.gguf`, or by a name in the loaded catalog of
your node. A bare runtime name of a peer, an alias, `local-gguf/sha256-...`
and `owner/repo` without a selector get no audit.

```bash
curl -sS "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'x-mesh-trust: receipts' \
  --data "$(jq -nc --arg model "$MODEL" '{
    model: $model,
    messages: [{role: "user", content: "Reply with a short greeting."}]
  }')"
```

When no candidate serves the model with receipts, the request fails with
HTTP 409 and the error code `trust_policy_unsatisfied`:

```json
{"error":{"message":"no candidate serves model 'Qwen3-4B-Q8_0' with receipts, which the request's trust policy requires","type":"invalid_request_error","code":"trust_policy_unsatisfied"}}
```

- Receipts cover `/v1/chat/completions` only. Other inference endpoints fail
  with the same code under this policy.
- `x-mesh-trust: any` is ordinary routing. An unknown tier, or an empty entry,
  returns HTTP 400. So do the tiers `trusted`, `measured` and `tee`, which
  are not built.
- `x-mesh-target` and `x-mesh-exclude` still apply. A target that does not
  serve the model with receipts is rejected with HTTP 409.
- Virtual models, pipeline dispatch, plugin endpoints, and model-less
  requests reject the policy with HTTP 409.
- A model that is still loading on your node answers HTTP 409 to a request
  under the policy.
- A split with a stage on another node does not serve the model with
  receipts, on your node or on a peer.
- An operator can make `receipts` the node default with
  `[receipts] require = true`. A request then cannot ask for `any`: that
  returns HTTP 400.
- With `[receipts] offline = true`, your node does no audit, and no peer
  satisfies the policy. Only your node's own runtime can serve the request.

The rules of the tier are in
[TRUST_TIERS.md](design/TRUST_TIERS.md#receipts).

## Gated providers

A provider pin does not bypass a plugin gate: a model that advertises `requires`
channels is only routable when a local plugin serves them, and the serving
node's exchange hook can still refuse the request (402/403). Setting a header
on a shared public HTTP gateway does not authorize that gateway to act for you.

Implementation and regression coverage live in
`mesh-llm-host-runtime/src/network/openai/ingress.rs`, `request_parse.rs`, and
`ingress_tests/tests.rs`. Response propagation lives under `openai/response/`.
