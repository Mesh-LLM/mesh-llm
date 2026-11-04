# Choose a specific inference provider

MeshLLM normally routes an exact model ID to an eligible provider. To select one
provider explicitly, send the HTTP header `x-mesh-target` with the provider's
**full 64-character hex endpoint ID**, while keeping the exact model ID in the
request body. This is a MeshLLM HTTP extension, not an OpenAI model-name suffix.

## Discover providers and call one

Start your local client with `mesh-llm client --auto`, then query its API:

```bash
BASE_URL=http://127.0.0.1:9337/v1
curl -fsS "$BASE_URL/models" | jq '
  .data[] | {id, requires}'
```

For the desired model, choose an offer's `provider_id`. The offer also reports
`paid`, `pricing`, and `rate_unit`. Discovery is not a binding price quote and
does not prove that a provider is currently healthy. Do not use a shortened node
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

## Gated providers

A provider pin does not bypass a plugin gate: a model that advertises `requires`
channels is only routable when a local plugin serves them, and the serving
node's exchange hook can still refuse the request (402/403). Setting a header
on a shared public HTTP gateway does not authorize that gateway to act for you.

Implementation and regression coverage live in
`mesh-llm-host-runtime/src/network/openai/ingress.rs`, `request_parse.rs`, and
`ingress_tests/tests.rs`. Response propagation lives under `openai/response/`.
