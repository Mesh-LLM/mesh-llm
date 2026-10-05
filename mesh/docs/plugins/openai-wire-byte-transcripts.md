# OpenAI wire byte commitments

Lifecycle `wire_bytes` commitments use SHA-256 over the HTTP entity bytes at
the named observation point. They exclude the HTTP status line, headers,
transfer chunk lengths, transfer chunk delimiters, and trailers. They include
JSON whitespace and field order. An SSE response includes every encoded SSE
field, line delimiter, event delimiter, and terminal event, including `[DONE]`
where that endpoint emits it.

The original request commitment covers the received entity before JSON parsing
or core transformations. The effective request commitment covers the serialized
entity supplied to the selected backend. A typed backend call must explicitly
identify its serialization observation point because it has no HTTP request
bytes of its own. It must not describe a canonical JSON hash as the received
request byte hash.

The final response commitment covers bytes offered at final HTTP emission in
the typed frontend and bytes accepted by the downstream writer in raw TCP/QUIC
ingress. It does not prove that the peer received or persisted those bytes.
Events distinguish these boundaries with `emission_boundary: "http_body_poll"`
or `emission_boundary: "socket_write_accept"`. The typed commitment becomes
complete when its last HTTP body frame is handed to the transport, even if the
transport subsequently drops that frame. It therefore cannot establish a
socket delivery prefix. A gateway's raw final client tap independently records
the bytes its downstream writer accepted. Neither boundary proves remote
execution or client receipt.
HTTP transport chunk boundaries do not change the commitment. SSE frame bytes
do. A compression or response-rewrite layer must precede the final tap if its
output is the promised observation point.

`incomplete` is absent only after the entire entity reaches its natural end.
Cancellation, transport error, timeout, and invalid framing preserve the hash
and length of the observed prefix and identify why it ended early. A successful
execution outcome must not imply complete evidence. `side_stream_complete`
becomes false if the bounded, ordered observer queue overflows or disconnects,
even when the host's constant-space SHA-256 computation reaches the end. The
observer cannot edit response bytes, and queue pressure never stalls streaming.

## Ordered observer delivery

Exact original, effective, and response entities travel on negotiated,
bidirectional local side streams with `receipt_protocol: "sha256-v1"`.
The host half-closes the payload writer at EOF. The plugin hashes the received
payload, then returns one JSON line containing `sha256` and `byte_count`.
The host bounds that receipt to 512 bytes, verifies both fields, and waits for
the receipt before sending the corresponding lifecycle event. A write success
alone does not establish complete observer delivery.

Each plugin has independent delivery evidence. A disconnected observer does
not stop delivery of later frames to a healthy observer. Terminal commitments
report the recipient's own receipt status. Revoking a grant cancels its active
callbacks and side streams, including queued bytes.

`max_in_flight` limits persistent response subscriptions as well as callback
concurrency. Each response subscription has an exact `max_queue_bytes` budget,
including an active write, so total queued response payload per plugin is
bounded by their product. Request delivery, admission callbacks, and response
stream setup share one absolute operator deadline. Required delivery or
capacity failures prevent backend dispatch; best-effort failures mark evidence
unavailable while inference continues.

Response annotations are namespaced by their author plugin. Lifecycle delivery
includes only the recipient's own annotation namespace, preventing a plugin
with body access from exposing prompt contents through annotations to other
observers. The metadata grant authorizes additions; it does not hide core
lifecycle descriptors or grant another plugin access to those additions.

A live grant reduction preserves the distinction between removal and an
unsatisfied required declaration. Removing the grant or the requested endpoint
deactivates that subscription. Keeping an endpoint and phase subscribed while
removing required body or metadata permission prevents backend dispatch and
reports `permissions_unavailable`; the host sends that plugin neither lifecycle
callbacks nor entity side streams. Restoring the required permissions permits
later exchanges without reinstalling the plugin.

## Published test vectors

These UTF-8 vectors use LF `\n`, with no BOM. The Rust tests use independently
calculated SHA-256 values and exercise all fragment sizes. The shell commands
below can reproduce them without a MeshLLM host or a plugin.

The request bytes are exactly 35 bytes, including the final LF.

```text
{ "model": "tiny", "stream":true }
```

```sh
printf '{ "model": "tiny", "stream":true }\n' | shasum -a 256
```

SHA-256 is `61b54b4afd933a3dad0f1854e1ee723d92e10e0289a84d14079fefe34b99921e`.

The SSE transcript is exactly the following bytes, including two LF bytes after
each data line. The last blank line is part of the entity.

```text
data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}

data: [DONE]

```

```sh
printf 'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n' | shasum -a 256
```

SHA-256 is `2774745745fc204699d02f6f2d032c205fd17caf18070316bcfbbb16956f66fd`.
Hashing just `hi`, canonical JSON chunks, or reconstructed assistant output does
not produce this commitment.

The raw ingress framing test separately encodes `data: hi\n\n` using two HTTP
transfer chunks with an extension and a trailer. It proves that transfer
framing is excluded at every possible input fragmentation size. Fixed-length
tests exclude bytes read past the declared entity length and sensitive headers.
Malformed or truncated transfer framing produces explicit incomplete evidence.
