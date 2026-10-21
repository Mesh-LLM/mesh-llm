# mesh-llm-transport

Mesh byte-stream interfaces, Iroh bidirectional streams and TCP/QUIC relay.
This crate owns byte movement and EOF/error propagation. Callers supply the
first-response timeout and retain peer admission, discovery, model routing,
HTTP interpretation and service lifecycle ownership.

Client transport facades and the host tunnel manager consume the same package.
Skippy does not depend on this crate.

The `protocol` module also defines the application-handler contract shared by the
host runtime and embedded SDK: descriptors, authenticated incoming streams,
scoped outbound opens (with a named `OPEN_STREAM_TIMEOUT` dial deadline), peer
snapshots and lifecycle cancellation. `BUILT_IN_PROTOCOL_NAMES` reserves the
names owned by built-in mesh subprotocols; applications may not register them,
and gossip filtering applies the same list. The host owns
admission and dispatch; applications own framing and payload limits. See the
[SDK example](../mesh-llm-sdk/README.md#application-protocols).

Application protocols may supply a `protocol::StageTransportProvider` through
`MeshProtocol::stage_transport`. The SDK owns registration and cancellation;
`ProtocolContext::accept_stage` delegates stage admission and routing to the host.
Providers return owned `ProtocolStream` values and must preserve ordered reliable
bytes, confidentiality, integrity and half-close. See the
[contract](../../docs/specs/APPLICATION_STAGE_TRANSPORT.md).
