# mesh-llm-transport

Mesh byte-stream interfaces, Iroh bidirectional streams and TCP/QUIC relay.
This crate owns byte movement and EOF/error propagation. Callers supply the
first-response timeout and retain peer admission, discovery, model routing,
HTTP interpretation and service lifecycle ownership.

Client transport facades and the host tunnel manager consume the same package.
Skippy does not depend on this crate.
