# Skippy Stage Protocol

The Skippy stage protocol is an internal, generation-gated contract. Do not preserve backward compatibility when changing it.

- Remove superseded request and response arms, messages, domain types, converters, validators, routing, state, tests, fixtures, and documentation completely.
- Do not add compatibility adapters, dual-read or dual-write paths, fallback decoding, aliases, or deprecated shims for older stage generations.
- Bump `STAGE_PROTOCOL_GENERATION` and its advertised `stage-generation-N` capability for every breaking wire change.
- Do not reserve removed protobuf fields. Renumber the surviving messages and fields into the clean current-generation wire layout; old payloads are unsupported.
- Do not retain old protocol tag numbers, protocol missing-field defaults, unknown-field decoder tests, or fixtures that prove an older generation still decodes.
- Keep mixed-generation peers fail closed through the generation capability gate.
