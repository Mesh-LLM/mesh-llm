# Public Mesh

Use the public mesh to discover and connect to publicly advertised nodes through Nostr. Publishing discovery information to Nostr makes it public, with or without a mesh name. A name helps people find a mesh; it does not restrict who can join.

Join as a serving node:

```sh
mesh-llm serve --auto
```

Join as an API-only client:

```sh
mesh-llm client --auto
```

Open the console:

```text
http://localhost:3131
```

Point OpenAI-compatible tools at:

```text
http://localhost:9337/v1
```

For predictable first-run behavior on your own hardware, use the [Quickstart](/docs/pages/quickstart/) single-node setup instead.
