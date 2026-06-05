# Public Mesh

Use the public mesh when you want to try discovery without naming your own private mesh.

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

For predictable first-run behavior on your own hardware, use the [Quickstart](/docs/pages/quickstart/) private mesh flow instead.
