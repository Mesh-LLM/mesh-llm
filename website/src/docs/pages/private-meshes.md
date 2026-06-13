# Private Meshes

A private mesh is a named group of your own machines. Use the same name on each machine you want to connect.

## Start the first serving node

```sh
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

## Add another serving machine

Install Mesh on the second machine, then use the same mesh name:

```sh
mesh-llm serve --discover my-private-mesh --model <model-ref>
```

## Join as an API-only client

Use this for a laptop that should send requests but not serve a model:

```sh
mesh-llm client --discover my-private-mesh
```

## Check that peers are visible

Open the console:

```text
http://localhost:3131
```

Or check status:

```sh
curl -s http://localhost:3131/api/status | jq .
```

Private meshes are useful for lab machines, office workstations, or a home cluster where you want your own machines to find each other by name.
