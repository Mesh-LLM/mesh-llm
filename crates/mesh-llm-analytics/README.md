# mesh-llm-analytics

Anonymous, opt-out product analytics for mesh-llm. This is the only crate in
the workspace that reports anything to a vendor, which is the point: keeping
it in one place makes the full reported surface reviewable in one sitting.

Not to be confused with the `[telemetry]` config section, which exports OTLP
metrics to an endpoint the operator chooses and never leaves their network.

## Guarantees

- **Closed vocabulary.** `Event` and `Properties` admit no free-form `String`.
  Text enters only through `Label::sanitize`, whose grammar rejects paths,
  prose, and anything over-long. A prompt cannot reach the wire through this
  API even by mistake, and an unrecognized value becomes `redacted`.
- **No key, no reporting.** The project key is compiled in by the release
  pipeline via `MESH_LLM_POSTHOG_KEY`. Source and development builds have no
  key and are completely inert.
- **Never load-bearing.** Capture is non-blocking and delivery is best effort.
  No failure here changes a command's behavior, output, or exit code, and
  shutdown gives up after a short budget rather than holding a command open.
- **Anonymous by construction.** The identifier is a random UUID derived from
  nothing about the machine, and deliberately not the published mesh identity.
- **No location.** Every event sets `$geoip_disable` and a null `$ip`.

## Opt-out precedence

`MESH_LLM_ANALYTICS` wins over everything (including the CI check, so the path
can still be tested deliberately), then `DO_NOT_TRACK`, then
`[analytics] enabled` in the config, then CI detection. A missing project key
disables reporting regardless of all of them.

## Layout

| File | Responsibility |
|---|---|
| `consent.rs` | Resolving whether this machine reports, and why |
| `event.rs` | The closed event and property vocabulary, and the label grammar |
| `install_id.rs` | The anonymous install identifier |
| `properties.rs` | Base properties attached to every event |
| `client.rs` | PostHog batch payload construction and delivery |
| `notice.rs` | The first-run disclosure |

## Verifying what a build sends

```bash
MESH_LLM_POSTHOG_HOST=http://127.0.0.1:8000 \
MESH_LLM_POSTHOG_KEY=phc_test \
MESH_LLM_ANALYTICS=1 \
  mesh-llm gpus
```

User-facing documentation lives in `website/src/docs/pages/analytics.md` and
must stay in step with the `Event` enum.
