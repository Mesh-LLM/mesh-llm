# skippy-analytics

Anonymous, opt-out product analytics for Skippy. This is the only crate in
the workspace that reports anything to a vendor, which keeps the full
reported surface reviewable in one sitting. The user-facing description is
`docs-site/content/configuration/analytics.mdx` and must stay in step with
the `Event` enum.

## Guarantees

- **Closed vocabulary.** `Event` and `Properties` admit no free-form
  `String`. Text enters only through `Label::sanitize`, whose grammar rejects
  paths, prose, and anything over-long.
- **No key, no reporting.** The release pipeline compiles the project key in
  through `SKIPPY_POSTHOG_KEY`. Source and development builds are inert unless
  that variable is set at run time.
- **Never load-bearing.** Capture is non-blocking and delivery is best effort.
  No failure here changes a command's behavior, output, or exit code.
- **Anonymous by construction.** The identifier is a random UUID derived from
  nothing about the machine. Every event sets `$geoip_disable` and a null `$ip`.

## Two modes

- **Standalone** (`init`): the `skippy` binary resolves consent
  (`SKIPPY_ANALYTICS`, then `DO_NOT_TRACK`, then the preference
  `skippy analytics disable` records, then CI detection), shows the first-run
  notice, and keeps its own install identifier.
- **Embedded** (`init_embedded`): a host such as Mesh LLM has already resolved
  consent and disclosed under its own rules. It passes a `Host` and its own
  install identifier; nothing is written to Skippy's state directory.

## Layout

| File | Responsibility |
|---|---|
| `consent.rs` | Resolving whether this machine reports, and why |
| `preference.rs` | The opt-out `skippy analytics` records |
| `event.rs` | The closed event and property vocabulary, and the label grammar |
| `generation.rs` | The `generation` event, from `skippy_serving::generation_stats` |
| `models.rs` | What events may say about a model: catalog name or `redacted` |
| `host_profile.rs` | The `host_profile` event and the machine class `generation` carries |
| `install_id.rs` | The anonymous install identifier and state directory |
| `version_state.rs` | The last version this install ran, so upgrades are reportable |
| `properties.rs` | Base properties attached to every event |
| `client.rs` | PostHog batch payload construction and delivery |
| `notice.rs` | The first-run disclosure |

## Verifying what a build sends

```bash
SKIPPY_POSTHOG_HOST=http://127.0.0.1:8000 \
SKIPPY_POSTHOG_KEY=phc_test \
SKIPPY_ANALYTICS=1 \
  skippy doctor
```
