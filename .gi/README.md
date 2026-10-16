# Jianyang templates

The template file is migrated from `general-inference/jianyang` so code and
job definitions can be reviewed together. Existing unrelated entries remain
unchanged. `agentx` replaces the old `agentic-replay` entry.

Validate without installing:

```sh
jy repos validate mesh-llm .gi/templates.yaml --json
```

After merging, update the repository registration to read its own
`.gi/templates.yaml` (`templates_repo: null`), then `jy repos sync mesh-llm`.
Do not sync a partial inventory: sync replaces all installed entries.

For a pre-merge run, use Jianyang's `renderRepoJob` with this validated template
and the immutable PR SHA, then submit that rendered specification with `jy
submit`. This uses the same renderer as repository launches and leaves the live
registration unchanged. See [AgentX operations](../skippy/docs/AGENTX.md).
