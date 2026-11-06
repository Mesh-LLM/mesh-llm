# Following a Llama Upgrade repair

Open the repair build's **Build exact candidate for distributed certification**
step in GitHub Actions. The console projects Goose's structured events into short
updates; the repair prompt asks for an action and its reason, then a result and
next step. Tool commands are abbreviated and successful tool-response bodies are
kept in the detailed log. In GitHub Actions, expand the **Goose command** and
**Goose tool result** log groups to inspect full commands and returned results.
The display neutralizes workflow-command syntax; `agent.log` retains the original
bytes. Successful tool results stay collapsed in the default view.

Example (illustrative timings):

```text
goose: Checking generated patches because clean preparation now passes.
goose running: developer__shell: scripts/llama-canary-log-command.sh check-family ...
goose still running: developer__shell: ... — 1080s elapsed
canary command: check-family.ABC123 — running; last output 12s ago; log: /.../output.log
canary output: checking shard 37/84
```

A one-minute activity report shows time since the most recent Goose event.
Command snapshots show the latest three lines of the full log even if Goose's
shell command pipes its visible result through `tail`. An idle command remains
running until an exit status is observed. Some tools emit nothing during long
work; an unchanged log is not proof of a hang. Counts and stages appear only
when the underlying command actually prints them.

The console names the full per-command `output.log`. An operator with existing
host access can follow that file while the command runs:

```sh
tail -f /path/printed/by/the/job/output.log
```

The original Goose events are retained as `agent.log` alongside the complete
per-command logs under the run-owned `.deps/llama-canary-state-*` directory.
After upload, download the job's `llama-build-<run>-<attempt>-<pass>-<OS>` artifact;
retention remains 14 days. These paths are local files, not live web links, and
GitHub does not expose the artifact until its upload step runs. This change adds
no live log service or host access permissions.

Events larger than the console projection's 1 MiB limit are omitted from display
with a notice; the original event remains in `agent.log`. Unknown/malformed lines
produce a bounded diagnostic. Tool failures and provider errors remain visible.
Console projection and log-write failure cannot turn an otherwise successful
agent turn into an accepted candidate. Existing process deadlines and independent
verification still own admission.
