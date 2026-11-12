# Following a Llama Upgrade repair

Open **Build exact candidate for distributed certification** in the GitHub
Actions run. The step shows Goose’s normal text output as it runs: its narration,
tool calls, and whichever tool results Goose chooses to print. There are no
additional `goose:` prefixes, activity tickers, or duplicate command summaries.
The repair prompt asks Goose to explain each meaningful action and why it is
needed.

The same output is saved in `agent.log`. Commands run through
`llama-canary-log-command.sh` also retain their complete output and exit status
under `agent-commands/`, even if Goose displays only a short tail. During a job,
an operator with runner access can read those files. After the build evidence is
uploaded, download the `llama-build-<run>-<attempt>-<pass>-<OS>` artifact for
these logs. Artifact retention remains 14 days.

A quiet command may be working or stalled; the console shows only what Goose
prints. Existing timeouts and independent checks still decide whether the
candidate passes. GitHub workflow commands are disabled while Goose output is
streamed so tool output cannot issue runner commands.
