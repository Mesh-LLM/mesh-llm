# Approved readiness contract

The maintainer decision recorded on 2026-09-27 in the root checkout's
`.omo/evidence/maintainer-decisions.md` permits either exact structured readiness
(`event=passive_mode`, `status=ready`, `role=client`) or a top-level string
`message` whose lowercase value contains `client ready`. An absent or nonstring
message cannot veto valid structured readiness. Arrays, objects, numbers,
booleans and null never supply the message alternative.

This deliberately differs from the unchanged legacy script's Python `str`
coercion. The ordinary array, nested value and object-key positives are now
negative. The retained inputs in `stringification.rs` exercise the old escaped
container counterexamples but require rejection, including control, private-use,
quote and backslash cases. The structured-arm positive remains positive.
`approved_readiness.rs` also records the Unicode-version counterexample U+0C5C
as negative. These are approved acceptance differences, not claims that legacy
agrees. Root historical review probes, receipts and Unicode measurements remain
unchanged. The removed private `message_repr.rs` formatter belongs only to this
adapter; no shared Python diagnostic renderer changes.

Depth tests put their nested data in an unrelated field beside a valid string
message so that removing container matching cannot make both boundary tests
pass merely by rejecting every container. Size, raw pre-redaction observation,
LF/EOF, independent streams, strict JSON, last-duplicate-key semantics, direct
string lowercasing and all lifecycle receipt/cleanup requirements are unchanged.
This decision does not approve the separately recorded grammar, framing,
environment or platform differences. No caller cutover is authorized.

## Command interruption

The command owns one signal scope, installed before private-state creation and
kept through supervisor shutdown, output drain and private-state deletion.
Callbacks only publish cancellation through a static atomic; no signal listener
thread, handler-side cleanup or second-interrupt exit is used. Repeated handled
interruptions leave the existing graceful/forced cleanup budgets intact.

Unix handles SIGINT and SIGTERM with sigaction, preserving default or ignored
dispositions for restoration. Before changing either disposition, it queries the
calling thread's inherited signal mask and rejects blocked SIGINT or SIGTERM,
including already-pending signals. Rejection leaves the mask, pending signals and
dispositions untouched and creates no private state or client process. The command
does not take ownership of unblocking a launcher's signals.
An existing custom handler causes registration to
fail before command side effects; it is not replaced or chained. Partial setup
rolls back, and teardown does not overwrite a later replacement handler. This is
a command-boundary contract for the synchronous one-shot xtask executable, whose
dispatch path has no concurrent signal registrants, not a library-wide signal
arbitration API. Overlapping command scopes fail registration.

Windows adds/removes only its own console callback. During the scope it returns
TRUE for CTRL_C_EVENT/CTRL_BREAK_EVENT so the default handler cannot abort cleanup;
other events return FALSE. It does not clear other handlers or toggle the process
ignore-Ctrl-C attribute. Console close/logoff/shutdown and uncatchable termination
are not promised cleanup paths. Native Windows control-event qualification remains
required; macOS PTY tests do not provide it.

The supervisor's admission and receipt remain unchanged. Pre-admission interruption
is Cancelled, including readiness printed only during cleanup. A command interrupt
after admission still preserves Ready/Admitted in diagnostics but makes the command
fail after cleanup, without success stdout. State-deletion and existing lifecycle
failures retain precedence. Handler teardown completes before success publication.
